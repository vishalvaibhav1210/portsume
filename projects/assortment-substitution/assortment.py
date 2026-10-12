"""Demand transferability from out-of-stock events, used to choose which products to delist.

When a product is out of stock, part of its demand moves to siblings in the
same subcategory and the rest walks out. Measuring that transfer rate from
natural out-of-stock events tells you which "low sellers" are safe to delist
and which ones carry demand nobody else captures.

Run from the ``projects`` folder:  python3 assortment-substitution/assortment.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retail_sim.demand_model import add_features, fit_models, predict  # noqa: E402
from retail_sim.simulate import build_world  # noqa: E402

OUT = Path(__file__).parent / "results"
DELIST_SHARE = 0.15        # range review target: remove 15% of SKUs
MAX_DELIST_PER_SUBCAT = 2  # keep at least 3 products in every subcategory
CANDIDATE_SHARE = 0.35     # tail of the range eligible for delisting
PRIOR_EVENTS = 2.0         # shrinkage strength, in "typical events"


def estimate_transfer(sales: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Transfer rate = extra sibling units / units the OOS product would have sold.

    Both sides come from the per-product demand model, which already accounts for
    price, promotions, cannibalisation and seasonality in the same week, so
    promo-heavy weeks no longer have to be thrown away.
    """
    df = add_features(sales, products)
    models = fit_models(df, exclude_oos_neighbours=True)
    df["expected"] = predict(models, df)
    sub_oos = df.groupby(["subcategory", "week"])["oos"].transform("sum")
    events = df[(df["oos"] == 1) & (sub_oos == 1)]  # exactly one product out in that subcategory-week

    sib = df[(df["oos"] == 0)].set_index(["subcategory", "week"]).sort_index()
    rows = []
    for _, ev in events.iterrows():
        s = sib.loc[(ev["subcategory"], ev["week"])]
        rows.append((ev["product_id"], float((s["units"] - s["expected"]).sum()), float(ev["expected"])))
    ev = pd.DataFrame(rows, columns=["product_id", "gained", "lost"])
    ev = ev.join(products.set_index("product_id")["subcategory"], on="product_id")

    # Ratio of sums per product, shrunk toward the subcategory rate with a prior worth K typical events.
    per = ev.groupby("product_id").agg(gained=("gained", "sum"), lost=("lost", "sum"), n=("lost", "size"),
                                       subcategory=("subcategory", "first"))
    sub = ev.groupby("subcategory")[["gained", "lost"]].sum()
    sub_rate = (sub["gained"] / sub["lost"]).clip(0, 1)
    per["raw"] = per["gained"] / per["lost"]
    k_lost = PRIOR_EVENTS * per["lost"] / per["n"]
    per["shrunk"] = ((per["gained"] + sub_rate[per["subcategory"]].values * k_lost) / (per["lost"] + k_lost)).clip(0, 1)

    out = products.set_index("product_id")[["subcategory", "true_transfer_rate"]].join(per[["raw", "n", "shrunk"]])
    out["shrunk"] = out["shrunk"].fillna(out["subcategory"].map(sub_rate))
    out["n"] = out["n"].fillna(0)
    return out, len(ev)


def pick_delists(score: pd.Series, subcat: pd.Series, n: int, pool: pd.Index) -> list[int]:
    chosen, per_sub = [], {}
    for pid in score.loc[pool].sort_values().index:
        if per_sub.get(subcat[pid], 0) < MAX_DELIST_PER_SUBCAT:
            chosen.append(pid)
            per_sub[subcat[pid]] = per_sub.get(subcat[pid], 0) + 1
        if len(chosen) == n:
            break
    return chosen


def main() -> None:
    OUT.mkdir(exist_ok=True)
    world = build_world()
    products, sales = world.products, world.weekly_sales
    est, n_events = estimate_transfer(sales, products)
    has = est["n"] > 0
    mae = {"raw": float((est.loc[has, "raw"].clip(0, 1) - est.loc[has, "true_transfer_rate"]).abs().mean()),
           "shrunk": float((est.loc[has, "shrunk"] - est.loc[has, "true_transfer_rate"]).abs().mean())}

    clean = sales[(sales["promo_depth"] == 0) & (sales["oos"] == 0) & (sales["week"] >= 78)]
    base = clean.groupby("product_id")["units"].mean()
    price = products.set_index("product_id")["regular_price"]
    est = est.join(base.rename("base_units"))
    est["base_rev"] = est["base_units"] * price
    n_delist = int(round(DELIST_SHARE * len(est)))

    strategies = {
        "lowest_sales": est["base_rev"],
        "transfer_aware": est["base_rev"] * (1 - est["shrunk"]),
        "oracle": est["base_rev"] * (1 - est["true_transfer_rate"]),
    }
    total_rev = est["base_rev"].sum()
    # Range reviews only consider the tail: delist candidates are the bottom 35% of SKUs by sales.
    pool = est["base_rev"].nsmallest(int(CANDIDATE_SHARE * len(est))).index
    results = {}
    for name, score in strategies.items():
        chosen = pick_delists(score, est["subcategory"], n_delist, pool)
        # Scored with TRUE transfer: transferred revenue lands at sibling average price ~ same revenue.
        lost = (est.loc[chosen, "base_rev"] * (1 - est.loc[chosen, "true_transfer_rate"])).sum()
        results[name] = {"revenue_lost_pct": round(float(100 * lost / total_rev), 2),
                         "products_delisted": len(chosen)}

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(est["true_transfer_rate"], est["raw"].clip(-0.5, 1.5), s=12, alpha=0.5, color="#adb5bd",
               label=f"Raw per-product (MAE {mae['raw']:.2f})")
    ax.scatter(est["true_transfer_rate"], est["shrunk"], s=14, color="#2a6f97",
               label=f"Shrunk to subcategory (MAE {mae['shrunk']:.2f})")
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    ax.set_xlabel("True demand transfer rate")
    ax.set_ylabel("Estimated from out-of-stock events")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "transfer_rates.png", dpi=150)

    est.round(3).to_csv(OUT / "transfer_estimates.csv")
    metrics = {"oos_events_used": n_events, "transfer_rate_mae": {k: round(v, 3) for k, v in mae.items()},
               "products_with_events": int((est["n"] > 0).sum()),
               "range_review": results,
               "loss_reduction_vs_lowest_sales_pct": round(float(100 * (1 - results["transfer_aware"]["revenue_lost_pct"]
                                                                       / results["lowest_sales"]["revenue_lost_pct"])), 1)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
