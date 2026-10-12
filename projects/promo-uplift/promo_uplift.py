"""Promotion uplift decomposition: gross lift -> cannibalisation -> pull-forward -> net.

Fits a log-linear demand model per product with seasonality, price, promo and
display effects, a post-promo dip term and a subcategory cannibalisation term.
Each promotion is then decomposed into the units it really added to the
subcategory, and scored on incremental margin.

Run from the ``projects`` folder:  python3 promo-uplift/promo_uplift.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retail_sim.demand_model import FEATURES, add_features, fit_models, predict  # noqa: E402
from retail_sim.simulate import build_world  # noqa: E402

OUT = Path(__file__).parent / "results"
HOLDOUT_WEEKS = 13
def main() -> None:
    OUT.mkdir(exist_ok=True)
    world = build_world()
    products = world.products.set_index("product_id")
    df = add_features(world.weekly_sales, world.products)
    train = df[df["week"] < df["week"].max() - HOLDOUT_WEEKS + 1]
    models = fit_models(train)

    # 1. Forecast accuracy on the 13-week holdout (in-stock weeks only).
    test = df[(df["week"] > df["week"].max() - HOLDOUT_WEEKS) & (df["oos"] == 0)].reset_index(drop=True)
    pred = predict(models, test)
    wape = np.abs(pred - test["units"]).sum() / test["units"].sum()
    naive = test["product_id"].map(train[train["promo"] == 0].groupby("product_id")["units"].mean())
    wape_naive = np.abs(naive - test["units"]).sum() / test["units"].sum()

    # 2. Parameter recovery against the simulator's ground truth.
    coef = pd.DataFrame({pid: m.coef_ for pid, m in models.items()}, index=FEATURES).T
    rec = pd.DataFrame({
        "est_elasticity": coef["log_rel_price"], "true_elasticity": products["true_elasticity"],
        "est_pull_forward": -coef["lag_depth"], "true_pull_forward": products["true_pull_forward"],
        "est_cannibalisation": -coef["sibling_depth"], "true_cannibalisation": products["true_cannibalisation"],
    })
    corr = {k: float(rec[f"est_{k}"].corr(rec[f"true_{k}"]))
            for k in ["elasticity", "pull_forward", "cannibalisation"]}

    # 3. Decompose every promotion into gross lift, cannibalisation, pull-forward and net.
    full = df.copy()
    base = predict(models, full, log_rel_price=np.log(full["shelf_price"] /
                   full["product_id"].map(products["regular_price"])).to_numpy(),
                   promo=0, display=0)
    with_promo = predict(models, full)
    full["baseline"], full["fitted"] = base, with_promo
    events = full[full["promo"] == 1].copy()
    rows = []
    for _, ev in events.iterrows():
        pid, wk, d = ev["product_id"], ev["week"], ev["promo_depth"]
        gross = max(ev["fitted"] - ev["baseline"], 0)
        sibs = full[(full["subcategory"] == ev["subcategory"]) & (full["week"] == wk)
                    & (full["product_id"] != pid)]
        canni = sum(s["baseline"] * (1 - np.exp(-max(-coef.loc[s["product_id"], "sibling_depth"], 0) * d))
                    for _, s in sibs.iterrows())
        nxt = full[(full["product_id"] == pid) & (full["week"] == wk + 1) & (full["promo"] == 0)]
        pull = float(nxt["baseline"].iloc[0] * max(-coef.loc[pid, "lag_depth"], 0) * d) if len(nxt) else 0.0
        reg = products.loc[pid, "regular_price"]
        cost = products.loc[pid, "unit_cost"]
        promo_price = ev["price"]
        margin = (gross - pull) * (promo_price - cost) - ev["baseline"] * (ev["shelf_price"] - promo_price) \
            - canni * (reg - cost)
        rows.append(dict(product_id=pid, week=wk, depth=d, display=ev["display"], gross_units=gross,
                         cannibalised_units=canni, pull_forward_units=pull,
                         net_units=gross - canni - pull, incremental_margin=margin))
    dec = pd.DataFrame(rows)
    dec.to_csv(OUT / "promo_decomposition.csv", index=False)

    by_depth = dec.groupby("depth")[["gross_units", "net_units", "incremental_margin"]].mean().round(1)
    share_negative = float((dec["incremental_margin"] < 0).mean())
    totals = dec[["gross_units", "cannibalised_units", "pull_forward_units", "net_units"]].sum()

    # Waterfall chart of total promo volume.
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = [("Gross lift", totals["gross_units"]), ("Cannibalised", -totals["cannibalised_units"]),
             ("Pull-forward", -totals["pull_forward_units"])]
    running = 0
    for i, (label, v) in enumerate(steps):
        ax.bar(i, v, bottom=running if v < 0 else 0, color="#2a6f97" if v > 0 else "#c8553d")
        running += v
    ax.bar(len(steps), running, color="#4d908e")
    ax.set_xticks(range(len(steps) + 1), [s[0] for s in steps] + ["Net incremental"])
    ax.set_ylim(0, totals["gross_units"] * 1.08)
    ax.set_ylabel("Units across all promotions")
    ax.set_title("Where promotional volume really comes from")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "promo_waterfall.png", dpi=150)

    metrics = {
        "holdout_wape_model": round(float(wape), 3),
        "holdout_wape_naive_mean": round(float(wape_naive), 3),
        "param_recovery_corr": {k: round(v, 3) for k, v in corr.items()},
        "n_promotions": int(len(dec)),
        "net_share_of_gross": round(float(totals["net_units"] / totals["gross_units"]), 3),
        "share_promos_margin_negative": round(share_negative, 3),
        "by_depth": by_depth.reset_index().to_dict(orient="records"),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
