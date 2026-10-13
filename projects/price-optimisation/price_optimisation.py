"""Price elasticity with empirical-Bayes shrinkage, then constrained price optimisation.

Per-product elasticities from shelf-price moves are noisy because each product
only moves price a handful of times. Shrinking them toward their category mean
(weighted by each estimate's standard error) gives more reliable inputs, and the
optimiser that uses them earns more margin once the plan meets true demand.

Run from the ``projects`` folder:  python3 price-optimisation/price_optimisation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retail_sim.simulate import build_world  # noqa: E402

OUT = Path(__file__).parent / "results"
BOUNDS = (0.90, 1.10)          # max +/-10% move per product
MIN_REVENUE_RATIO = 0.99       # keep category revenue within 1% of today
MAX_PRICE_INDEX = 1.02         # sales-weighted category price index cap


def estimate_elasticities(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """OLS of log units on log shelf price + seasonality, regular (non-promo, in-stock) weeks only."""
    clean = sales[(sales["promo_depth"] == 0) & (sales["oos"] == 0) & (sales["units"] > 0)]
    lag = sales.groupby("product_id")["promo_depth"].shift(1).reindex(clean.index).fillna(0)
    clean = clean[lag == 0]
    rows = []
    for pid, g in clean.groupby("product_id"):
        w = g["week"].to_numpy()
        X = np.column_stack([np.ones(len(g)), np.log(g["shelf_price"]), np.sin(2 * np.pi * w / 52),
                             np.cos(2 * np.pi * w / 52), w / 52])
        y = np.log(g["units"].to_numpy())
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = resid @ resid / (len(y) - X.shape[1])
        se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
        rows.append((pid, beta[1], se))
    est = pd.DataFrame(rows, columns=["product_id", "raw", "se"]).merge(
        products[["product_id", "category", "true_elasticity"]], on="product_id")

    # Empirical Bayes: normal-normal shrinkage within category.
    def shrink(g: pd.DataFrame) -> pd.Series:
        mu = np.average(g["raw"], weights=1 / g["se"] ** 2)
        # Robust between-product variance (MAD), so one wild estimate cannot inflate it.
        mad_var = (1.4826 * np.median(np.abs(g["raw"] - np.median(g["raw"])))) ** 2
        tau2 = max(mad_var - np.median(g["se"] ** 2), 0.02)
        w = tau2 / (tau2 + g["se"] ** 2)
        return w * g["raw"] + (1 - w) * mu

    est["shrunk"] = est.groupby("category", group_keys=False)[["raw", "se"]].apply(shrink)
    return est


def optimise_category(base_q, price, cost, elast):
    def units(m):
        return base_q * m ** elast

    def neg_margin(m):
        return -np.sum(units(m) * (price * m - cost))

    rev0 = np.sum(base_q * price)
    cons = [
        {"type": "ineq", "fun": lambda m: np.sum(units(m) * price * m) - MIN_REVENUE_RATIO * rev0},
        {"type": "ineq", "fun": lambda m: MAX_PRICE_INDEX - np.sum(base_q * price * m) / rev0},
    ]
    res = minimize(neg_margin, np.ones(len(price)), method="SLSQP",
                   bounds=[BOUNDS] * len(price), constraints=cons)
    return res.x


def evaluate(plan, base_q, price, cost, elast):
    q = base_q * plan ** elast
    return float(np.sum(q * (price * plan - cost))), float(np.sum(q * price * plan))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    world = build_world()
    products, sales = world.products, world.weekly_sales
    est = estimate_elasticities(sales, products)
    rmse = {k: float(np.sqrt(((est[k] - est["true_elasticity"]) ** 2).mean())) for k in ["raw", "shrunk"]}

    # Current state: last 13 weeks average regular-week units at the latest shelf price.
    last = sales[sales["week"] >= sales["week"].max() - 12]
    cur = last.groupby("product_id").agg(price=("shelf_price", "last"))
    base = sales[(sales["promo_depth"] == 0) & (sales["oos"] == 0) & (sales["week"] >= sales["week"].max() - 25)]
    cur["base_q"] = base.groupby("product_id")["units"].mean()
    cur = cur.join(products.set_index("product_id")[["category", "unit_cost", "true_elasticity"]])
    cur = cur.join(est.set_index("product_id")[["raw", "shrunk"]])

    results, plans = [], {}
    for source in ["raw", "shrunk", "true_elasticity"]:
        tot_m0 = tot_m1 = tot_r0 = tot_r1 = 0.0
        plan_all = pd.Series(1.0, index=cur.index)
        for cat, g in cur.groupby("category"):
            # Clip implausible raw estimates so the optimiser stays in a sane region.
            e_used = np.clip(g[source].to_numpy(), -5, -0.2)
            plan = optimise_category(g["base_q"].to_numpy(), g["price"].to_numpy(),
                                     g["unit_cost"].to_numpy(), e_used)
            plan_all.loc[g.index] = plan
            args = (g["base_q"].to_numpy(), g["price"].to_numpy(), g["unit_cost"].to_numpy(),
                    g["true_elasticity"].to_numpy())
            m0, r0 = evaluate(np.ones(len(g)), *args)
            m1, r1 = evaluate(plan, *args)  # scored against TRUE demand
            tot_m0, tot_m1, tot_r0, tot_r1 = tot_m0 + m0, tot_m1 + m1, tot_r0 + r0, tot_r1 + r1
        plans[source] = plan_all
        results.append(dict(elasticity_source=source,
                            margin_uplift_pct=round(100 * (tot_m1 / tot_m0 - 1), 2),
                            revenue_change_pct=round(100 * (tot_r1 / tot_r0 - 1), 2)))

    pd.DataFrame(plans).round(3).to_csv(OUT / "price_plans.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, k, title in [(axes[0], "raw", "Per-product OLS"), (axes[1], "shrunk", "Empirical-Bayes shrinkage")]:
        ax.scatter(est["true_elasticity"], est[k].clip(-6, 1), s=14, alpha=0.7, color="#2a6f97")
        ax.plot([-4, 0], [-4, 0], color="#999", lw=1, ls="--")
        ax.set_title(f"{title}\nRMSE vs truth = {rmse[k]:.2f}")
        ax.set_xlabel("True elasticity")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Estimated elasticity")
    fig.tight_layout()
    fig.savefig(OUT / "elasticity_shrinkage.png", dpi=150)

    medae = {k: round(float((est[k] - est["true_elasticity"]).abs().median()), 3) for k in ["raw", "shrunk"]}
    wrong_sign = {k: round(float((est[k] >= 0).mean()), 3) for k in ["raw", "shrunk"]}
    metrics = {"elasticity_rmse": {k: round(v, 3) for k, v in rmse.items()},
               "elasticity_median_abs_error": medae, "share_wrong_sign": wrong_sign,
               "optimisation_scored_on_true_demand": results,
               "constraints": {"price_move": BOUNDS, "min_revenue_ratio": MIN_REVENUE_RATIO,
                               "max_price_index": MAX_PRICE_INDEX}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
