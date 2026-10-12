"""Customer lifetime value with BG/NBD (purchase frequency + churn) and Gamma-Gamma (spend).

Both models are fitted by maximum likelihood from scratch (no ``lifetimes``
dependency) on 78 weeks of calibration data, then scored on the following
26 weeks against a naive "past run-rate" baseline.

Run from the ``projects`` folder:  python3 customer-lifetime-value/clv.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retail_sim.simulate import build_world  # noqa: E402

OUT = Path(__file__).parent / "results"
CAL_END, HORIZON = 78.0, 26.0


def rfm_summary(trips: pd.DataFrame) -> pd.DataFrame:
    cal = trips[trips["t_week"] < CAL_END]
    hold = trips[trips["t_week"] >= CAL_END]
    g = cal.groupby("customer_id")
    s = pd.DataFrame({"first": g["t_week"].min(), "last": g["t_week"].max(), "n": g.size(),
                      "cal_spend": g["spend"].sum()})
    s["x"] = s["n"] - 1
    s["t_x"] = s["last"] - s["first"]
    s["T"] = CAL_END - s["first"]
    rep = cal[cal["t_week"] > cal.groupby("customer_id")["t_week"].transform("min")]
    s["m_x"] = rep.groupby("customer_id")["spend"].mean()
    s["hold_trips"] = hold.groupby("customer_id").size().reindex(s.index).fillna(0)
    s["hold_spend"] = hold.groupby("customer_id")["spend"].sum().reindex(s.index).fillna(0)
    return s


def bgnbd_nll(params, x, t_x, T):
    r, alpha, a, b = np.exp(params)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    a3 = -(r + x) * np.log(alpha + T)
    a4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-9)) - (r + x) * np.log(alpha + t_x), -np.inf)
    return -np.sum(a1 + a2 + np.logaddexp(a3, a4))


def bgnbd_expected(params, x, t_x, T, t):
    r, alpha, a, b = params
    hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
    num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
    den = 1 + (x > 0) * a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x)
    return num / den


def gg_nll(params, x, m):
    p, q, g = np.exp(params)
    ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(g) + (p * x - 1) * np.log(m)
          + p * x * np.log(x) - (p * x + q) * np.log(g + m * x))
    return -np.sum(ll)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    world = build_world()
    trips = world.transactions.groupby(["customer_id", "t_week"], as_index=False)["spend"].sum()
    s = rfm_summary(trips)

    bg = minimize(bgnbd_nll, np.log([0.5, 5.0, 1.0, 5.0]), args=(s["x"].values, s["t_x"].values, s["T"].values),
                  method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-6})
    bg_params = np.exp(bg.x)
    s["pred_trips"] = bgnbd_expected(bg_params, s["x"].values, s["t_x"].values, s["T"].values, HORIZON)

    rep = s[s["x"] > 0]
    gg = minimize(gg_nll, np.log([2.0, 2.0, 10.0]), args=(rep["x"].values, rep["m_x"].values), method="Nelder-Mead",
                  options={"maxiter": 4000})
    p, q, g = np.exp(gg.x)
    pop_mean = p * g / (q - 1)
    s["exp_spend"] = np.where(s["x"] > 0, (g + s["m_x"].fillna(0) * s["x"]) * p / (p * s["x"] + q - 1), pop_mean)
    s["pred_value"] = s["pred_trips"] * s["exp_spend"]

    # Baseline: assume each customer keeps their calibration-period spend rate.
    s["naive_value"] = s["cal_spend"] / s["T"].clip(lower=1) * HORIZON

    def scores(col):
        err = np.abs(s[col] - s["hold_spend"]).sum() / s["hold_spend"].sum()
        top = s.nlargest(int(0.1 * len(s)), col)["hold_spend"].sum() / s["hold_spend"].sum()
        return dict(wape=round(float(err), 3), top_decile_capture=round(float(top), 3),
                    aggregate_error_pct=round(float(100 * (s[col].sum() / s["hold_spend"].sum() - 1)), 1),
                    spearman=round(float(s[col].corr(s["hold_spend"], method="spearman")), 3))

    # Calibration chart: actual vs predicted holdout trips by calibration repeat frequency.
    s["freq_bin"] = s["x"].clip(upper=40) // 5 * 5
    agg = s.groupby("freq_bin")[["hold_trips", "pred_trips"]].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(agg.index, agg["hold_trips"], "o-", color="#2a6f97", label="Actual")
    ax.plot(agg.index, agg["pred_trips"], "s--", color="#c8553d", label="BG/NBD predicted")
    ax.set_xlabel("Repeat trips in 78-week calibration window (binned, 40+ capped)")
    ax.set_ylabel("Trips in next 26 weeks")
    ax.set_title("Holdout purchase forecast by customer frequency")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "clv_calibration.png", dpi=150)

    s[["x", "t_x", "T", "pred_trips", "exp_spend", "pred_value", "hold_spend"]].round(3).to_csv(OUT / "clv_scores.csv")
    metrics = {
        "n_customers": int(len(s)),
        "bgnbd_params": dict(zip(["r", "alpha", "a", "b"], np.round(bg_params, 4).tolist())),
        "gamma_gamma_params": dict(p=round(p, 3), q=round(q, 3), gamma=round(g, 3)),
        "holdout_trips_aggregate_error_pct": round(float(100 * (s["pred_trips"].sum() / s["hold_trips"].sum() - 1)), 1),
        "clv_model": scores("pred_value"),
        "naive_run_rate": scores("naive_value"),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
