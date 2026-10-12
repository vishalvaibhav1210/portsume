"""Per-product log-linear demand model shared by the promo and assortment projects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

FEATURES = ["sin1", "cos1", "sin2", "cos2", "trend", "log_rel_price", "promo", "display",
            "lag_depth", "sibling_depth"]


def add_features(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "week"]).copy()
    w = df["week"]
    df["sin1"], df["cos1"] = np.sin(2 * np.pi * w / 52), np.cos(2 * np.pi * w / 52)
    df["sin2"], df["cos2"] = np.sin(4 * np.pi * w / 52), np.cos(4 * np.pi * w / 52)
    df["trend"] = w / 52
    reg = df["product_id"].map(products.set_index("product_id")["regular_price"])
    df["log_rel_price"] = np.log(df["price"] / reg)
    df["promo"] = (df["promo_depth"] > 0).astype(int)
    df["lag_depth"] = df.groupby("product_id")["promo_depth"].shift(1).fillna(0) * (1 - df["promo"])
    sub_week = df.groupby(["subcategory", "week"])["promo_depth"].transform("sum")
    df["sibling_depth"] = sub_week - df["promo_depth"]
    df["log_units"] = np.log1p(df["units"])
    return df.reset_index(drop=True)


def fit_models(df: pd.DataFrame, exclude_oos_neighbours: bool = False) -> dict[int, Ridge]:
    """One ridge regression per product, trained on in-stock weeks.

    With ``exclude_oos_neighbours`` the weeks where a sibling was out of stock are
    also dropped, so the fitted level is free of demand transferred in.
    """
    keep = df["oos"] == 0
    if exclude_oos_neighbours:
        sib_oos = df.groupby(["subcategory", "week"])["oos"].transform("sum") - df["oos"]
        keep &= sib_oos == 0
    models = {}
    for pid, g in df[keep].groupby("product_id"):
        models[pid] = Ridge(alpha=1.0).fit(g[FEATURES], g["log_units"])
    return models


def predict(models, df: pd.DataFrame, **overrides) -> np.ndarray:
    out = np.empty(len(df))
    for pid, idx in df.groupby("product_id").indices.items():
        X = df.iloc[idx][FEATURES].copy()
        for k, v in overrides.items():
            X[k] = v if np.isscalar(v) else np.asarray(v)[idx]
        out[idx] = np.expm1(models[pid].predict(X))
    return np.clip(out, 0, None)
