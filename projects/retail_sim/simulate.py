"""Synthetic grocery retailer used by every project in this repo.

Two datasets come out of one seeded simulation, and the true parameters that
generated them are returned alongside, so each method can be checked against
ground truth rather than only against held-out data.

* ``weekly_sales``: product x week panel at chain level with shelf price,
  promotions (depth + display), within-subcategory cannibalisation, post-promo
  pull-forward dips, seasonality and out-of-stock events with demand transfer.
* ``transactions``: customer-level baskets driven by latent need states, with a
  BG/NBD-style buy-till-you-die purchase process and Gamma-Gamma spend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATEGORIES = {
    "Fresh Produce": ["Fruit", "Vegetables", "Salads"],
    "Dairy": ["Milk", "Cheese", "Yogurt"],
    "Bakery": ["Bread", "Cakes", "Morning Goods"],
    "Meat & Fish": ["Poultry", "Red Meat", "Fish"],
    "Ready Meals": ["Chilled Meals", "Frozen Meals", "Pizza"],
    "Snacks": ["Crisps", "Chocolate", "Biscuits"],
    "Beverages": ["Soft Drinks", "Juice", "Coffee & Tea"],
    "Household": ["Laundry", "Cleaning", "Paper"],
}

NEED_STATES = {
    "Fresh & Healthy": ["Fruit", "Vegetables", "Salads", "Yogurt", "Fish"],
    "Family Stock-up": ["Milk", "Bread", "Poultry", "Biscuits", "Juice"],
    "Convenience": ["Chilled Meals", "Frozen Meals", "Pizza", "Soft Drinks", "Crisps"],
    "Premium Indulgence": ["Cheese", "Cakes", "Chocolate", "Red Meat", "Coffee & Tea"],
    "Home Essentials": ["Laundry", "Cleaning", "Paper", "Morning Goods"],
}

PRODUCTS_PER_SUBCAT = 5
N_WEEKS = 104


@dataclass
class RetailWorld:
    products: pd.DataFrame
    weekly_sales: pd.DataFrame
    customers: pd.DataFrame
    transactions: pd.DataFrame
    n_weeks: int = N_WEEKS


def _products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    cat_elasticity = dict(zip(CATEGORIES, rng.uniform(-2.6, -1.1, len(CATEGORIES))))
    for cat, subcats in CATEGORIES.items():
        for sub in subcats:
            for k in range(PRODUCTS_PER_SUBCAT):
                price = round(float(rng.uniform(0.8, 9.0)), 2)
                rows.append(
                    dict(
                        category=cat,
                        subcategory=sub,
                        product=f"{sub[:3].upper()}-{k + 1:02d}",
                        regular_price=price,
                        unit_cost=round(price * rng.uniform(0.55, 0.78), 2),
                        base_units=float(rng.lognormal(5.0, 0.6)),
                        true_elasticity=cat_elasticity[cat] + rng.normal(0, 0.35),
                        true_promo_boost=rng.uniform(0.10, 0.45),
                        true_display_boost=rng.uniform(0.10, 0.35),
                        true_pull_forward=rng.uniform(0.0, 0.5),
                    )
                )
    df = pd.DataFrame(rows)
    df["product_id"] = np.arange(len(df))
    # Transferability is mostly a subcategory trait (milk is interchangeable, branded
    # chocolate less so) with some product-level brand loyalty on top.
    sub_transfer = {s: rng.uniform(0.15, 0.85) for s in df["subcategory"].unique()}
    df["true_transfer_rate"] = (df["subcategory"].map(sub_transfer)
                                + rng.normal(0, 0.08, len(df))).clip(0.05, 0.95)
    df["true_cannibalisation"] = df["subcategory"].map(
        {s: rng.uniform(0.2, 0.9) for s in df["subcategory"].unique()}
    )
    return df


def _weekly_sales(rng: np.random.Generator, products: pd.DataFrame) -> pd.DataFrame:
    n_p, n_w = len(products), N_WEEKS
    weeks = np.arange(n_w)

    # Category seasonality: annual cycle with category-specific amplitude and phase.
    cat_codes = products["category"].astype("category").cat.codes.to_numpy()
    amp = rng.uniform(0.05, 0.25, cat_codes.max() + 1)[cat_codes]
    phase = rng.uniform(0, 2 * np.pi, cat_codes.max() + 1)[cat_codes]
    season = 1 + amp[:, None] * np.sin(2 * np.pi * weeks[None, :] / 52 + phase[:, None])
    trend = 1 + 0.0015 * weeks[None, :]

    # Shelf price: a few permanent moves of up to +/-15% over two years.
    shelf = np.repeat(products["regular_price"].to_numpy()[:, None], n_w, axis=1)
    for i in range(n_p):
        for _ in range(rng.integers(1, 4)):
            t0 = rng.integers(8, n_w - 8)
            shelf[i, t0:] *= 1 + rng.uniform(-0.15, 0.15)

    # Promotions: ~12% of weeks, at most one product per subcategory-week bias.
    promo = rng.random((n_p, n_w)) < 0.12
    depth = np.where(promo, rng.choice([0.1, 0.2, 0.25, 0.33, 0.4], size=(n_p, n_w)), 0.0)
    display = promo & (rng.random((n_p, n_w)) < 0.4)
    price = shelf * (1 - depth)

    e = products["true_elasticity"].to_numpy()[:, None]
    base = products["base_units"].to_numpy()[:, None]
    demand = (
        base
        * season
        * trend
        * (price / products["regular_price"].to_numpy()[:, None]) ** e
        * np.exp(products["true_promo_boost"].to_numpy()[:, None] * promo)
        * np.exp(products["true_display_boost"].to_numpy()[:, None] * display)
    )

    # Pull-forward: the week after a promo loses a share of demand.
    pf = products["true_pull_forward"].to_numpy()[:, None]
    lag_depth = np.zeros_like(depth)
    lag_depth[:, 1:] = depth[:, :-1]
    demand *= 1 - pf * lag_depth * (~promo)

    # Cannibalisation: promoted siblings in the same subcategory steal demand.
    sub = products["subcategory"].to_numpy()
    canni = products["true_cannibalisation"].to_numpy()[:, None]
    sibling_depth = np.zeros_like(depth)
    for s in np.unique(sub):
        idx = np.where(sub == s)[0]
        total = depth[idx].sum(axis=0)
        sibling_depth[idx] = total[None, :] - depth[idx]
    demand *= np.exp(-canni * sibling_depth)

    # Out-of-stocks: lost demand partly transfers to in-stock siblings.
    oos = (rng.random((n_p, n_w)) < 0.025) & (~promo)
    transfer = products["true_transfer_rate"].to_numpy()
    extra = np.zeros_like(demand)
    for s in np.unique(sub):
        idx = np.where(sub == s)[0]
        for t in np.where(oos[idx].any(axis=0))[0]:
            out = idx[oos[idx, t]]
            avail = idx[~oos[idx, t]]
            if len(avail) == 0:
                continue
            moved = (demand[out, t] * transfer[out]).sum()
            share = demand[avail, t] / demand[avail, t].sum()
            extra[avail, t] += moved * share
    demand = np.where(oos, 0.0, demand + extra)

    units = rng.poisson(demand)
    df = pd.DataFrame(
        {
            "product_id": np.repeat(products["product_id"].to_numpy(), n_w),
            "week": np.tile(weeks, n_p),
            "shelf_price": shelf.ravel().round(2),
            "price": price.ravel().round(2),
            "promo_depth": depth.ravel(),
            "display": display.ravel().astype(int),
            "oos": oos.ravel().astype(int),
            "units": units.ravel(),
        }
    )
    df = df.merge(products[["product_id", "category", "subcategory", "unit_cost"]], on="product_id")
    df["revenue"] = (df["units"] * df["price"]).round(2)
    return df


def _customers_and_transactions(
    rng: np.random.Generator, products: pd.DataFrame, n_customers: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subcats = [s for subs in CATEGORIES.values() for s in subs]
    states = list(NEED_STATES)
    # Each need state: a Dirichlet-ish preference over subcategories.
    prefs = np.full((len(states), len(subcats)), 0.15)
    for k, st in enumerate(states):
        for s in NEED_STATES[st]:
            prefs[k, subcats.index(s)] += 3.0
    prefs /= prefs.sum(axis=1, keepdims=True)

    seg = rng.choice(len(states), size=n_customers, p=[0.22, 0.24, 0.2, 0.14, 0.2])
    # Customers mix missions: a dominant need state plus a Dirichlet share of the others.
    mix = 0.4 * rng.dirichlet(np.full(len(states), 0.5), size=n_customers)
    mix[np.arange(n_customers), seg] += 0.6
    # BG/NBD process (weekly time units).
    lam = rng.gamma(shape=1.6, scale=1 / 3.5, size=n_customers)  # trips per week
    p_drop = rng.beta(1.2, 9.0, size=n_customers)
    # Gamma-Gamma spend: per-customer mean spend.
    nu = rng.gamma(shape=6.0, scale=1 / 0.25, size=n_customers)

    first = rng.integers(0, 52, size=n_customers)  # acquisition week
    sub_items = {s: g["product_id"].to_numpy() for s, g in products.groupby("subcategory")}
    rows = []
    for c in range(n_customers):
        t = float(first[c])
        alive = True
        is_first = True
        while alive and t < N_WEEKS:
            if not is_first:
                t += rng.exponential(1 / lam[c])
                if t >= N_WEEKS:
                    break
            is_first = False
            n_items = max(1, rng.poisson(9))
            mission = rng.choice(len(states), p=mix[c])
            sub_draw = rng.choice(len(subcats), size=n_items, p=prefs[mission])
            spend = rng.gamma(shape=12.0, scale=nu[c] / 12.0)
            weights = rng.dirichlet(np.ones(n_items))
            for s_idx, w in zip(sub_draw, weights):
                ids = sub_items[subcats[s_idx]]
                rows.append((c, round(t, 3), int(ids[rng.integers(len(ids))]),
                             subcats[s_idx], round(spend * w, 2)))
            if rng.random() < p_drop[c]:
                alive = False
    tx = pd.DataFrame(rows, columns=["customer_id", "t_week", "product_id", "subcategory", "spend"])
    tx["trip_id"] = tx.groupby(["customer_id", "t_week"]).ngroup()
    customers = pd.DataFrame(
        {
            "customer_id": np.arange(n_customers),
            "true_need_state": [states[k] for k in seg],
            **{f"true_mix_{st}": mix[:, k] for k, st in enumerate(states)},
            "true_lambda": lam,
            "true_p_drop": p_drop,
            "true_mean_spend": nu,
            "first_week": first,
        }
    )
    return customers, tx


def build_world(seed: int = 7, n_customers: int = 6000) -> RetailWorld:
    rng = np.random.default_rng(seed)
    products = _products(rng)
    weekly = _weekly_sales(rng, products)
    customers, tx = _customers_and_transactions(rng, products, n_customers)
    return RetailWorld(products, weekly, customers, tx)


if __name__ == "__main__":
    w = build_world()
    print(w.products.shape, w.weekly_sales.shape, w.customers.shape, w.transactions.shape)
    print(w.weekly_sales.describe().T[["mean", "min", "max"]])
