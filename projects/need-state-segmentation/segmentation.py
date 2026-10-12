"""Customer need-state segmentation with LDA topic modelling on baskets.

Treats each customer's purchase history as a "document" of subcategory
"words", so LDA finds the shopping missions customers mix together rather than
forcing every basket into one hard cluster. Compared with K-means on
subcategory spend shares, and checked against the simulator's true need states.

Run from the ``projects`` folder:  python3 need-state-segmentation/segmentation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retail_sim.simulate import NEED_STATES, build_world  # noqa: E402

OUT = Path(__file__).parent / "results"
K = len(NEED_STATES)
MIN_ITEMS = 20


def main() -> None:
    OUT.mkdir(exist_ok=True)
    world = build_world()
    counts = pd.crosstab(world.transactions["customer_id"], world.transactions["subcategory"])
    counts = counts[counts.sum(axis=1) >= MIN_ITEMS]
    truth = world.customers.set_index("customer_id").loc[counts.index, "true_need_state"]

    lda = LatentDirichletAllocation(n_components=K, learning_method="batch", max_iter=60, random_state=0)
    theta = lda.fit_transform(counts.values)
    lda_label = theta.argmax(axis=1)

    shares = counts.div(counts.sum(axis=1), axis=0)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(StandardScaler().fit_transform(shares))

    ari = {"lda": adjusted_rand_score(truth, lda_label), "kmeans_shares": adjusted_rand_score(truth, km.labels_)}

    # Name each topic by matching it to the true need state it overlaps most (Hungarian assignment).
    states = list(NEED_STATES)
    overlap = pd.crosstab(lda_label, truth).reindex(index=range(K), columns=states, fill_value=0).values
    rows, cols = linear_sum_assignment(-overlap)
    names = {r: states[c] for r, c in zip(rows, cols)}
    accuracy = overlap[rows, cols].sum() / overlap.sum()

    topics = pd.DataFrame(lda.components_, columns=counts.columns)
    topics = topics.div(topics.sum(axis=1), axis=0)
    topics.index = [names[i] for i in range(K)]
    top_words = {n: topics.loc[n].nlargest(5).index.tolist() for n in topics.index}

    # Mission mix recovery: LDA gives each customer a share per need state; K-means cannot.
    cust = world.customers.set_index("customer_id").loc[counts.index]
    true_mix = cust[[f"true_mix_{st}" for st in states]].to_numpy()
    est_mix = np.zeros_like(true_mix)
    for topic, state in names.items():
        est_mix[:, states.index(state)] = theta[:, topic]
    mix_mae = float(np.abs(est_mix - true_mix).mean())
    mix_corr = float(np.corrcoef(est_mix.ravel(), true_mix.ravel())[0, 1])
    mixed_share = float((np.sort(est_mix, axis=1)[:, -2] >= 0.2).mean())

    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(topics.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(K), topics.index)
    ax.set_xticks(range(topics.shape[1]), topics.columns, rotation=60, ha="right", fontsize=8)
    ax.set_title("Need states discovered by LDA: share of each subcategory")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "need_states.png", dpi=150)

    profile = pd.Series([names[i] for i in lda_label]).value_counts(normalize=True).round(3).to_dict()
    metrics = {
        "n_customers": int(len(counts)),
        "adjusted_rand_index": {k: round(float(v), 3) for k, v in ari.items()},
        "lda_assignment_accuracy": round(float(accuracy), 3),
        "mission_mix_mae": round(mix_mae, 3),
        "mission_mix_corr": round(mix_corr, 3),
        "share_customers_second_mission_20pct_plus": round(mixed_share, 3),
        "segment_sizes": profile,
        "top_subcategories": top_words,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
