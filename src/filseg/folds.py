"""Validation folds grouped in time.

Filaments persist for days, so images taken a few days apart show the same structures.
Images are linked into clusters when they are at most ``gap_days`` apart, and whole
clusters are assigned to folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_clusters(times: pd.Series, gap_days: float = 3.0) -> np.ndarray:
    """Cluster id per row: consecutive images (in time) at most ``gap_days`` apart share a cluster."""
    t = pd.to_datetime(times).to_numpy().astype("datetime64[s]").astype(np.int64) / 86400.0
    order = np.argsort(t, kind="stable")
    cid_sorted = np.r_[0, np.cumsum(np.diff(t[order]) > gap_days)]
    out = np.empty(len(t), dtype=np.int64)
    out[order] = cid_sorted
    return out


def assign_folds(cluster_ids: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Greedy balanced assignment of whole clusters to folds (largest clusters first)."""
    ids, sizes = np.unique(cluster_ids, return_counts=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    ids, sizes = ids[perm], sizes[perm]
    order = np.argsort(-sizes, kind="stable")
    load = np.zeros(n_folds, dtype=np.int64)
    fold_of = {}
    for i in order:
        f = int(np.argmin(load))
        fold_of[ids[i]] = f
        load[f] += sizes[i]
    return np.array([fold_of[c] for c in cluster_ids])
