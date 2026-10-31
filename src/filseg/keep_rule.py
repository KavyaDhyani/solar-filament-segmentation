"""Learned keep rule: predict each candidate's PQ value and keep the ones worth submitting.

Target per candidate: mean over the image's annotator records of (IoU if IoU > 0.5 else 0).
Since every submitted instance costs 0.5 in the PQ denominator, a candidate helps when its
expected value exceeds PQ / 2; the threshold is swept on out-of-fold predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .instances import selection_pq

NON_FEATURES = {"stem", "param", "inst", "target_q", "fold", "q_hat"}
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)
N_ROUNDS = 400


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def oof_predictions(inst: pd.DataFrame, fold_of_stem: dict) -> np.ndarray:
    """Out-of-fold q_hat with the same image folds as the network."""
    import lightgbm as lgb

    cols = feature_columns(inst)
    fold = inst.stem.map(fold_of_stem).values
    q_hat = np.zeros(len(inst))
    for k in np.unique(fold):
        tr, va = fold != k, fold == k
        m = lgb.train(LGB_PARAMS, lgb.Dataset(inst.loc[tr, cols], inst.loc[tr, "target_q"]), N_ROUNDS)
        q_hat[va] = m.predict(inst.loc[va, cols])
    return q_hat


def fit_full(inst: pd.DataFrame):
    import lightgbm as lgb

    cols = feature_columns(inst)
    return lgb.train(LGB_PARAMS, lgb.Dataset(inst[cols], inst["target_q"]), N_ROUNDS), cols


def selection_from_scores(inst: pd.DataFrame, score: np.ndarray, threshold: float) -> dict:
    keep = score > threshold
    return {key: keep[idx] for key, idx in inst.groupby(["stem", "param"]).indices.items()}


def sweep(inst: pd.DataFrame, recs: list[dict], score: np.ndarray, thresholds) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        c = selection_pq(recs, selection_from_scores(inst, score, t))
        rows.append(dict(threshold=t, **c.as_dict()))
    return pd.DataFrame(rows)
