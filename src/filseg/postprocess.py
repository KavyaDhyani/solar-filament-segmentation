"""From probability maps to filament instances, and out-of-fold scoring with the official rule."""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import data, solar
from .metrics import PQCounts, record_counts_labelmap

S8 = np.ones((3, 3), bool)


def upsample(prob_u8: np.ndarray) -> np.ndarray:
    """uint8 probability map at working size -> float32 at 2048 (bilinear, half-pixel aligned)."""
    return cv2.resize(prob_u8.astype(np.float32) / 255.0, (data.W, data.H), interpolation=cv2.INTER_LINEAR)


def label_components(prob: np.ndarray, rho: np.ndarray, t: float, rho_max: float) -> tuple[np.ndarray, int]:
    return ndi.label((prob > t) & (rho < rho_max), S8)


def filter_area(lab: np.ndarray, n: int, min_area: int) -> tuple[np.ndarray, int]:
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    keep = areas >= min_area
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    return remap[lab], int(keep.sum())


def score_grid_job(args: tuple) -> list[dict]:
    """args = (stem, prob_u8, (cx, cy, r), record_rles_list, t_grid, area_grid, rho_grid).

    Returns pooled-ready counts for every (t, min_area, rho_max) and every record of the image.
    """
    stem, prob_u8, (cx, cy, r), record_rles, t_grid, area_grid, rho_grid = args
    cv2.setNumThreads(1)
    prob = upsample(prob_u8)
    rho = solar.radius_map(solar.Disk(cx, cy, r), prob.shape)
    gts = [data.decode(rl) for rl in record_rles]
    out = []
    for t, rho_max in itertools.product(t_grid, rho_grid):
        lab, n = label_components(prob, rho, t, rho_max)
        for a in area_grid:
            lab2, n2 = filter_area(lab, n, a)
            for gt in gts:
                c = record_counts_labelmap(gt, lab2, n2)
                out.append(dict(stem=stem, t=t, rho_max=rho_max, min_area=a, n_pred=n2, **vars(c)))
    return out


def pooled_table(rows) -> pd.DataFrame:

    df = pd.DataFrame(rows)
    g = df.groupby(["t", "rho_max", "min_area"])[["sum_iou", "tp", "fp", "fn"]].sum()
    c = [PQCounts(*v) for v in g.itertuples(index=False)]
    g["pq"] = [x.pq for x in c]
    g["sq"] = [x.sq for x in c]
    g["rq"] = [x.rq for x in c]
    return g.sort_values("pq", ascending=False)
