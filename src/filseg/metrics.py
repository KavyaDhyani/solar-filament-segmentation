"""Panoptic Quality exactly as in the organizers' self-evaluation notebook.

Rules reproduced from that notebook:

- every ground-truth *record* (annotator x image) is scored separately against all
  predictions of its image;
- a (gt, pred) pair is a hit when IoU > 0.5, and every hit counts as a TP;
- a prediction with no hit is a FP, a GT filament with no hit is a FN;
- PQ = sum(IoU of hits) / (TP + 0.5 FP + 0.5 FN), pooled over all records
  (not averaged per image).

IoUs are computed on RLEs with pycocotools, which gives the same numbers as the
notebook's dense matrix product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils

IOU_THRESHOLD = 0.5


@dataclass
class PQCounts:
    sum_iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "PQCounts") -> "PQCounts":
        return PQCounts(self.sum_iou + other.sum_iou, self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def pq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.sum_iou / d if d > 0 else 0.0

    @property
    def sq(self) -> float:
        return self.sum_iou / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else 0.0

    def as_dict(self) -> dict:
        return dict(pq=self.pq, sq=self.sq, rq=self.rq, tp=self.tp, fp=self.fp, fn=self.fn)


def iou_matrix(gt_rles: list[dict], pred_rles: list[dict]) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix."""
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)))
    return np.asarray(mask_utils.iou(pred_rles, gt_rles, [0] * len(gt_rles))).T.reshape(len(gt_rles), len(pred_rles))


def record_counts(gt_rles: list[dict], pred_rles: list[dict]) -> PQCounts:
    """PQ counts of one ground-truth record against the predictions of its image."""
    n_gt, n_pred = len(gt_rles), len(pred_rles)
    if n_gt == 0:
        return PQCounts(fp=n_pred)
    if n_pred == 0:
        return PQCounts(fn=n_gt)
    iou = iou_matrix(gt_rles, pred_rles)
    hit = iou > IOU_THRESHOLD
    return PQCounts(
        sum_iou=float(iou[hit].sum()),
        tp=int(hit.sum()),
        fp=int((hit.sum(axis=0) == 0).sum()),
        fn=int((hit.sum(axis=1) == 0).sum()),
    )


def record_counts_labelmap(gt_masks: np.ndarray, labels: np.ndarray, n_labels: int | None = None) -> PQCounts:
    """Same as ``record_counts`` for disjoint predictions given as a label map (0 = background).

    Much faster than RLE encoding when there are many candidate predictions.
    Labels that do not occur in ``labels`` are not counted as predictions.
    """
    n_labels = int(labels.max()) if n_labels is None else n_labels
    pred_area = np.bincount(labels.ravel(), minlength=n_labels + 1)[1:]
    present = pred_area > 0
    n_gt, n_pred = len(gt_masks), int(present.sum())
    if n_gt == 0:
        return PQCounts(fp=n_pred)
    if n_pred == 0:
        return PQCounts(fn=n_gt)
    iou = np.zeros((n_gt, n_labels))
    for i, g in enumerate(gt_masks):
        g = g.astype(bool)
        inter = np.bincount(labels[g], minlength=n_labels + 1)[1:]
        union = g.sum() + pred_area - inter
        iou[i] = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    iou = iou[:, present]
    hit = iou > IOU_THRESHOLD
    return PQCounts(
        sum_iou=float(iou[hit].sum()),
        tp=int(hit.sum()),
        fp=int((hit.sum(axis=0) == 0).sum()),
        fn=int((hit.sum(axis=1) == 0).sum()),
    )


def pooled(counts) -> PQCounts:
    total = PQCounts()
    for c in counts:
        total = total + c
    return total
