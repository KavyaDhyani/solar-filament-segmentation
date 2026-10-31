"""Candidate filament instances from a probability map, their features, and their PQ value.

PQ bookkeeping used for the keep rule: every submitted instance adds 0.5 to the PQ
denominator whether it hits or not (a hit turns an FN into a TP: +1 - 0.5), and adds its
IoU to the numerator only when IoU > 0.5. Each instance is therefore described by its IoU
against every annotator record of its image, and a selection of instances can be scored
exactly without recomputing masks (instances are disjoint, so a GT filament is hit by at
most one instance).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import data, solar
from .metrics import PQCounts
from .postprocess import S8, upsample


@dataclass(frozen=True)
class CandidateParams:
    t_lo: float = 0.4  # outline threshold
    t_hi: float = 0.7  # a component needs at least one pixel above this
    rho_max: float = 0.95  # components centred beyond this radius are dropped
    link_px: int = 0  # fragments closer than this are merged into one instance
    min_area: int = 50
    weights: tuple = ()  # blend weights over the input probability maps (empty = equal)


def blend(probs: list[np.ndarray], weights: tuple) -> np.ndarray:
    w = np.asarray(weights if weights else [1.0] * len(probs), np.float64)
    out = probs[0] * np.float32(w[0] / w.sum())
    for pm, wi in zip(probs[1:], w[1:], strict=True):
        out += pm * np.float32(wi / w.sum())
    return out


def _seg_max(lab_fg: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n + 1, -np.inf)
    np.maximum.at(out, lab_fg, v)
    return out[1:]


def _seg_min(lab_fg: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n + 1, np.inf)
    np.minimum.at(out, lab_fg, v)
    return out[1:]


def candidates(prob: np.ndarray, rho: np.ndarray, p: CandidateParams) -> tuple[np.ndarray, int, np.ndarray]:
    """Label map (2048 x 2048, 0 = background) of disjoint candidate instances, and the number of
    separate fragments each instance was linked from."""
    lab, n = ndi.label((prob > p.t_lo) & (rho < 1.0), S8)  # experts never annotate off the disk
    if n == 0:
        return lab, 0, np.zeros(0)
    fg = np.flatnonzero(lab)
    lb = lab.ravel()[fg]
    area = np.bincount(lb, minlength=n + 1)[1:]
    peak = _seg_max(lb, prob.ravel()[fg], n)
    r_mean = np.bincount(lb, rho.ravel()[fg], n + 1)[1:] / np.maximum(area, 1)
    keep = np.r_[False, (peak > p.t_hi) & (area >= p.min_area) & (r_mean < p.rho_max)]
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, keep.sum() + 1)
    lab, n = remap[lab], int(keep.sum())
    parts = np.ones(n)
    if p.link_px > 0 and n > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.link_px + 1,) * 2)
        glab, ng = ndi.label(cv2.dilate((lab > 0).astype(np.uint8), k) > 0, S8)
        fg = np.flatnonzero(lab)
        group_of = np.zeros(n + 1, np.int64)
        group_of[lab.ravel()[fg]] = glab.ravel()[fg]  # every fragment lies inside one dilated group
        present = np.zeros(ng + 1, bool)
        present[group_of[1:]] = True
        gmap = np.zeros(ng + 1, np.int32)
        gmap[present] = np.arange(1, present.sum() + 1)
        new_of = gmap[group_of]
        new_of[0] = 0
        lab, n = new_of[lab].astype(np.int32), int(present.sum())
        parts = np.bincount(new_of[1:], minlength=n + 1)[1:].astype(np.float64)
    return lab, n, parts


def features(lab: np.ndarray, n: int, parts: np.ndarray, prob: np.ndarray, rho: np.ndarray, con: np.ndarray | None) -> pd.DataFrame:
    """One row per instance; computed on the foreground pixels only."""
    if n == 0:
        return pd.DataFrame()
    fg = np.flatnonzero(lab)
    lb = lab.ravel()[fg]
    area = np.bincount(lb, minlength=n + 1)[1:].astype(np.float64)

    def mean(v):
        return np.bincount(lb, v, n + 1)[1:] / area

    pv, rv = prob.ravel()[fg].astype(np.float64), rho.ravel()[fg].astype(np.float64)
    p_mean = mean(pv)
    f = dict(
        area=area,
        log_area=np.log(area),
        p_mean=p_mean,
        p_max=_seg_max(lb, pv, n),
        p_std=np.sqrt(np.maximum(mean(pv * pv) - p_mean**2, 0)),
        rho_mean=mean(rv),
        rho_max=_seg_max(lb, rv, n),
        p_frac_hi=mean((pv > 0.7).astype(np.float64)),
    )
    if con is not None:
        cv_ = con.ravel()[fg].astype(np.float64)
        f["con_mean"] = mean(cv_)
        f["con_min"] = _seg_min(lb, cv_, n)
    # shape: second moments, bounding box, number of linked fragments, nearest neighbour
    ys, xs = np.divmod(fg, lab.shape[1])
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)
    cx, cy = mean(xs), mean(ys)
    vxx, vyy, vxy = mean(xs * xs) - cx**2, mean(ys * ys) - cy**2, mean(xs * ys) - cx * cy
    half, disc = (vxx + vyy) / 2, np.sqrt(((vxx - vyy) / 2) ** 2 + vxy**2)
    f["elong"] = np.sqrt(np.maximum(half + disc, 0) / np.maximum(half - disc, 1e-3))
    h = _seg_max(lb, ys, n) - _seg_min(lb, ys, n) + 1
    w = _seg_max(lb, xs, n) - _seg_min(lb, xs, n) + 1
    f["extent"] = area / (h * w)
    f["bbox_long"] = np.maximum(h, w)
    f["parts"] = parts
    d = np.hypot(cx[:, None] - cx[None], cy[:, None] - cy[None])
    np.fill_diagonal(d, np.inf)
    f["nn_dist"] = d.min(1) if n > 1 else np.full(n, 4096.0)
    f["n_in_image"] = np.full(n, n)
    f["image_p_mean"] = np.full(n, float(prob[rho < 1].mean()))
    return pd.DataFrame(f)


def record_ious(lab_flat: np.ndarray, n: int, pred_area: np.ndarray, gt_idx: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(n,) best IoU of each instance with a GT filament of one record, and (n,) index of that GT.

    gt_idx: flat pixel indices of each GT filament of the record.
    """
    if n == 0 or len(gt_idx) == 0:
        return np.zeros(n), np.full(n, -1)
    iou = np.zeros((len(gt_idx), n))
    for g, gi in enumerate(gt_idx):
        inter = np.bincount(lab_flat[gi], minlength=n + 1)[1:]
        iou[g] = inter / np.maximum(len(gi) + pred_area - inter, 1)
    return iou.max(0), iou.argmax(0)


def candidate_job(args: tuple) -> tuple[pd.DataFrame, list[dict]]:
    """args = (stem, prob_u8 or list of prob_u8 (one per model), (cx, cy, r), record_rles or None,
    params list, con_u8 or None).

    Returns the instance table (all params) and, for training images, per-record hit tables.
    """
    stem, probs_u8, (cx, cy, r), record_rles, params_list, con_u8 = args
    cv2.setNumThreads(1)
    probs = [upsample(pm) for pm in (probs_u8 if isinstance(probs_u8, list | tuple) else [probs_u8])]
    rho = solar.radius_map(solar.Disk(cx, cy, r), probs[0].shape)
    con = None if con_u8 is None else upsample(con_u8)
    gts = None
    if record_rles is not None:  # flat pixel indices of every GT filament, per record
        gts = [[np.flatnonzero(m) for m in data.decode(rl)] for rl in record_rles]
    rows, recs, blended = [], [], {}
    for pi, p in enumerate(params_list):
        if p.weights not in blended:
            blended[p.weights] = blend(probs, p.weights)
        prob = blended[p.weights]
        lab, n, parts = candidates(prob, rho, p)
        if n == 0:  # the image's GT filaments must still count as misses
            if gts is not None:
                recs += [dict(stem=stem, param=pi, rec=ri, n_gt=len(g), iou=np.zeros(0), gt=np.zeros(0, int)) for ri, g in enumerate(gts)]
            continue
        f = features(lab, n, parts, prob, rho, con)
        f.insert(0, "inst", np.arange(n))
        f.insert(0, "param", pi)
        f.insert(0, "stem", stem)
        if gts is not None:
            lab_flat = lab.ravel()
            pred_area = np.bincount(lab_flat, minlength=n + 1)[1:]
            q = np.zeros(n)
            for ri, g in enumerate(gts):
                best, which = record_ious(lab_flat, n, pred_area, g)
                q += np.where(best > 0.5, best, 0.0)
                recs.append(dict(stem=stem, param=pi, rec=ri, n_gt=len(g), iou=best, gt=which))
            f["target_q"] = q / len(gts)
        rows.append(f)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), recs


def selection_pq(rec_tables: list[dict], selected: dict) -> PQCounts:
    """Exact pooled PQ of a selection. selected[(stem, param)] = boolean array over that image's instances.

    rec_tables: per (stem, param, record) dicts from ``candidate_job``; images without any candidate
    must still be passed as records with empty arrays so their GT counts as FN.
    """
    tot = PQCounts()
    for r in rec_tables:
        sel = selected.get((r["stem"], r["param"]))
        if sel is None or len(r["iou"]) == 0:
            tot = tot + PQCounts(fn=r["n_gt"])
            continue
        hit = sel & (r["iou"] > 0.5)
        n_hit_gt = len(set(r["gt"][hit].tolist()))
        tot = tot + PQCounts(sum_iou=float(r["iou"][hit].sum()), tp=int(hit.sum()), fp=int((sel & ~(r["iou"] > 0.5)).sum()), fn=r["n_gt"] - n_hit_gt)
    return tot


def encode_instances(lab: np.ndarray, n: int) -> list[str]:
    """Compressed COCO RLE counts strings, one per instance."""
    from pycocotools import mask as mask_utils

    out = []
    for i in range(1, n + 1):
        rle = mask_utils.encode(np.asfortranarray((lab == i).astype(np.uint8)))
        out.append(rle["counts"].decode("ascii"))
    return out
