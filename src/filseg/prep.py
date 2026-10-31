"""Per-image preprocessing into the arrays the network trains on.

For each image (at ``size`` x ``size``, default 1024):

- ``img``: the H-alpha image, area-downsampled, uint8;
- ``con``: limb-darkening-corrected contrast I / I_quiet(r), stored as uint8
  (c = CON_LO + u / 255 * CON_SPAN);
- ``tgt``: soft consensus target = share of the image's annotators whose record covers
  the pixel, area-downsampled, uint8 (0..255);
- disk centre/radius (2048-pixel coordinates) and the disk's mean/std for normalisation.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import data, solar

SIZE = 1024
CON_LO, CON_SPAN = 0.4, 1.0


def consensus_target(record_rles: list[list[dict]]) -> np.ndarray:
    """Share of records covering each pixel, float32 (2048 x 2048)."""
    acc = np.zeros((data.H, data.W), np.float32)
    for rles in record_rles:
        acc += data.decode(rles).any(axis=0)
    return acc / max(len(record_rles), 1)


def prepare(stem: str, split: str = "train", record_rles: list[list[dict]] | None = None, size: int = SIZE) -> dict:
    cv2.setNumThreads(1)
    img = data.load_image(stem, split)
    disk = solar.fit_disk_robust(img)
    con, rho = solar.flatten(img, disk)
    inside = rho < 1.0
    out = dict(
        stem=stem,
        cx=disk.cx,
        cy=disk.cy,
        r=disk.r,
        mean=float(img[inside].mean()),
        std=float(max(img[inside].std(), 1.0)),
        img=cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
        con=np.clip((cv2.resize(con, (size, size), interpolation=cv2.INTER_AREA) - CON_LO) / CON_SPAN * 255 + 0.5, 0, 255).astype(np.uint8),
        tgt=None,
    )
    if record_rles is not None:
        tgt = consensus_target(record_rles)
        out["tgt"] = np.clip(cv2.resize(tgt, (size, size), interpolation=cv2.INTER_AREA) * 255 + 0.5, 0, 255).astype(np.uint8)
    return out


def contrast_job(args: tuple) -> np.ndarray:
    """args = (stem, split, (cx, cy, r), size): the stored contrast channel for a known disk (no disk fit)."""
    stem, split, (cx, cy, r), size = args
    cv2.setNumThreads(1)
    con, _ = solar.flatten(data.load_image(stem, split), solar.Disk(cx, cy, r))
    return np.clip((cv2.resize(con, (size, size), interpolation=cv2.INTER_AREA) - CON_LO) / CON_SPAN * 255 + 0.5, 0, 255).astype(np.uint8)


def prepare_job(args: tuple) -> dict:
    """Pool-friendly wrapper: args = (stem, split, record_rles, size)."""
    return prepare(*args)
