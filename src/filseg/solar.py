"""Solar-disk geometry and limb-darkening correction for full-disk H-alpha images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass
class Disk:
    cx: float
    cy: float
    r: float


def fit_disk(img: np.ndarray, work: int = 512) -> Disk:
    """Disk centre and radius from an Otsu threshold of a downsampled image.

    The radius is taken from the filled disk area, which is robust to dark
    filaments and sunspots inside the disk.
    """
    s = img.shape[0] / work
    small = cv2.resize(img, (work, work), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (5, 5), 0)
    _, bw = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lab, n = ndi.label(bw)
    if n == 0:
        return Disk(img.shape[1] / 2, img.shape[0] / 2, 0.45 * img.shape[0])
    largest = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    largest = ndi.binary_fill_holes(largest)
    ys, xs = np.nonzero(largest)
    return Disk(cx=(xs.mean() + 0.5) * s - 0.5, cy=(ys.mean() + 0.5) * s - 0.5, r=np.sqrt(largest.sum() / np.pi) * s)


def fit_disk_robust(img: np.ndarray, work: int = 512) -> Disk:
    """Limb fit by exhaustive circle search on an edge map; robust to haze, limb rims and halos.

    GONG frames show, besides the limb, a bright rim, a grey halo and a field-stop edge near
    r ~ 1030 px, which defeat thresholding and local edge following. Every circle with its centre
    within +-160 px (at 2048) of the frame centre and a radius of 0.40-0.485 of the frame (the
    solar disk is ~905-920 px) is scored by the mean edge strength along it; the best one wins
    and is refined on a finer grid.
    """
    s = img.shape[0] / work
    small = cv2.GaussianBlur(cv2.resize(img, (work, work), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), 1.5)
    edge = np.hypot(cv2.Sobel(small, cv2.CV_32F, 1, 0), cv2.Sobel(small, cv2.CV_32F, 0, 1))
    ang = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)

    def best_circle(cxs, cys, rs):
        cx, cy = np.meshgrid(cxs, cys, indexing="ij")
        cx, cy = cx.ravel(), cy.ravel()
        best = (-1.0, 0.0, 0.0, 0.0)
        for r in rs:
            xs, ys = cx[:, None] + r * ca, cy[:, None] + r * sa
            sc = ndi.map_coordinates(edge, [ys.ravel(), xs.ravel()], order=1, mode="constant").reshape(xs.shape).mean(1)
            i = int(np.argmax(sc))
            if sc[i] > best[0]:
                best = (float(sc[i]), float(cx[i]), float(cy[i]), float(r))
        return best[1:]

    c0, span = (work - 1) / 2, 0.078 * work
    cx, cy, r = best_circle(np.arange(c0 - span, c0 + span + 0.1, 2.0), np.arange(c0 - span, c0 + span + 0.1, 2.0),
                            np.arange(0.40 * work, 0.485 * work, 1.0))
    cx, cy, r = best_circle(np.arange(cx - 2, cx + 2.01, 0.25), np.arange(cy - 2, cy + 2.01, 0.25), np.arange(r - 1.5, r + 1.51, 0.25))
    return Disk(cx=(cx + 0.5) * s - 0.5, cy=(cy + 0.5) * s - 0.5, r=r * s)


def radius_map(disk: Disk, shape: tuple[int, int]) -> np.ndarray:
    """Distance from the disk centre in units of the disk radius."""
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float32)
    return np.hypot(xx - disk.cx, yy - disk.cy) / disk.r


def radial_profile(img: np.ndarray, rho: np.ndarray, n_bins: int = 100, rho_max: float = 1.0) -> np.ndarray:
    """Median intensity in rings of equal width in rho = r / R."""
    idx = np.clip((rho / rho_max * n_bins).astype(np.int32), 0, n_bins)
    inside = idx < n_bins
    prof = np.full(n_bins, np.nan, np.float32)
    vals, bins = img[inside].astype(np.float32), idx[inside]
    order = np.argsort(bins, kind="stable")
    vals, bins = vals[order], bins[order]
    splits = np.searchsorted(bins, np.arange(n_bins + 1))
    for b in range(n_bins):
        seg = vals[splits[b] : splits[b + 1]]
        if seg.size:
            prof[b] = np.median(seg)
    return prof


def flatten(img: np.ndarray, disk: Disk, n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Divide out the radial limb-darkening profile.

    Returns (contrast, rho): contrast = I / I_profile(rho) inside the disk (1 = quiet Sun,
    < 1 = darker than the local quiet Sun), 1 outside the disk.
    """
    rho = radius_map(disk, img.shape)
    prof = radial_profile(img, rho, n_bins)
    prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
    centres = (np.arange(n_bins) + 0.5) / n_bins
    expected = np.interp(rho, centres, prof).astype(np.float32)
    contrast = np.where(rho < 1.0, img.astype(np.float32) / np.maximum(expected, 1.0), 1.0).astype(np.float32)
    return contrast, rho
