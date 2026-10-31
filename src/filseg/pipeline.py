"""End-to-end inference: fold weights -> test probability maps -> filament instances -> submission.csv.

Reuses the training code paths (``prep``, ``train.SplitData``, ``train.predict_all``) so that the maps
match the out-of-fold maps the post-processing was tuned on, and the post-processing settings and
keep-rule model saved by the post-processing kernel (``postprocess.json``, ``keep_rule.txt``).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import instances, prep, solar
from .instances import CandidateParams
from .model import build_model
from .postprocess import upsample
from .train import SplitData, predict_all, prepare_all, save_split


def load_fold_model(path: Path, arch: str, encoder: str, device: torch.device) -> torch.nn.Module:
    model = build_model(arch, encoder, weights=None).to(device).to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def probability_maps(stems: list[str], split: str, run: dict, weight_dir: Path, device: torch.device, workers: int = 0) -> tuple[np.ndarray, pd.DataFrame]:
    """Fold-averaged probability maps of one network run, uint8 (N, size, size), and the disk table.

    Same arithmetic as ``train.run``: per-fold maps in float16, averaged in float32, stored as uint8.
    """
    import os

    arrays = prepare_all(stems, split, None, run["size"], workers or os.cpu_count())
    with tempfile.TemporaryDirectory() as tmp:
        save_split(arrays, Path(tmp))
        sd = SplitData(Path(tmp), device)
    acc = None
    for k in range(run["n_folds"]):
        model = load_fold_model(weight_dir / f"fold{k}.pt", run["arch"], run["encoder"], device)
        p = predict_all(model, sd, np.arange(len(stems)), run["tta_views"]).astype(np.float16).astype(np.float32)
        acc = p if acc is None else acc + p
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    maps = np.clip(acc / run["n_folds"] * 255 + 0.5, 0, 255).astype(np.uint8)
    return maps, arrays["disk"]


def instances_for_image(probs_u8: list[np.ndarray], disk: tuple[float, float, float], con_u8: np.ndarray,
                        params: CandidateParams, keep_model, feature_columns: list[str], threshold: float) -> tuple[np.ndarray, int]:
    """Blend the runs' maps, build candidates, keep those whose predicted PQ value exceeds the threshold."""
    prob = instances.blend([upsample(p) for p in probs_u8], params.weights)
    rho = solar.radius_map(solar.Disk(*disk), prob.shape)
    lab, n, parts = instances.candidates(prob, rho, params)
    if n == 0:
        return lab, 0
    f = instances.features(lab, n, parts, prob, rho, upsample(con_u8))
    keep = keep_model.predict(f[feature_columns]) > threshold
    remap = np.zeros(n + 1, np.int32)
    remap[1:][keep] = np.arange(1, keep.sum() + 1)
    return remap[lab], int(keep.sum())


def make_submission(stems: list[str], maps: list[np.ndarray], disks: pd.DataFrame, cons: list[np.ndarray], post: dict, keep_model) -> pd.DataFrame:
    """One row per kept filament: ``{stem}_{i}`` and its compressed COCO RLE counts string."""
    params = CandidateParams(**{**post["params"], "weights": tuple(post["params"]["weights"])})
    d = disks.set_index("stem")
    rows = []
    for j, stem in enumerate(stems):
        disk = tuple(float(d.loc[stem, c]) for c in ("cx", "cy", "r"))
        lab, n = instances_for_image([m[j] for m in maps], disk, cons[j], params, keep_model, post["feature_columns"], post["threshold"])
        rows += [(f"{stem}_{i}", counts) for i, counts in enumerate(instances.encode_instances(lab, n), 1)]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def load_post(weights_root: Path):
    import lightgbm as lgb

    post = json.load(open(weights_root / "postprocess.json"))
    return post, lgb.Booster(model_file=str(weights_root / "keep_rule.txt"))


def contrast_maps(stems: list[str], split: str, disks: pd.DataFrame, workers: int = 0) -> list[np.ndarray]:
    """Contrast channel at 1024 for the candidate features (as in the post-processing kernel)."""
    import multiprocessing as mp
    import os

    d = disks.set_index("stem")
    jobs = [(s, split, tuple(float(d.loc[s, c]) for c in ("cx", "cy", "r")), prep.SIZE) for s in stems]
    with mp.get_context("spawn").Pool(workers or os.cpu_count()) as pool:
        return list(pool.imap(prep.contrast_job, jobs, chunksize=4))
