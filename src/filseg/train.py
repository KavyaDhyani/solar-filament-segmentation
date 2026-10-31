"""Grouped-fold training of the semantic filament network.

Run end to end with ``run(Config(...))``. Outputs (in ``cfg.out_dir``):

- ``oof_prob.npy``  (N_train, S, S) uint8 out-of-fold probabilities, rows in ``train_stems.json`` order;
- ``test_prob.npy`` (N_test, S, S) uint8 fold-averaged test probabilities, rows in ``test_stems.json`` order;
- ``parts/`` per-fold out-of-fold probabilities and epoch logs (folds run in parallel, one process per GPU);
- ``disks.csv`` disk fits of all images, ``folds.csv``, ``weights/fold{k}.pt`` (EMA, fp16), ``log.json``;
- ``oof_pq_grid.csv`` out-of-fold PQ of a simple post-processing grid.
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import data, folds, postprocess, prep
from .augment import augment, make_input, rho_map
from .model import amp_dtype, build_model, forward_safe, predict_tta


@dataclass
class Config:
    arch: str = "unet"
    encoder: str = "tu-tf_efficientnetv2_s"
    size: int = prep.SIZE
    epochs: int = 40
    batch: int = 8
    train_crop: int = 768  # random crops of the working-size image (full images do not fit a T4 at batch 8)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    ema: float = 0.995
    dice_weight: float = 1.0
    n_folds: int = 5
    fold_gap_days: float = 3.0
    fold_seed: int = 0
    folds_to_run: list[int] | None = None
    seed: int = 0
    tta_views: int = 8
    eval_every: int = 5
    aug: dict = field(default_factory=lambda: dict(
        p_affine=0.7, max_rot=15, max_scale=0.08, max_shift=0.03, gain=0.15, bias=0.15,
        p_blur=0.25, max_blur=1.0, p_noise=0.3, max_noise=0.08, p_haze=0.2, haze_amp=0.3))
    debug: bool = False
    out_dir: str = "/kaggle/working"
    workers: int = 0  # 0 = all CPUs
    init_from: str | None = None  # output folder of an earlier run: skip training, re-predict with its fold weights ("auto" = find it under /kaggle/input)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def loss_fn(logits: torch.Tensor, y: torch.Tensor, dice_weight: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = logits.sigmoid()
    inter = (p * y).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)
    return bce + dice_weight * dice.mean()


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.steps = decay, 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for e, p in zip(self.model.state_dict().values(), model.state_dict().values(), strict=True):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


def prepare_all(stems, split, record_rles_of, size, workers) -> dict:
    jobs = [(s, split, record_rles_of(s) if record_rles_of else None, size) for s in stems]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        items = list(pool.imap(prep.prepare_job, jobs, chunksize=4))
    out = {k: np.stack([it[k] for it in items]) for k in ("img", "con")}
    if record_rles_of:
        out["tgt"] = np.stack([it["tgt"] for it in items])
    out["disk"] = pd.DataFrame([{k: it[k] for k in ("stem", "cx", "cy", "r", "mean", "std")} for it in items])
    return out


class SplitData:
    """uint8 images of one split kept in CPU memory; batches are assembled on the GPU on demand."""

    def __init__(self, folder: Path, device: torch.device):
        self.device = device
        self.img = torch.from_numpy(np.load(folder / "img.npy"))
        self.con = torch.from_numpy(np.load(folder / "con.npy"))
        self.tgt = torch.from_numpy(np.load(folder / "tgt.npy")) if (folder / "tgt.npy").exists() else None
        d = pd.read_csv(folder / "disk.csv")
        f = lambda c: torch.tensor(d[c].values, dtype=torch.float32, device=device)  # noqa: E731
        self.cx, self.cy, self.r, self.mean, self.std = f("cx"), f("cy"), f("r"), f("mean"), f("std")
        self.size = self.img.shape[-1]

    def inputs(self, idx: np.ndarray) -> torch.Tensor:
        j = torch.as_tensor(idx)
        jd = j.to(self.device)
        rho = rho_map(self.cx[jd], self.cy[jd], self.r[jd], self.size)
        return make_input(self.img[j].to(self.device), self.con[j].to(self.device), self.mean[jd], self.std[jd], rho)

    def targets(self, idx: np.ndarray) -> torch.Tensor:
        return self.tgt[torch.as_tensor(idx)].to(self.device).float().unsqueeze(1) / 255.0


def save_split(arrays: dict, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k in ("img", "con", "tgt"):
        if k in arrays:
            np.save(folder / f"{k}.npy", arrays[k])
    arrays["disk"].to_csv(folder / "disk.csv", index=False)


@torch.no_grad()
def predict_all(model, sd: SplitData, idx: np.ndarray, views: int, batch: int = 4) -> np.ndarray:
    out = np.zeros((len(idx), sd.size, sd.size), np.float32)
    for i in range(0, len(idx), batch):
        out[i:i + batch] = predict_tta(model, sd.inputs(idx[i:i + batch]), views).cpu().numpy()
    return out


@torch.no_grad()
def soft_dice(model, sd: SplitData, idx: np.ndarray, batch: int = 8) -> float:
    inter = denom = 0.0
    for i in range(0, len(idx), batch):
        j = idx[i:i + batch]
        p = forward_safe(model, sd.inputs(j).contiguous(memory_format=torch.channels_last)).sigmoid()
        y = sd.targets(j)
        inter += float((p * y).sum())
        denom += float(p.sum() + y.sum())
    return 2 * inter / max(denom, 1e-6)


def train_fold(cfg: Config, fold: int, sd: SplitData, tr_idx: np.ndarray, va_idx: np.ndarray, flog: list) -> torch.nn.Module:
    device = sd.device
    torch.manual_seed(cfg.seed * 100 + fold)
    gen = torch.Generator().manual_seed(cfg.seed * 100 + fold)
    rng = np.random.default_rng(cfg.seed * 100 + fold)
    model = build_model(cfg.arch, cfg.encoder).to(device).to(memory_format=torch.channels_last)
    ema = EMA(model, cfg.ema)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype() == torch.float16)
    steps_per_epoch = max(1, len(tr_idx) // cfg.batch)
    total, warm = cfg.epochs * steps_per_epoch, int(cfg.warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        losses = []
        for i in range(steps_per_epoch):
            j = perm[i * cfg.batch:(i + 1) * cfg.batch]
            x, y = augment(sd.inputs(j), sd.targets(j), gen, cfg.aug)
            x, y = random_crop(x, y, cfg.train_crop, rng)
            with torch.autocast("cuda", dtype=amp_dtype()):
                logits = model(x.contiguous(memory_format=torch.channels_last))
            loss = loss_fn(logits.float(), y, cfg.dice_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            losses.append(loss.item())
        rec = dict(fold=fold, epoch=epoch + 1, loss=float(np.mean(losses)), lr=sched.get_last_lr()[0], minutes=(time.time() - t0) / 60)
        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs:
            rec["val_soft_dice_ema"] = soft_dice(ema.model, sd, va_idx)
        if epoch == 0:
            rec["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        flog.append(rec)
        log(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
    return ema.model


def random_crop(x: torch.Tensor, y: torch.Tensor, crop: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent random square crop per sample (no-op when crop >= image size)."""
    size = x.shape[-1]
    if crop >= size:
        return x, y
    o = rng.integers(0, size - crop + 1, size=(x.shape[0], 2))
    xs = torch.stack([x[i, :, a:a + crop, b:b + crop] for i, (a, b) in enumerate(o)])
    ys = torch.stack([y[i, :, a:a + crop, b:b + crop] for i, (a, b) in enumerate(o)])
    return xs, ys


PREP_DIR = Path("/tmp/filseg_prep")  # local disk, not part of the kernel output


def fold_worker(rank: int, cfg: Config, fold_lists: list[list[int]]) -> None:
    """Trains the folds assigned to one GPU and writes their predictions to out_dir/parts."""
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    out = Path(cfg.out_dir)
    st = pd.read_csv(out / "folds.csv")
    tr, te = SplitData(PREP_DIR / "train", device), SplitData(PREP_DIR / "test", device)
    for k in fold_lists[rank]:
        va_idx = np.nonzero(st.fold.values == k)[0]
        tr_idx = np.nonzero(st.fold.values != k)[0]
        log(f"[gpu {rank}] fold {k}: train {len(tr_idx)}, val {len(va_idx)}")
        flog: list = []
        if cfg.init_from:
            model = build_model(cfg.arch, cfg.encoder, weights=None).to(device).to(memory_format=torch.channels_last).eval()
            model.load_state_dict(torch.load(Path(cfg.init_from) / "weights" / f"fold{k}.pt", map_location=device))
            flog = json.load(open(Path(cfg.init_from) / "parts" / f"log_fold{k}.json"))
            flog[-1]["val_soft_dice_ema"] = soft_dice(model, tr, va_idx)
            log(f"[gpu {rank}] fold {k}: loaded weights, val soft dice {flog[-1]['val_soft_dice_ema']:.4f}")
        else:
            model = train_fold(cfg, k, tr, tr_idx, va_idx, flog)
        np.save(out / "parts" / f"oof_fold{k}.npy", np.clip(predict_all(model, tr, va_idx, cfg.tta_views) * 255 + 0.5, 0, 255).astype(np.uint8))
        np.save(out / "parts" / f"test_fold{k}.npy", predict_all(model, te, np.arange(te.img.shape[0]), cfg.tta_views).astype(np.float16))
        torch.save({n: v.half() if v.dtype.is_floating_point else v for n, v in model.state_dict().items()}, out / "weights" / f"fold{k}.pt")
        json.dump(flog, open(out / "parts" / f"log_fold{k}.json", "w"), indent=1)
        del model
        torch.cuda.empty_cache()


def run(cfg: Config) -> None:
    out = Path(cfg.out_dir)
    if cfg.init_from == "auto":
        import glob
        cfg.init_from = str(Path(sorted(glob.glob("/kaggle/input/**/weights/fold0.pt", recursive=True))[0]).parents[1])
        log(f"re-predicting with the weights in {cfg.init_from}")
    for sub in ("weights", "parts"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    workers = cfg.workers or os.cpu_count()
    coco = data.load_coco()
    rec = data.records_frame(coco)
    rles = data.record_rles(coco)
    stem_records = rec.groupby("stem")["record_id"].apply(list).to_dict()
    st = rec.groupby("stem").time.first().reset_index().sort_values("stem").reset_index(drop=True)
    st["cluster"] = folds.time_clusters(st.time, cfg.fold_gap_days)
    st["fold"] = folds.assign_folds(st.cluster.values, cfg.n_folds, cfg.fold_seed)
    test_stems = sorted(p.stem for p in data.TEST_IMAGES.glob("*.jpeg"))
    if cfg.debug:
        st = st.groupby("fold").head(12).reset_index(drop=True)
        test_stems = test_stems[:4]
    train_stems = st.stem.tolist()
    st.to_csv(out / "folds.csv", index=False)
    json.dump(train_stems, open(out / "train_stems.json", "w"))
    json.dump(test_stems, open(out / "test_stems.json", "w"))
    log(f"train images {len(train_stems)}, test images {len(test_stems)}, fold sizes {st.fold.value_counts().sort_index().tolist()}")

    t0 = time.time()
    tr = prepare_all(train_stems, "train", lambda s: [rles[r] for r in stem_records[s]], cfg.size, workers)
    te = prepare_all(test_stems, "test", None, cfg.size, workers)
    save_split(tr, PREP_DIR / "train")
    save_split(te, PREP_DIR / "test")
    disks = pd.concat([tr["disk"].assign(split="train"), te["disk"].assign(split="test")])
    disks.to_csv(out / "disks.csv", index=False)
    tr = {"disk": tr["disk"]}  # the arrays are on disk now; the GPU workers load their own copies
    r = tr["disk"].r
    log(f"prepared in {(time.time() - t0) / 60:.1f} min; train disk radius min/median/max {r.min():.1f}/{r.median():.1f}/{r.max():.1f}, "
        f"outside 895-925 px: {int(((r < 895) | (r > 925)).sum())}")
    del te

    fold_ids = cfg.folds_to_run if cfg.folds_to_run is not None else list(range(cfg.n_folds))
    n_gpu = max(1, min(torch.cuda.device_count(), len(fold_ids)))
    fold_lists = [fold_ids[i::n_gpu] for i in range(n_gpu)]
    log(f"training folds {fold_lists} on {n_gpu} GPU(s)")
    if n_gpu == 1:
        fold_worker(0, cfg, fold_lists)
    else:
        torch.multiprocessing.spawn(fold_worker, args=(cfg, fold_lists), nprocs=n_gpu, join=True)

    flog = [e for k in fold_ids for e in json.load(open(out / "parts" / f"log_fold{k}.json"))]
    json.dump(dict(config=asdict(cfg), epochs=flog), open(out / "log.json", "w"), indent=1)
    for k in fold_ids:
        last = [e for e in flog if e["fold"] == k][-1]
        log(f"fold {k} final: " + ", ".join(f"{a}={b:.4f}" if isinstance(b, float) else f"{a}={b}" for a, b in last.items()))
    oof = np.zeros((len(train_stems), cfg.size, cfg.size), np.uint8)
    test_sum = None
    for k in fold_ids:
        oof[st.fold.values == k] = np.load(out / "parts" / f"oof_fold{k}.npy")
        tp = np.load(out / "parts" / f"test_fold{k}.npy").astype(np.float32)
        test_sum = tp if test_sum is None else test_sum + tp
    np.save(out / "oof_prob.npy", oof)
    np.save(out / "test_prob.npy", np.clip(test_sum / len(fold_ids) * 255 + 0.5, 0, 255).astype(np.uint8))
    for k in fold_ids:
        (out / "parts" / f"test_fold{k}.npy").unlink()
    log("saved probabilities and weights")

    # quick out-of-fold PQ with a simple post-processing grid (CPU, official matching rule)
    done = np.isin(st.fold.values, fold_ids)
    disk_of = tr["disk"].set_index("stem")
    jobs = [(s, oof[i], tuple(disk_of.loc[s, ["cx", "cy", "r"]]), [rles[r] for r in stem_records[s]],
             [0.3, 0.4, 0.5, 0.6, 0.7], [100, 300, 600, 1000], [0.92, 0.95, 1.0]) for i, s in enumerate(train_stems) if done[i]]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.imap_unordered(postprocess.score_grid_job, jobs, chunksize=2) for r in part]
    tab = postprocess.pooled_table(rows)
    tab.to_csv(out / "oof_pq_grid.csv")
    log("out-of-fold PQ, best 10 settings:\n" + tab.head(10).round(4).to_string())
