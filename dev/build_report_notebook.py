"""Builds the Kaggle CPU kernel (kaggle/report/) that makes the report's figures and uncertainty numbers.

From the out-of-fold maps of both networks and the out-of-fold keep-rule scores (post-processing kernel):
bootstrap confidence intervals over images, per-fold / per-site / per-expert-count PQ, recall by filament
size, PQ against the keep threshold, and example segmentations of training images (out-of-fold).
"""

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
pkg = {p.name: p.read_text() for p in sorted((ROOT / "src" / "filseg").glob("*.py"))}
cells = []


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


cells.append(nbf.v4.new_markdown_cell("# Solar filament segmentation: report figures and uncertainty (out-of-fold, training set only)"))

code(
    "import os, sys, glob, json, time\n"
    "from pathlib import Path\n"
    "pkg = Path('/kaggle/working/src/filseg'); pkg.mkdir(parents=True, exist_ok=True)\n"
    f"FILES = {pkg!r}\n"
    "for name, text in FILES.items():\n"
    "    (pkg / name).write_text(text)\n"
    "sys.path.insert(0, '/kaggle/working/src')\n"
    "os.environ['PYTHONPATH'] = '/kaggle/working/src:' + os.environ.get('PYTHONPATH', '')"
)

code(r"""
import multiprocessing as mp
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from filseg import data, instances, solar
from filseg.instances import CandidateParams
from filseg.postprocess import upsample
warnings.filterwarnings("ignore", category=RuntimeWarning)
FIG = Path("/kaggle/working/figs"); FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "savefig.dpi": 300, "savefig.bbox": "tight"})

def mount(kernel):
    hits = sorted(p for p in glob.glob("/kaggle/input/**/", recursive=True) if Path(p).name == kernel)
    assert hits, kernel
    return Path(hits[0])

RUNS = [mount("solar-filament-predict"), mount("solar-filament-train1536")]
POST = mount("solar-filament-post")
post = json.load(open(POST / "postprocess.json"))
P = CandidateParams(**{**post["params"], "weights": tuple(post["params"]["weights"])})
THR = post["threshold"]
oofs = [np.load(r / "oof_prob.npy", mmap_mode="r") for r in RUNS]
train_stems = json.load(open(RUNS[0] / "train_stems.json"))
folds = pd.read_csv(RUNS[0] / "folds.csv")
fold_of = dict(zip(folds.stem, folds.fold))
disks = pd.read_csv(RUNS[0] / "disks.csv").set_index(["split", "stem"])
cand = pd.read_csv(POST / "oof_candidates_best.csv")
coco = data.load_coco()
rec = data.records_frame(coco)
rles = data.record_rles(coco)
stem_records = rec.groupby("stem")["record_id"].apply(list).to_dict()
print(P, THR, len(cand), "out-of-fold candidates")
""")

code(r"""
def job(i):
    # candidates of the chosen setting on the blended out-of-fold map, with per-record IoU tables and GT areas
    stem = train_stems[i]
    d = disks.loc[("train", stem)]
    prob = instances.blend([upsample(o[i]) for o in oofs], P.weights)
    rho = solar.radius_map(solar.Disk(d.cx, d.cy, d.r), prob.shape)
    lab, n, _ = instances.candidates(prob, rho, P)
    lab_flat = lab.ravel()
    pred_area = np.bincount(lab_flat, minlength=n + 1)[1:]
    out = []
    for r in stem_records[stem]:
        g = [np.flatnonzero(m) for m in data.decode(rles[r])]
        best, which = instances.record_ious(lab_flat, n, pred_area, g)
        out.append(dict(iou=best, gt=which, n_gt=len(g), gt_area=np.array([len(x) for x in g])))
    return stem, n, out

t0 = time.time()
with mp.get_context("fork").Pool(os.cpu_count()) as pool:
    res = pool.map(job, range(len(train_stems)), chunksize=4)
print(f"tables in {(time.time() - t0) / 60:.1f} min")
q_of = {s: g.sort_values("inst").q_hat.values for s, g in cand.groupby("stem")}
for stem, n, _ in res:
    assert len(q_of.get(stem, [])) == n, (stem, n, len(q_of.get(stem, [])))
""")

code(r"""
def image_counts(sel_fn):
    # per image: (sum IoU, TP, FP, FN) pooled over its records, for a selection rule q_hat -> keep mask
    rows = []
    for stem, n, recs in res:
        keep = sel_fn(q_of.get(stem, np.zeros(0))) if n else np.zeros(0, bool)
        c = np.zeros(4)
        for r in recs:
            hit = keep & (r["iou"] > 0.5)
            c += [r["iou"][hit].sum(), hit.sum(), (keep & ~(r["iou"] > 0.5)).sum(), r["n_gt"] - len(set(r["gt"][hit].tolist()))]
        rows.append(c)
    return np.array(rows)

def pq(c):
    c = c.sum(0)
    return c[0] / (c[1] + 0.5 * c[2] + 0.5 * c[3])

learned = image_counts(lambda q: q > THR)
keep_all = image_counts(lambda q: np.ones(len(q), bool))
rng = np.random.default_rng(0)
B = 2000
idx = rng.integers(0, len(res), size=(B, len(res)))
bs_l = np.array([pq(learned[i]) for i in idx]); bs_a = np.array([pq(keep_all[i]) for i in idx])
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
stats = dict(pq_learned=pq(learned), ci_learned=ci(bs_l), pq_keep_all=pq(keep_all), ci_keep_all=ci(bs_a),
             gain=pq(learned) - pq(keep_all), ci_gain=ci(bs_l - bs_a), bootstrap=B)
stems = [s for s, _, _ in res]
sites = np.array([s[-2:] for s in stems]); fl = np.array([fold_of[s] for s in stems])
nexp = np.array([len(stem_records[s]) for s in stems])
stats["per_fold"] = {int(k): pq(learned[fl == k]) for k in range(5)}
stats["per_site"] = {s: pq(learned[sites == s]) for s in sorted(set(sites))}
stats["per_site_ci"] = {s: ci([pq(learned[sites == s][rng.integers(0, (sites == s).sum(), (sites == s).sum())]) for _ in range(500)]) for s in sorted(set(sites))}
stats["per_n_experts"] = {int(k): pq(learned[nexp == k]) for k in (1, 2, 3)}
tot = learned.sum(0)
stats["counts"] = dict(sum_iou=tot[0], tp=int(tot[1]), fp=int(tot[2]), fn=int(tot[3]), sq=tot[0] / tot[1], rq=tot[1] / (tot[1] + 0.5 * tot[2] + 0.5 * tot[3]))
print(json.dumps(stats, indent=1, default=float))
""")

code(r"""
# recall and IoU of matched filaments by ground-truth size (all records pooled)
areas, hits, ious = [], [], []
for stem, n, recs in res:
    keep = (q_of.get(stem, np.zeros(0)) > THR) if n else np.zeros(0, bool)
    for r in recs:
        ok = keep & (r["iou"] > 0.5)
        matched = {int(g): float(v) for g, v in zip(r["gt"][ok], r["iou"][ok])}
        for g, a in enumerate(r["gt_area"]):
            areas.append(a); hits.append(g in matched); ious.append(matched.get(g, np.nan))
areas, hits, ious = np.array(areas), np.array(hits), np.array(ious)
edges = np.quantile(areas, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
b = np.clip(np.searchsorted(edges, areas, side="right") - 1, 0, 4)
size_tab = pd.DataFrame(dict(area_lo=edges[:-1], area_hi=edges[1:], n=np.bincount(b, minlength=5),
                             recall=[hits[b == k].mean() for k in range(5)], mean_iou_matched=[np.nanmean(ious[b == k]) for k in range(5)]))
print(size_tab.round(3).to_string(index=False))
size_tab.to_csv(FIG / "recall_by_size.csv", index=False)
stats["recall_by_size"] = size_tab.to_dict("list")

# PQ against the keep threshold (out-of-fold), with the PQ/2 rule
ts = np.round(np.arange(0.0, 0.5001, 0.01), 3)
curve = [pq(image_counts(lambda q, t=t: q > t)) for t in ts]
stats["threshold_curve"] = dict(t=ts.tolist(), pq=curve)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.2))
ax[0].plot(ts, curve, color="C0")
ax[0].axvline(THR, color="C3", ls="--", lw=0.8, label=f"chosen {THR:.2f}")
ax[0].axvline(max(curve) / 2, color="gray", ls=":", lw=0.8, label=f"PQ/2 = {max(curve) / 2:.2f}")
ax[0].axhline(stats["pq_keep_all"], color="C1", lw=0.8, label=f"keep all {stats['pq_keep_all']:.3f}")
ax[0].set_xlabel("keep threshold on predicted PQ contribution"); ax[0].set_ylabel("out-of-fold PQ"); ax[0].legend(frameon=False, fontsize=6)
x = np.arange(5)
ax[1].bar(x, size_tab.recall, color="C0", label="recall")
ax[1].plot(x, size_tab.mean_iou_matched, "o-", color="C3", ms=3, label="mean IoU of matches")
ax[1].set_xticks(x, [f"{int(a)}-{int(c)}" for a, c in zip(size_tab.area_lo, size_tab.area_hi)], fontsize=6)
ax[1].set_xlabel("expert filament area (px at 2048, quintiles)"); ax[1].set_ylim(0, 1); ax[1].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(FIG / "analysis.pdf")
json.dump(stats, open(FIG / "stats.json", "w"), indent=1, default=float)
""")

code(r"""
# example segmentations: out-of-fold, three sites, images whose PQ is closest to the median (not hand-picked)
img_pq = np.array([c[0] / max(c[1] + 0.5 * c[2] + 0.5 * c[3], 1e-9) for c in learned])
order = np.argsort(np.abs(img_pq - np.median(img_pq)))
chosen, seen = [], set()
for i in order:
    if sites[i] not in seen and nexp[i] >= 2:
        chosen.append(i); seen.add(sites[i])
    if len(chosen) == 3:
        break
print("examples:", [(stems[i], round(float(img_pq[i]), 3), int(nexp[i])) for i in chosen])

def show(ax, im, extent):
    ax.imshow(im, cmap="gray", extent=extent, interpolation="lanczos"); ax.set_xticks([]); ax.set_yticks([])

fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.4))
for row, i in enumerate(chosen):
    stem = stems[i]
    d = disks.loc[("train", stem)]
    img = data.load_image(stem, "train")
    prob = instances.blend([upsample(o[i]) for o in oofs], P.weights)
    rho = solar.radius_map(solar.Disk(d.cx, d.cy, d.r), prob.shape)
    lab, n, _ = instances.candidates(prob, rho, P)
    keep = q_of[stem] > THR
    kept = np.isin(lab, np.flatnonzero(keep) + 1); dropped = ((lab > 0) & ~kept).astype(float)
    experts = [data.decode(rles[r]).any(0).astype(float) for r in stem_records[stem]]
    x0, x1 = int(d.cx - 1.02 * d.r), int(d.cx + 1.02 * d.r); y0, y1 = int(d.cy - 1.02 * d.r), int(d.cy + 1.02 * d.r)
    sl = (slice(max(y0, 0), min(y1, data.H)), slice(max(x0, 0), min(x1, data.W)))
    ext = (sl[1].start, sl[1].stop, sl[0].stop, sl[0].start)
    ax = axes[row, 0]; show(ax, img[sl], ext)
    for k, m in enumerate(experts):
        ax.contour(np.arange(sl[1].start, sl[1].stop), np.arange(sl[0].start, sl[0].stop), m[sl], [0.5], colors=[f"C{k + 1}"], linewidths=0.3)
    ax.set_ylabel(f"{stem[:8]} {stem[-2:]}\nimage PQ {img_pq[i]:.2f}")
    if row == 0: ax.set_title("image + expert outlines")
    ax = axes[row, 1]; ax.imshow(prob[sl], cmap="magma", vmin=0, vmax=1, extent=ext); ax.set_xticks([]); ax.set_yticks([])
    if row == 0: ax.set_title("ensemble probability")
    ax = axes[row, 2]; show(ax, img[sl], ext)
    xs, ys = np.arange(sl[1].start, sl[1].stop), np.arange(sl[0].start, sl[0].stop)
    ax.contour(xs, ys, kept[sl].astype(float), [0.5], colors=["lime"], linewidths=0.3)
    ax.contour(xs, ys, dropped[sl], [0.5], colors=["red"], linewidths=0.3)
    if row == 0: ax.set_title("kept / dropped")
    # zoom on the largest kept instance
    areas_k = np.bincount(lab[kept], minlength=n + 1)
    j = int(np.argmax(areas_k))
    yy, xx = np.nonzero(lab == j)
    cy_, cx_ = int(yy.mean()), int(xx.mean()); h = 160
    zs = (slice(max(cy_ - h, 0), min(cy_ + h, data.H)), slice(max(cx_ - h, 0), min(cx_ + h, data.W)))
    zx, zy = np.arange(zs[1].start, zs[1].stop), np.arange(zs[0].start, zs[0].stop)
    ax = axes[row, 3]; show(ax, img[zs], (zs[1].start, zs[1].stop, zs[0].stop, zs[0].start))
    for k, m in enumerate(experts):
        ax.contour(zx, zy, m[zs], [0.5], colors=[f"C{k + 1}"], linewidths=0.6)
    ax.contour(zx, zy, kept[zs].astype(float), [0.5], colors=["lime"], linewidths=0.8)
    if row == 0: ax.set_title("zoom")
fig.tight_layout(); fig.savefig(FIG / "examples.png", dpi=250)
print(sorted(p.name for p in FIG.iterdir()))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out_dir = ROOT / "kaggle" / "report"
out_dir.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out_dir / "solar-filament-report.ipynb")
meta = {
    "id": "kavyadhyani/solar-filament-report",
    "title": "Solar Filament Report",
    "code_file": "solar-filament-report.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": ["filament-segmentation-2026"],
    "kernel_sources": ["kavyadhyani/solar-filament-predict", "kavyadhyani/solar-filament-train1536", "kavyadhyani/solar-filament-post"],
    "model_sources": [],
}
json.dump(meta, open(out_dir / "kernel-metadata.json", "w"), indent=2)
print("built", out_dir)
