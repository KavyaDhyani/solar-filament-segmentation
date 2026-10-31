"""Builds notebooks/01_eda.ipynb from the cell sources below (keeps the notebook diff-friendly)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
PKG = {p.name: p.read_text() for p in sorted((ROOT / "src" / "filseg").glob("*.py"))}
BOOTSTRAP = (
    "# Kaggle bootstrap: install missing packages and write the project's filseg package (src/filseg) next to the notebook.\n"
    "import importlib.util, subprocess, sys, platform, os\n"
    "from pathlib import Path\n"
    "missing = [p for m, p in [('pycocotools', 'pycocotools'), ('shapely', 'shapely'), ('cv2', 'opencv-python-headless')] if importlib.util.find_spec(m) is None]\n"
    "if missing:\n"
    "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])\n"
    "if Path('/kaggle/working').exists():\n"
    "    pkg = Path('/kaggle/working/src/filseg'); pkg.mkdir(parents=True, exist_ok=True)\n"
    f"    FILES = {PKG!r}\n"
    "    for name, text in FILES.items():\n"
    "        (pkg / name).write_text(text)\n"
    "print('python', platform.python_version(), '| cpus', os.cpu_count(), '| missing installed:', missing)\n"
)


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# Solar Filament Segmentation 2026: exploratory data analysis

**Goal of this notebook:** understand the MAGFiLO training data and, above all, the
scoring rule well enough to choose a solution design *before* spending GPU time.
Everything here runs on a CPU-only Kaggle kernel (no GPU quota); the heavy passes are
cached in `cache/` so a re-run only redraws the figures.

**Data rule followed here:** only the training set is analysed. The test images are
never opened in this notebook.

## How the leaderboard scores a submission (from the organizers' self-evaluation notebook)

- Each row of the submission is one predicted filament mask (COCO RLE, 2048 x 2048).
- The ground truth is a set of **records**: one record = one annotator's set of
  filaments on one image. An image annotated by 3 experts is scored **3 times**, once
  per annotator, against the same predictions.
- In each record, a (GT, prediction) pair is a hit when IoU > 0.5. Unmatched predictions
  are FP and unmatched GT filaments are FN. Counts are pooled over all records:

$$PQ = \frac{\sum_{hits} IoU}{TP + \tfrac12 FP + \tfrac12 FN} = \underbrace{\overline{IoU}_{TP}}_{SQ} \times \underbrace{\frac{TP}{TP + \tfrac12 FP + \tfrac12 FN}}_{RQ}$$

`src/filseg/metrics.py` reproduces this rule exactly (checked against the notebook's
dense torch implementation in section 0).
""")

code(BOOTSTRAP)

code(r"""
import os, sys, time, json, warnings
from pathlib import Path
from collections import Counter

ON_KAGGLE = Path("/kaggle/working").exists()
ROOT = Path("/kaggle/working") if ON_KAGGLE else (Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ["PYTHONPATH"] = f"{SRC}:{os.environ.get('PYTHONPATH', '')}"   # joblib workers import filseg too

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
import cv2
from scipy import ndimage as ndi
from joblib import Parallel, delayed
from pycocotools import mask as mask_utils

from filseg import data, metrics, solar
from filseg.metrics import PQCounts, pooled, record_counts, record_counts_labelmap

warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 160, "display.max_columns", 30, "display.precision", 3)

CACHE = ROOT / ("cache" if ON_KAGGLE else "dev/cache")
CACHE.mkdir(parents=True, exist_ok=True)
N_JOBS = os.cpu_count() if ON_KAGGLE else 4
print("workers:", N_JOBS)

# Chart style: one recessive grid, thin marks, fixed categorical order.
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4",
         green="#008300", violet="#6250d6", red="#e34948", ink="#0b0b0b", ink2="#52514e",
         grid="#e4e3df", surface="#fcfcfb")
CAT = [C[k] for k in ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]]
plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "xtick.color": C["ink2"], "ytick.color": C["ink2"], "axes.grid": True, "grid.color": C["grid"],
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.prop_cycle": cycler(color=CAT), "lines.linewidth": 2, "legend.frameon": False,
})
SITES = list(data.OBSERVATORIES)
SITE_COLOR = dict(zip(SITES, CAT))
""")

md(r"""
## 0. Load annotations and check the metric implementation
""")

code(r"""
coco = data.load_coco()
rec = data.records_frame(coco)
ann = data.annotations_frame(coco)
rles = data.record_rles(coco)                      # record_id -> list of RLEs
stem_records = rec.groupby("stem")["record_id"].apply(list).to_dict()
stems = sorted(stem_records)
print(f"{len(stems)} images (stems), {len(rec)} records, {len(ann)} annotated filaments")
""")

code(r"""
import torch  # only for the cross-check against the organizers' dense implementation

def official_counts(gt, pred):
    G = torch.from_numpy(data.decode(gt).astype(np.float32)).reshape(len(gt), -1)
    P = torch.from_numpy(data.decode(pred).astype(np.float32)).reshape(len(pred), -1)
    inter = G @ P.T
    union = G.sum(1)[:, None] + P.sum(1)[None] - inter
    iou = torch.where(union == 0, torch.tensor(0.0), inter / union)
    hit = iou > 0.5
    return PQCounts(iou[hit].sum().item(), int(hit.sum()), int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum()))

rng = np.random.default_rng(0)
multi = [s for s in stems if len(stem_records[s]) > 1]
for s in rng.choice(multi, 8, replace=False):
    a, b = stem_records[s][:2]
    ours, ref = record_counts(rles[a], rles[b]), official_counts(rles[a], rles[b])
    m = data.decode(rles[a])
    lab = np.zeros(m.shape[1:], np.int32)
    for i, mi in enumerate(data.decode(rles[b]), 1):
        lab[mi > 0] = i
    assert (ours.tp, ours.fp, ours.fn) == (ref.tp, ref.fp, ref.fn) and abs(ours.sum_iou - ref.sum_iou) < 1e-4
    if data.decode(rles[b]).sum(0).max() <= 1:      # label maps can only hold disjoint predictions
        fast = record_counts_labelmap(m, lab, len(rles[b]))
        assert (fast.tp, fast.fp, fast.fn) == (ref.tp, ref.fp, ref.fn) and abs(fast.sum_iou - ref.sum_iou) < 1e-4
print("RLE and label-map PQ match the official dense implementation on 8 random record pairs")
del torch
""")

md(r"""
## 1. Inventory: images, records, annotators, sites and time
""")

code(r"""
st = rec.groupby("stem").agg(observatory=("observatory", "first"), time=("time", "first"),
                             n_records=("record_id", "size"),
                             mean_filaments=("n_filaments", "mean")).reset_index()
st["year"] = st.time.dt.year
print("records per image:", dict(Counter(st.n_records)))
print("records with zero filaments:", int((rec.n_filaments == 0).sum()))
display(pd.crosstab(st.year, st.observatory, margins=True))
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
yc = st.year.value_counts().sort_index()
ax[0].bar(yc.index.astype(str), yc.values, color=C["blue"], width=0.7)
ax[0].set_title("Images per year"); ax[0].tick_params(axis="x", rotation=45)
hrs = st.assign(hour=st.time.dt.hour)
for s in SITES:
    h = hrs[hrs.observatory == s].hour.value_counts().reindex(range(24), fill_value=0)
    ax[1].plot(h.index, h.values, color=SITE_COLOR[s], label=f"{s} {data.OBSERVATORIES[s]}")
ax[1].set_title("Capture hour (UTC) by site"); ax[1].set_xlabel("hour"); ax[1].legend(fontsize=7)
nf = rec.n_filaments.value_counts().sort_index()
ax[2].bar(nf.index, nf.values, color=C["blue"], width=0.8)
ax[2].set_title("Filaments per record"); ax[2].set_xlabel("filaments")
plt.tight_layout(); plt.show()
print(rec.groupby("observatory").n_filaments.describe()[["mean", "50%", "max"]].T)
""")

md(r"""
**Annotator ids.** A record id prefix such as `010402` looks like batch `0104` + slot `02`.
Below: how batches relate to images and years, and whether the 2- and 3-record images
always come from one batch.
""")

code(r"""
bt = rec.groupby("batch").agg(records=("record_id", "size"), images=("stem", "nunique"),
                              slots=("slot", lambda x: "".join(sorted(set(x)))),
                              years=("time", lambda t: f"{t.dt.year.min()}-{t.dt.year.max()}"),
                              mean_filaments=("n_filaments", "mean"))
display(bt)
mixed = rec.groupby("stem").batch.nunique()
print("images whose records come from more than one batch:", int((mixed > 1).sum()))
print("batches with a single slot:", int((bt.slots.str.len() == 1).sum()), "of", len(bt))
""")

md(r"""
## 2. Filament geometry

`spine_length` is the length of the annotated spine polyline. `area / spine_length`
estimates the mean filament **width** in pixels, which decides how much resolution a
model needs.
""")

code(r"""
ann["width_est"] = ann.area / ann.spine_length.replace(0, np.nan)
ann["long_side"] = ann[["bbox_w", "bbox_h"]].max(axis=1)
ann["aspect"] = ann.spine_length / ann.width_est
ann = ann.merge(rec[["record_id", "annotator", "batch", "observatory", "n_records_of_stem"]], on="record_id")
q = ann[["area", "long_side", "spine_length", "width_est", "aspect", "n_vertices"]].quantile([.05, .25, .5, .75, .95]).T
display(q)
print("category counts:", ann.category_id.map({1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}).value_counts().to_dict())
print("spines with < 2 points:", int((ann.n_spine_points < 2).sum()))
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
ax[0].hist(np.log10(ann.area), bins=60, color=C["blue"])
ax[0].set_title("Filament area (log10 px)"); ax[0].set_xlabel("log10 area")
ax[1].hist(ann.width_est.clip(0, 40), bins=60, color=C["blue"])
ax[1].set_title("Mean width = area / spine length (px)"); ax[1].set_xlabel("px at 2048")
ax[2].hist(ann.long_side.clip(0, 800), bins=60, color=C["blue"])
ax[2].set_title("Bounding-box long side (px)"); ax[2].set_xlabel("px")
plt.tight_layout(); plt.show()
""")

code(r"""
from shapely.geometry import Polygon
valid = [Polygon(np.asarray(a["segmentation"][0]).reshape(-1, 2)).is_valid for a in coco["annotations"]]
print(f"valid (non self-intersecting) polygons: {np.mean(valid):.3f}")
""")

md(r"""
## 3. Images: solar disk, limb darkening, filament contrast (one pass over all images)

Per image: disk centre and radius, limb-darkening profile, noise. The image is flattened
as `contrast = I / median(I at the same r/R)` so that 1 = quiet Sun. Per filament: mean
contrast inside the mask and in a background ring around it, position on the disk.
The same pass runs a **classical baseline** (contrast threshold + connected components)
for a grid of thresholds, scored with the official rule.
""")

code(r"""
T_GRID = [0.80, 0.84, 0.87, 0.90]
A_GRID = [100, 300, 1000]
HBINS = np.linspace(0, 1.6, 321)
OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
RING_IN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
RING_OUT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
S8 = np.ones((3, 3), bool)


def image_pass(stem, recs, rles):
    cv2.setNumThreads(1)
    img = data.load_image(stem)
    disk = solar.fit_disk(img)
    con, rho = solar.flatten(img, disk)
    inside = rho < 0.97
    hp = img.astype(np.float32) - cv2.medianBlur(img, 5).astype(np.float32)
    out = dict(stem=stem, cx=disk.cx, cy=disk.cy, r=disk.r,
               disk_mean=float(img[rho < 1].mean()), disk_p01=float(np.percentile(img[rho < 1], 1)),
               disk_p99=float(np.percentile(img[rho < 1], 99)), off_disk_mean=float(img[rho > 1.05].mean()),
               saturated=float((img[rho < 1] >= 254).mean()), noise=float(hp[rho < 0.9].std()),
               profile=solar.radial_profile(img, rho, 50))
    # per-filament stats (all records)
    fil, union = [], np.zeros(img.shape, bool)
    masks = {r: data.decode(rles[r]).astype(bool) for r in recs}
    for r in recs:
        for k, m in enumerate(masks[r]):
            union |= m
            ys, xs = np.nonzero(m)
            y0, y1, x0, x1 = max(ys.min() - 25, 0), ys.max() + 26, max(xs.min() - 25, 0), xs.max() + 26
            mc = m[y0:y1, x0:x1].astype(np.uint8)
            ring = (cv2.dilate(mc, RING_OUT) > 0) & ~(cv2.dilate(mc, RING_IN) > 0)
            cc = con[y0:y1, x0:x1]
            fil.append(dict(record_id=r, k=k, rho=float(rho[ys, xs].mean()), rho_max=float(rho[ys, xs].max()),
                            x_n=float((xs.mean() - disk.cx) / disk.r), y_n=float((ys.mean() - disk.cy) / disk.r),
                            c_in=float(cc[mc > 0].mean()), c_in_p10=float(np.percentile(cc[mc > 0], 10)),
                            c_ring=float(np.median(cc[ring])) if ring.any() else np.nan,
                            frac_off_disk=float((rho[ys, xs] > 1).mean())))
    # pixel histograms: filament (any record) vs other disk pixels
    h_pos = np.histogram(con[inside & union], HBINS)[0]
    h_neg = np.histogram(con[inside & ~union], HBINS)[0]
    # classical baseline: threshold the contrast map, connected components, minimum area
    base = []
    for t in T_GRID:
        bw = cv2.morphologyEx(((con < t) & inside).astype(np.uint8), cv2.MORPH_OPEN, OPEN)
        lab, n = ndi.label(bw, S8)
        areas = np.bincount(lab.ravel(), minlength=n + 1)
        for a_min in A_GRID:
            keep = areas >= a_min; keep[0] = False
            remap = np.zeros(n + 1, np.int32); remap[keep] = np.arange(1, keep.sum() + 1)
            lab2 = remap[lab]
            for r in recs:
                c = record_counts_labelmap(masks[r], lab2, int(keep.sum()))
                base.append(dict(record_id=r, t=t, a_min=a_min, **vars(c)))
    return out, fil, h_pos, h_neg, base


f_cache = CACHE / "eda_image_pass.pkl"
if f_cache.exists():
    img_stats, fil_stats, H_POS, H_NEG, base_df = pd.read_pickle(f_cache)
else:
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, batch_size=4)(delayed(image_pass)(s, stem_records[s], {r: rles[r] for r in stem_records[s]}) for s in stems)
    img_stats = pd.DataFrame([r[0] for r in res])
    fil_stats = pd.DataFrame([f for r in res for f in r[1]])
    H_POS, H_NEG = sum(r[2] for r in res), sum(r[3] for r in res)
    base_df = pd.DataFrame([b for r in res for b in r[4]])
    pd.to_pickle((img_stats, fil_stats, H_POS, H_NEG, base_df), f_cache)
    print(f"image pass: {time.time() - t0:.0f}s")
img_stats = img_stats.merge(st[["stem", "observatory", "year", "time"]], on="stem")
""")

code(r"""
display(img_stats.groupby("observatory")[["r", "disk_mean", "disk_p01", "disk_p99", "off_disk_mean", "saturated", "noise"]].agg(["median"]).round(3))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
x = (np.arange(50) + 0.5) / 50
for s in SITES:
    P = np.stack(img_stats[img_stats.observatory == s].profile.values)
    ax[0].plot(x, np.nanmedian(P / P[:, :1], 0), color=SITE_COLOR[s], label=s)
ax[0].set_title("Limb darkening: median I(r) / I(centre)"); ax[0].set_xlabel("r / R"); ax[0].legend(fontsize=7, ncol=2)
for i, s in enumerate(SITES):
    v = img_stats[img_stats.observatory == s]
    ax[1].scatter(v.time, v.r, s=8, color=SITE_COLOR[s], label=s)
ax[1].set_title("Disk radius (px) over time"); ax[1].tick_params(axis="x", rotation=30)
ax[2].boxplot([img_stats[img_stats.observatory == s].noise for s in SITES], widths=0.5)
ax[2].set_xticks(range(1, len(SITES) + 1)); ax[2].set_xticklabels(SITES)
ax[2].set_title("High-frequency noise (std of I - median5(I))")
plt.tight_layout(); plt.show()
""")

md(r"""
### How separable are filament pixels by darkness alone?
""")

code(r"""
cen = (HBINS[:-1] + HBINS[1:]) / 2
tpr = np.cumsum(H_POS) / H_POS.sum()          # fraction of filament pixels darker than threshold
fpr = np.cumsum(H_NEG) / H_NEG.sum()
auc = (np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(tpr, fpr)
prec = np.cumsum(H_POS) / np.maximum(np.cumsum(H_POS) + np.cumsum(H_NEG), 1)
print(f"filament pixels are {H_POS.sum() / (H_POS.sum() + H_NEG.sum()):.2%} of disk pixels; pixel AUC of darkness = {auc:.3f}")
for t in [0.8, 0.85, 0.9, 0.95]:
    i = np.searchsorted(cen, t)
    print(f"contrast < {t}: recall {tpr[i]:.2f}, precision {prec[i]:.2f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].plot(cen, H_NEG / H_NEG.sum(), color=C["blue"], label="other disk pixels")
ax[0].plot(cen, H_POS / H_POS.sum(), color=C["orange"], label="filament pixels")
ax[0].set_xlim(0.5, 1.3); ax[0].set_title("Contrast I / I_quiet(r)"); ax[0].legend()
fs = fil_stats.assign(depth=1 - fil_stats.c_in / fil_stats.c_ring)
ax[1].hist(fs.depth.clip(-0.05, 0.3), bins=70, color=C["blue"])
ax[1].set_title("Filament depth = 1 - mean inside / ring background"); ax[1].set_xlabel("depth")
plt.tight_layout(); plt.show()
print("filaments with depth < 0.03 (barely darker than surroundings):", f"{(fs.depth < 0.03).mean():.1%}")
""")

md(r"""
### Where filaments sit on the disk
""")

code(r"""
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].scatter(fil_stats.x_n, -fil_stats.y_n, s=1, alpha=0.3, color=C["blue"])
ax[0].add_patch(plt.Circle((0, 0), 1, fill=False, color=C["ink2"])); ax[0].set_aspect("equal")
ax[0].set_title("Filament centroids (disk units, north up)")
edges = np.linspace(0, 1.05, 22)
cnt = np.histogram(fil_stats.rho, edges)[0]
ring_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
ax[1].bar((edges[:-1] + edges[1:]) / 2, cnt / ring_area / cnt.sum(), width=0.045, color=C["blue"])
ax[1].set_title("Filament density vs r / R (per unit area)"); ax[1].set_xlabel("r / R")
ax[2].hist(-fil_stats.y_n.clip(-1, 1), bins=50, orientation="horizontal", color=C["blue"])
ax[2].set_title("North-south position (sin latitude, approx.)")
plt.tight_layout(); plt.show()
print(f"filaments touching the limb (any pixel r > R): {(fil_stats.frac_off_disk > 0).mean():.1%}; "
      f"centroid beyond 0.9 R: {(fil_stats.rho > 0.9).mean():.1%}")
""")

md(r"""
### A look at the images

One image per site, flattened contrast, with every annotator's polygons drawn in a
different colour. The zoomed panels show the fine structure (barbs) and how annotators
differ in extent and in how they split a filament.
""")

code(r"""
def show_stem(stem, axes, zoom=None):
    img = data.load_image(stem)
    disk = solar.fit_disk(img)
    con, _ = solar.flatten(img, disk)
    recs = stem_records[stem]
    for ax in axes:
        ax.imshow(con, cmap="gray", vmin=0.6, vmax=1.25)
        for j, r in enumerate(recs):
            for m in data.decode(rles[r]):
                for cnt in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                    p = np.vstack([cnt[:, 0], cnt[:1, 0]])
                    ax.plot(p[:, 0], p[:, 1], color=CAT[j], lw=0.7)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].set_title(f"{stem} ({len(recs)} annotators)", fontsize=8)
    if zoom is not None:
        y, x, h = zoom
        axes[1].set_xlim(x, x + h); axes[1].set_ylim(y + h, y)
        axes[1].set_title("zoom", fontsize=8)

pick = []
for s in SITES:
    cand = st[(st.observatory == s) & (st.n_records == 3)].stem.tolist() or st[st.observatory == s].stem.tolist()
    pick.append(cand[len(cand) // 2])
fig, axes = plt.subplots(2, 6, figsize=(18, 6.4))
for j, s in enumerate(pick):
    a = ann[ann.record_id == stem_records[s][0]].sort_values("area").iloc[-1]
    zc = (int(a.bbox_y + a.bbox_h / 2 - 150), int(a.bbox_x + a.bbox_w / 2 - 150), 300)
    show_stem(s, axes[:, j], zoom=zc)
plt.tight_layout(); plt.show()
""")

md(r"""
## 4. Instance structure: how close are neighbouring filaments, and what do resolution and boundary errors cost?

These are **oracle** experiments on the ground truth itself. Each one takes the true masks,
applies one degradation that a real model would suffer, and scores the result against
the same record with the official rule:

1. **Resolution**: downsample each true mask to the model's working size and upsample it
   back (the best a model working at that size can do).
2. **Boundary**: erode or dilate each mask by k pixels.
3. **Semantic + connected components**: take the union of all filaments of a record and
   split it into instances by 8-connected components, at full and reduced resolution.
   This is the ceiling of the simplest pipeline (U-Net + CC).
""")

code(r"""
SCALES = [0.75, 0.5, 0.375, 0.25]
KS = [-3, -2, -1, 1, 2, 3]
UNION_SCALES = [1.0, 0.5, 0.25]
DIL = {k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(k) + 1, 2 * abs(k) + 1)) for k in KS}


def crop_box(m, margin, align=8):
    ys, xs = np.nonzero(m)
    y0 = max((ys.min() - margin) // align * align, 0); x0 = max((xs.min() - margin) // align * align, 0)
    y1 = min(-(-(ys.max() + margin) // align) * align, m.shape[0]); x1 = min(-(-(xs.max() + margin) // align) * align, m.shape[1])
    return y0, y1, x0, x1


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def mask_pass(rid, r_rles):
    cv2.setNumThreads(1)
    masks = data.decode(r_rles)
    n = len(masks)
    lab = np.zeros(masks.shape[1:], np.int32)
    for i, m in enumerate(masks, 1):
        lab[m > 0] = i
    overlap_px = int((masks.sum(0) > 1).sum())
    inst = []
    for i, m in enumerate(masks):
        row = dict(record_id=rid, k=i, area=int(m.sum()))
        y0, y1, x0, x1 = crop_box(m, 24)
        mc = m[y0:y1, x0:x1].astype(np.float32); h, w = mc.shape; gt = mc > 0
        for s in SCALES:
            small = cv2.resize(mc, (max(int(round(w * s)), 1), max(int(round(h * s)), 1)), interpolation=cv2.INTER_AREA)
            up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            row[f"iou_s{s}"] = iou(up, gt) if up.any() else -1.0     # -1: the mask vanished
            row[f"ncc_s{s}"] = ndi.label(up, S8)[1]
        for k in KS:
            op = cv2.dilate if k > 0 else cv2.erode
            pm = op(gt.astype(np.uint8), DIL[k]) > 0
            row[f"iou_b{k}"] = iou(pm, gt) if pm.any() else -1.0
        # gap to the nearest other filament of the same record (capped at 64 px)
        y0, y1, x0, x1 = crop_box(m, 64, 1)
        lc = lab[y0:y1, x0:x1]
        others = (lc > 0) & (lc != i + 1)
        if others.any():
            dist = ndi.distance_transform_edt(lc != i + 1)
            row["gap"] = float(dist[others].min())
        else:
            row["gap"] = np.inf
        inst.append(row)
    # semantic union -> connected components
    union = (masks.sum(0) > 0).astype(np.float32)
    uni = []
    for s in UNION_SCALES:
        if s == 1.0:
            u = union > 0
        else:
            small = cv2.resize(union, (int(2048 * s),) * 2, interpolation=cv2.INTER_AREA)
            u = cv2.resize(small, (2048, 2048), interpolation=cv2.INTER_LINEAR) > 0.5
        cc, ncc = ndi.label(u, S8)
        c = record_counts_labelmap(masks, cc, ncc)
        uni.append(dict(record_id=rid, scale=s, n_cc=ncc, n_gt=n, **vars(c)))
    return inst, uni, dict(record_id=rid, overlap_px=overlap_px)


f_cache = CACHE / "eda_mask_pass.pkl"
if f_cache.exists():
    inst_df, union_df, ovl_df = pd.read_pickle(f_cache)
else:
    t0 = time.time()
    res = Parallel(n_jobs=N_JOBS, batch_size=8)(delayed(mask_pass)(r, rles[r]) for r in rec.record_id)
    inst_df = pd.DataFrame([x for r in res for x in r[0]])
    union_df = pd.DataFrame([x for r in res for x in r[1]])
    ovl_df = pd.DataFrame([r[2] for r in res])
    pd.to_pickle((inst_df, union_df, ovl_df), f_cache)
    print(f"mask pass: {time.time() - t0:.0f}s")
print("records with overlapping GT filaments:", int((ovl_df.overlap_px > 0).sum()), "of", len(ovl_df))
""")

code(r"""
def pq_from_iou(ious):
    # each true filament gives one prediction; IoU <= 0.5 -> FP + FN, vanished (-1) -> FN only
    ious = np.asarray(ious)
    hit = ious > 0.5
    return PQCounts(ious[hit].sum(), int(hit.sum()), int(((ious <= 0.5) & (ious >= 0)).sum()), int((~hit).sum()))

rows = [dict(degradation="none (native 2048)", **pq_from_iou(np.ones(len(inst_df))).as_dict(), mean_iou=1.0)]
for s in SCALES:
    v = inst_df[f"iou_s{s}"]
    rows.append(dict(degradation=f"resize to {int(2048 * s)} and back", **pq_from_iou(v).as_dict(), mean_iou=v.clip(0).mean(),
                     broken=(inst_df[f"ncc_s{s}"] > 1).mean()))
for k in KS:
    v = inst_df[f"iou_b{k}"]
    rows.append(dict(degradation=f"{'dilate' if k > 0 else 'erode'} {abs(k)} px", **pq_from_iou(v).as_dict(), mean_iou=v.clip(0).mean()))
oracle_tab = pd.DataFrame(rows).set_index("degradation")
display(oracle_tab[["pq", "sq", "rq", "mean_iou", "broken"]].round(3))
""")

code(r"""
gaps = inst_df.gap.replace(np.inf, 999)
print("gap to the nearest filament of the same record (px at 2048):")
print({f"<= {g}px": f"{(gaps <= g).mean():.1%}" for g in [1, 2, 4, 8, 16, 32, 64]})
ut = union_df.groupby("scale").apply(lambda d: pd.Series({**pooled(PQCounts(r.sum_iou, r.tp, r.fp, r.fn) for r in d.itertuples()).as_dict(),
                                                          "records_with_merges": (d.n_cc < d.n_gt).mean()}))
print("\nSemantic union -> connected components (oracle masks):")
display(ut.round(3))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].hist(np.log10(gaps.clip(0.5, 999)), bins=60, color=C["blue"])
ax[0].set_title("Gap to nearest filament (log10 px)"); ax[0].set_xlabel("log10 px (3 = none within 64 px)")
bins = [0, 300, 600, 1000, 2000, 4000, 1e9]
inst_df["area_bin"] = pd.cut(inst_df.area, bins)
for s, col in zip([0.5, 0.25], [C["blue"], C["orange"]]):
    g = inst_df.groupby("area_bin", observed=True)[f"iou_s{s}"].apply(lambda v: (v > 0.5).mean())
    ax[1].plot(range(len(g)), g.values, marker="o", color=col, label=f"{int(2048 * s)} px")
ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels([str(i) for i in g.index], rotation=30, fontsize=7)
ax[1].set_title("Share of filaments still matched after resizing"); ax[1].legend(); ax[1].set_xlabel("area (px)")
plt.tight_layout(); plt.show()
""")

md(r"""
## 5. Human agreement: the real ceiling

296 images have 2 or 3 independent annotation records. Scoring one expert's record as a
"prediction" against another expert's record gives the PQ of a perfect human
annotator under this metric.
""")

code(r"""
pair_rows, pair_inst = [], []
for s in stems:
    rs = stem_records[s]
    for a in rs:
        for b in rs:
            if a == b:
                continue
            c = record_counts(rles[b], rles[a])          # a is the "prediction", b the ground truth
            iou_m = metrics.iou_matrix(rles[b], rles[a])
            pair_rows.append(dict(stem=s, pred=a, gt=b, n_pred=len(rles[a]), n_gt=len(rles[b]), **vars(c)))
            for i in range(len(rles[b])):
                row = iou_m[i] if iou_m.shape[1] else np.zeros(1)
                pair_inst.append(dict(stem=s, gt=b, k=i, best_iou=row.max(), n_overlapping=int((row > 0).sum())))
pairs = pd.DataFrame(pair_rows).merge(st[["stem", "observatory"]], on="stem")
pinst = pd.DataFrame(pair_inst)
hh = pooled(PQCounts(r.sum_iou, r.tp, r.fp, r.fn) for r in pairs.itertuples())
print("expert vs expert (all ordered pairs):", {k: round(v, 3) if isinstance(v, float) else v for k, v in hh.as_dict().items()})
display(pairs.groupby("observatory").apply(lambda d: pd.Series(pooled(PQCounts(r.sum_iou, r.tp, r.fp, r.fn) for r in d.itertuples()).as_dict())).round(3))
print("|n_pred - n_gt| between experts on the same image:", pairs.eval("abs(n_pred - n_gt)").describe()[["mean", "50%", "max"]].to_dict())
""")

code(r"""
pinst = pinst.merge(inst_df[["record_id", "k", "area"]].rename(columns={"record_id": "gt"}), on=["gt", "k"])
fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
ax[0].hist(pinst.best_iou, bins=50, color=C["blue"])
ax[0].axvline(0.5, color=C["orange"], lw=1.5, ls="--"); ax[0].set_title("Best IoU of an expert's filament in another expert's record")
g = pinst.groupby(pd.cut(pinst.area, bins), observed=True).best_iou.apply(lambda v: (v > 0.5).mean())
ax[1].plot(range(len(g)), g.values, marker="o", color=C["blue"])
ax[1].set_xticks(range(len(g))); ax[1].set_xticklabels([str(i) for i in g.index], rotation=30, fontsize=7)
ax[1].set_title("Share matched by the other expert, by area"); ax[1].set_ylim(0, 1)
nov = pinst.n_overlapping.clip(0, 4).value_counts().sort_index()
ax[2].bar([str(i) if i < 4 else "4+" for i in nov.index], nov.values, color=C["blue"])
ax[2].set_title("Other expert's filaments touching it (1 = clean, 2+ = split)")
plt.tight_layout(); plt.show()
print(f"matched share: {(pinst.best_iou > 0.5).mean():.2f}; mean IoU when matched: {pinst.best_iou[pinst.best_iou > 0.5].mean():.3f}; "
      f"overlapping but not matched (0 < IoU <= 0.5): {((pinst.best_iou > 0) & (pinst.best_iou <= 0.5)).mean():.2f}")
""")

md(r"""
### Which target should a model learn? Label fusion scored on a held-out expert

For the images with 3 experts, two records are fused into one "prediction" and scored
against the third. Fusion options:

- **single**: one expert's record as is;
- **union-CC**: union of both records, split by connected components;
- **intersection-CC**: pixels both experts marked, split by connected components;
- **agreed-only**: one expert's filaments that the other expert also outlined (IoU > 0.3);
- **matched-union**: filaments matched between the two experts (IoU > 0.3) replaced by
  the union of the pair, unmatched filaments dropped.
""")

code(r"""
def to_lab(masks):
    lab = np.zeros((2048, 2048), np.int32)
    for i, m in enumerate(masks, 1):
        lab[m > 0] = i
    return lab, len(masks)


def fusion_pass(stem, rles):
    cv2.setNumThreads(1)
    rs = sorted(rles)
    M = {r: data.decode(rles[r]).astype(bool) for r in rs}
    out = []
    for held in rs:
        a, b = [r for r in rs if r != held]
        gt = M[held]
        preds = {"single": to_lab(M[a])}
        U = M[a].any(0) | M[b].any(0) if len(M[a]) and len(M[b]) else None
        if U is not None:
            preds["union-CC"] = ndi.label(U, S8)
            preds["intersection-CC"] = ndi.label(cv2.morphologyEx((M[a].any(0) & M[b].any(0)).astype(np.uint8), cv2.MORPH_OPEN, OPEN) > 0, S8)
            iou_ab = metrics.iou_matrix(rles[a], rles[b])
            keep = iou_ab.max(1) > 0.3
            preds["agreed-only"] = to_lab(M[a][keep])
            mu = [M[a][i] | M[b][iou_ab[i].argmax()] for i in np.nonzero(keep)[0]]
            preds["matched-union"] = to_lab(mu)
        for name, (lab, n) in preds.items():
            out.append(dict(stem=stem, held=held, fusion=name, **vars(record_counts_labelmap(gt, lab, n))))
    return out

three = [s for s in stems if len(stem_records[s]) == 3]
f_cache = CACHE / "eda_fusion.pkl"
if f_cache.exists():
    fus = pd.read_pickle(f_cache)
else:
    t0 = time.time()
    fus = pd.DataFrame([x for r in Parallel(n_jobs=N_JOBS)(delayed(fusion_pass)(s, {r: rles[r] for r in stem_records[s]}) for s in three) for x in r])
    fus.to_pickle(f_cache); print(f"fusion pass: {time.time() - t0:.0f}s")
display(fus.groupby("fusion").apply(lambda d: pd.Series(pooled(PQCounts(r.sum_iou, r.tp, r.fp, r.fn) for r in d.itertuples()).as_dict()))
        .sort_values("pq", ascending=False).round(3))
""")

md(r"""
### Which predicted filaments are worth submitting?

A useful identity: adding one prediction **always** adds 0.5 to the PQ denominator
(a hit turns an FN into a TP: +1 - 0.5; a miss adds an FP: +0.5). It adds its IoU to the
numerator only if it hits. So a candidate improves the score when

$$P(\text{hit}) \cdot E[IoU \mid \text{hit}] > \tfrac12 \, PQ_{\text{current}}$$

With PQ around 0.4, a filament must have an expected IoU-weighted hit rate above ~0.2
(averaged over the image's annotators) to be worth submitting. Below, expert-vs-expert
PQ when the "predicting" expert's small or near-limb filaments are dropped shows how
much the score depends on that keep rule.
""")

code(r"""
area_of = inst_df.set_index(["record_id", "k"]).area.to_dict()
rho_of = fil_stats.set_index(["record_id", "k"]).rho.to_dict()
rows = []
for a_min in [0, 200, 400, 700, 1000, 1500, 2500]:
    for rho_max in [1.1, 0.95]:
        tot = PQCounts()
        for r in pairs.itertuples():
            keep = [i for i in range(r.n_pred) if area_of[(r.pred, i)] >= a_min and rho_of[(r.pred, i)] < rho_max]
            tot = tot + record_counts(rles[r.gt], [rles[r.pred][i] for i in keep])
        rows.append(dict(min_area=a_min, centroid_rho_below=rho_max, **tot.as_dict()))
display(pd.DataFrame(rows).round(3))
""")

md(r"""
## 6. Classical baseline: darkness threshold + connected components

Contrast map < t inside 0.97 R, 3x3 opening, 8-connected components, minimum area.
Scored on every training record with the official rule.
""")

code(r"""
bt_tab = base_df.groupby(["t", "a_min"]).apply(lambda d: pd.Series(pooled(PQCounts(r.sum_iou, r.tp, r.fp, r.fn) for r in d.itertuples()).as_dict()))
display(bt_tab.round(3).unstack("a_min")["pq"])
best = bt_tab.pq.idxmax()
print("best:", best, bt_tab.loc[best].round(3).to_dict())
""")

md(r"""
## 7. Time structure and the validation split

Filaments live for days to weeks and the Sun rotates ~13 degrees a day, so images taken
on the same or nearby days show the **same filaments**. Below: how close in time the
training images are, and how well one image's labels score on another image taken the
same day by a different site (no derotation). If this "neighbour label" score is high,
random folds would leak.
""")

code(r"""
st_sorted = st.sort_values("time").reset_index(drop=True)
t_days = st_sorted.time.values.astype("datetime64[s]").astype(np.int64) / 86400
dt_prev = np.diff(t_days)
nn = np.minimum(np.r_[np.inf, dt_prev], np.r_[dt_prev, np.inf])
print({f"another image within {d} day(s)": f"{(nn <= d).mean():.1%}" for d in [0.25, 1, 3, 7, 14]})
for gap in [1, 3, 7]:
    cluster = np.r_[0, np.cumsum(dt_prev > gap)]
    sizes = pd.Series(cluster).value_counts()
    print(f"clusters linking images <= {gap} days apart: {len(sizes)} clusters, largest {sizes.max()} images, "
          f"{(sizes == 1).sum()} singletons")

same_day, tmap = [], st.set_index("stem").time
by_day = st.groupby(st.time.dt.date).stem.apply(list)
for day, ss in by_day.items():
    for i, a in enumerate(ss):
        for b in ss[i + 1:]:
            ra, rb = stem_records[a][0], stem_records[b][0]
            hours = abs((tmap[a] - tmap[b]).total_seconds()) / 3600
            same_day.append(dict(a=a, b=b, hours=hours, **vars(record_counts(rles[rb], rles[ra]))))
sd = pd.DataFrame(same_day)
if len(sd):
    for lo, hi in [(0, 2), (2, 6), (6, 24)]:
        d = sd[(sd.hours >= lo) & (sd.hours < hi)]
        if len(d):
            p = pooled(PQCounts(r.sum_iou, r.tp, r.fp, r.fn) for r in d.itertuples())
            print(f"labels of a same-day image {lo}-{hi} h apart, scored on this image: PQ {p.pq:.3f} ({len(d)} pairs)")
""")

md(r"""
## 8. Annotator styles

Records differ by annotator. The test labels come from the same pool of annotators, so a
model should learn the *average* style rather than one expert's.
""")

code(r"""
asty = rec.merge(ann.groupby("record_id").agg(med_area=("area", "median"), med_vertices=("n_vertices", "median")).reset_index(), on="record_id", how="left")
sty = asty.groupby("annotator").agg(records=("record_id", "size"), mean_filaments=("n_filaments", "mean"),
                                    med_area=("med_area", "median"), med_vertices=("med_vertices", "median"))
display(sty.sort_values("records", ascending=False).head(15).round(1))
# paired comparison on the same images: spread of filament counts between experts of one image
spread = rec[rec.n_records_of_stem > 1].groupby("stem").n_filaments.agg(lambda x: x.max() - x.min())
print("max - min filament count between experts of the same image:", spread.describe()[["mean", "50%", "max"]].to_dict())
""")

md(r"""
## 9. Findings and the resulting design

Numbers below are from Kaggle run Version 1 (CPU, 4 workers, ~27 min).

**Data**
- 707 images, 1154 records (411 images with 1 expert, 145 with 2, 151 with 3), 8199 filaments,
  7.1 filaments per record (1-26). Every record has at least one filament. All polygons are valid,
  and only 5 records have overlapping filaments.
- Six GONG sites, 2011-2022 (mostly 2011-2017). Intensity is already normalised across sites
  (disk mean ~124, 1st-99th percentile 58-162, no saturation). Th has the most noise and the most scattered light.
- **Disk fit fails on ~12% of images** (radius ~1030 px instead of 905-915): hazy frames with a
  bright sky. A robust limb fit (gradient edge + circle fit with a radius prior) is needed before
  using r/R for features or masks.

**Filaments**
- Median area 1,228 px at 2048 (10-90%: 410-4,685); median width 11 px (10-90%: 6.5-21); 90% of centroids lie within 0.81 R and 99% within 0.90 R.
  No annotation crosses the limb. Annotators ignore the limb zone, so predictions there are false positives.
- Filaments are only 0.72% of disk pixels, and darkness alone separates them badly (pixel AUC 0.78).
  The best darkness-threshold baseline scores **PQ 0.083**. Context, shape and learned features are essential.

**What the metric punishes (oracles on the true masks)**

| degradation | PQ |
|---|---|
| work at 1536 and upsample | 0.996 |
| work at 1024 and upsample | 0.977 |
| work at 512 and upsample | 0.919 |
| outline 1 px too thin / too thick | 0.78 / 0.82 |
| outline 2 px too thin / too thick | 0.38 / 0.67 |
| semantic union -> connected components (2048 / 1024 / 512) | 0.997 / 0.969 / 0.838 |

- Resolution is cheap: a 1024 model loses ~2%.
- **Boundary placement is everything**: a 2 px bias destroys most matches because filaments are only ~11 px wide.
  The probability threshold (which acts as a dilation or erosion) must be calibrated against the experts.
- Instance separation is solved by connected components (neighbours are rarely within 8 px: 2%).
  No instance head is needed for separating filaments. The open question is how experts split or merge pieces.

**Humans**
- Expert vs expert: **PQ 0.340** (SQ 0.634, RQ 0.536). Only 54% of one expert's filaments are matched
  by another expert; 29% overlap but reach IoU <= 0.5 (different extent, split or merged).
- Fusion scored on a held-out third expert: matched-union 0.348 > union-CC 0.332 > agreed-only 0.328
  > single expert 0.318 > intersection 0.292. A model should learn the **consensus** (filaments the
  experts agree on) with **generous, union-like outlines**.
- Public out-of-fold scores of ~0.45 from U-Nets are above the human pairwise level. That is consistent:
  a model trained on all experts predicts the average expert, which scores better against each expert than any one expert does.

**Validation**
- 78% of images have another image within 3 days. Same-day labels from another site score PQ 0.17
  when the images are less than 2 h apart, and 0 beyond that (the Sun rotates). Folds are grouped on
  3-day time clusters (329 clusters) to be safe.

**Design that follows**
1. 5-fold GroupKFold on 3-day clusters; every model is judged on pooled out-of-fold PQ with the official rule.
2. Semantic U-Net (pretrained encoder) on the full disk at 1024 (1536 if affordable), trained on the
   **soft consensus target** (mean of the experts' masks), BCE + Dice/Lovasz loss, flips/rot90 and
   photometric/haze augmentation.
3. Upsample probabilities to 2048. Suppress beyond ~0.93 R (robust disk fit). Take connected components,
   then link nearby fragments into one instance (a submitted mask may have several parts).
4. **Keep rule learned for PQ**: adding a prediction always adds 0.5 to the denominator, so keep a component
   only if its expected IoU-weighted hit rate (averaged over experts) exceeds PQ/2. A gradient-boosting
   model on out-of-fold component features (area, mean/max probability, r/R, elongation, contrast) predicts it.
5. Calibrate the outline (threshold, or a per-component dilation) on out-of-fold data, because ±1 px moves PQ by ~20%.
6. Ensemble folds x TTA (flips) to cut run-to-run noise.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
# the Kaggle copy carries the package; the repository copy imports it from src/
kaggle_out = ROOT / "kaggle" / "eda" / "solar-filament-eda.ipynb"
kaggle_out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, kaggle_out)
local_bootstrap = "".join(line for line in BOOTSTRAP.splitlines(keepends=True)
                          if "kaggle/working" not in line and "FILES" not in line and "(pkg / name)" not in line)
for c in nb["cells"]:
    if c["cell_type"] == "code" and c["source"] == BOOTSTRAP.strip("\n"):
        c["source"] = local_bootstrap.replace("Kaggle bootstrap: install missing packages and write the project's filseg package (src/filseg) next to the notebook.",
                                              "Install missing packages (the filseg package is imported from ../src).").strip("\n")
nbf.write(nb, ROOT / "notebooks" / "01_eda.ipynb")
print("wrote", kaggle_out, "and notebooks/01_eda.ipynb")
