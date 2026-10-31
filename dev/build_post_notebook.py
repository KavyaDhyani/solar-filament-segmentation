"""Builds the CPU post-processing notebook (kaggle/post/): candidates -> learned keep rule -> submission.csv.

It reads the probability maps of one or more network kernels (``--source``, repeatable) and blends them.
"""

import argparse
import json
from pathlib import Path

import nbformat as nbf

ap = argparse.ArgumentParser()
ap.add_argument("--source", action="append", default=None,
                help="kernel whose oof_prob.npy / test_prob.npy is an ensemble member (repeatable)")
args = ap.parse_args()
SOURCES = args.source or ["kavyadhyani/solar-filament-predict"]

ROOT = Path(__file__).resolve().parents[1]
pkg = {p.name: p.read_text() for p in sorted((ROOT / "src" / "filseg").glob("*.py"))}
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# Solar filament segmentation: post-processing and submission

Input: out-of-fold and test probability maps from one or more network kernels (same image folds).

1. Candidate instances for each model and their blend, over a grid of outline thresholds and
   fragment-linking distances.
2. Per candidate: features, and its PQ value against every annotator record (IoU if > 0.5).
3. A LightGBM model predicts that value out-of-fold (same image folds as the network); candidates
   are kept above a threshold swept on out-of-fold PQ (official matching rule, exact).
4. The best setting is refit on all training candidates and applied to the test maps -> `submission.csv`.
""")

code(
    "import os, sys, json, glob, time, platform\n"
    "from pathlib import Path\n"
    "pkg = Path('/kaggle/working/src/filseg'); pkg.mkdir(parents=True, exist_ok=True)\n"
    f"FILES = {pkg!r}\n"
    "for name, text in FILES.items():\n"
    "    (pkg / name).write_text(text)\n"
    "sys.path.insert(0, '/kaggle/working/src')\n"
    "os.environ['PYTHONPATH'] = '/kaggle/working/src:' + os.environ.get('PYTHONPATH', '')\n"
    "print('python', platform.python_version(), '| cpus', os.cpu_count())"
)

code(r"""
import multiprocessing as mp
import warnings
import numpy as np
import pandas as pd
from filseg import data, prep, instances, keep_rule
from filseg.instances import CandidateParams
from filseg.metrics import PQCounts
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.precision", 4)

RUNS = sorted(Path(p).parent for p in glob.glob("/kaggle/input/**/oof_prob.npy", recursive=True))
print("network runs:", [str(r) for r in RUNS])
cfgs = [json.load(open(r / "log.json"))["config"] for r in RUNS]
for r, c in zip(RUNS, cfgs):
    print(f"  {r.name}: {c['arch']} / {c['encoder']} / size {c['size']} / {c['epochs']} epochs / seed {c['seed']} / debug={c['debug']}")
    assert not c["debug"], "debug run mounted"
TRAIN_OUT = RUNS[0]
oofs = [np.load(r / "oof_prob.npy", mmap_mode="r") for r in RUNS]
test_probs = [np.load(r / "test_prob.npy", mmap_mode="r") for r in RUNS]
train_stems = json.load(open(TRAIN_OUT / "train_stems.json"))
test_stems = json.load(open(TRAIN_OUT / "test_stems.json"))
folds = pd.read_csv(TRAIN_OUT / "folds.csv")
for r in RUNS[1:]:  # blending is only valid on identical images and folds
    assert json.load(open(r / "train_stems.json")) == train_stems and json.load(open(r / "test_stems.json")) == test_stems
    assert (pd.read_csv(r / "folds.csv").fold.values == folds.fold.values).all()
disks = pd.read_csv(TRAIN_OUT / "disks.csv").set_index(["split", "stem"])
fold_of = dict(zip(folds.stem, folds.fold))
use = list(range(len(train_stems)))
print(f"out-of-fold images: {len(use)}; test images: {len(test_stems)}")

coco = data.load_coco()
rec = data.records_frame(coco)
rles = data.record_rles(coco)
stem_records = rec.groupby("stem")["record_id"].apply(list).to_dict()
WORKERS = os.cpu_count()
""")

md("## Contrast maps (for candidate features)")
code(r"""
def disk_tuple(split, stem):
    d = disks.loc[(split, stem)]
    return (float(d.cx), float(d.cy), float(d.r))

t0 = time.time()
jobs = [(s, "train", disk_tuple("train", s), 1024) for s in train_stems] + [(s, "test", disk_tuple("test", s), 1024) for s in test_stems]
with mp.get_context("spawn").Pool(WORKERS) as pool:
    cons = list(pool.imap(prep.contrast_job, jobs, chunksize=4))
con_of = {(split, s): c for (s, split, _, _), c in zip(jobs, cons)}
del cons
print(f"contrast maps in {(time.time() - t0) / 60:.1f} min")
""")

md("## Candidates for a grid of settings (out-of-fold)")
code(r"""
M = len(RUNS)
COMBOS = {RUNS[i].name: tuple(float(i == j) for j in range(M)) for i in range(M)}
if M > 1:
    COMBOS["blend"] = tuple([1.0] * M)
PARAMS = [CandidateParams(t_lo=lo, t_hi=0.6, rho_max=0.95, link_px=lk, min_area=50, weights=w)
          for w in COMBOS.values() for lo in (0.4, 0.45, 0.5, 0.55) for lk in (8, 16)]
combo_name = {w: k for k, w in COMBOS.items()}
for i, p in enumerate(PARAMS):
    print(i, combo_name[p.weights], p)

t0 = time.time()
jobs = ((train_stems[i], [o[i] for o in oofs], disk_tuple("train", train_stems[i]), [rles[r] for r in stem_records[train_stems[i]]],
         PARAMS, con_of[("train", train_stems[i])]) for i in use)
with mp.get_context("spawn").Pool(WORKERS) as pool:
    res = list(pool.imap_unordered(instances.candidate_job, jobs, chunksize=2))
inst = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
recs = [x for r in res for x in r[1]]
inst = inst.sort_values(["stem", "param", "inst"]).reset_index(drop=True)
print(f"candidates in {(time.time() - t0) / 60:.1f} min; {len(inst)} candidates, {len(recs)} record tables")
print(inst.groupby("param").size().rename("candidates").to_frame().T)
""")

md("## Keep rule: all candidates vs learned value model")
code(r"""
rows, sweeps = [], {}
for pi, p in enumerate(PARAMS):
    ip = inst[inst.param == pi].reset_index(drop=True)
    rp = [r for r in recs if r["param"] == pi]
    all_keep = instances.selection_pq(rp, {k: np.ones(len(v), bool) for k, v in ip.groupby(["stem", "param"]).indices.items()})
    q_hat = keep_rule.oof_predictions(ip, fold_of)
    sw = keep_rule.sweep(ip, rp, q_hat, np.r_[-1.0, np.round(np.arange(0.0, 0.61, 0.02), 3)])
    sweeps[pi] = (sw, q_hat)
    best = sw.loc[sw.pq.idxmax()]
    rows.append(dict(param=pi, model=combo_name[p.weights], t_lo=p.t_lo, t_hi=p.t_hi, link=p.link_px, n_cand=len(ip), pq_all=all_keep.pq,
                     pq_learned=best.pq, threshold=best.threshold, sq=best.sq, rq=best.rq, tp=best.tp, fp=best.fp, fn=best.fn))
summary = pd.DataFrame(rows).sort_values("pq_learned", ascending=False)
summary.to_csv("/kaggle/working/oof_keep_rule_summary.csv", index=False)
print(summary.round(4).to_string(index=False))
print("best per model:")
print(summary.groupby("model").head(1).round(4).to_string(index=False))
""")

code(r"""
best = summary.iloc[0]
BEST_PARAM, BEST_T = int(best.param), float(best.threshold)
sw, q_hat = sweeps[BEST_PARAM]
print("chosen:", combo_name[PARAMS[BEST_PARAM].weights], PARAMS[BEST_PARAM], "threshold", BEST_T, f"OOF PQ {best.pq_learned:.4f}")
print(sw.round(4).to_string(index=False))
ip = inst[inst.param == BEST_PARAM].reset_index(drop=True).assign(q_hat=q_hat)
ip.to_csv("/kaggle/working/oof_candidates_best.csv", index=False)
# PQ by site for the chosen setting
rp = [r for r in recs if r["param"] == BEST_PARAM]
sel = keep_rule.selection_from_scores(ip, q_hat, BEST_T)
for site in sorted({s[-2:] for s in ip.stem}):
    c = instances.selection_pq([r for r in rp if r["stem"].endswith(site)], sel)
    print(site, {k: round(v, 4) if isinstance(v, float) else v for k, v in c.as_dict().items()})
""")

md("## Test: refit on all out-of-fold candidates and write the submission")
code(r"""
import cv2
from filseg import solar
from filseg.postprocess import upsample

model, cols = keep_rule.fit_full(ip)
p = PARAMS[BEST_PARAM]
# everything the pipeline notebook needs to rebuild the submission from the fold weights
model.save_model("/kaggle/working/keep_rule.txt")
json.dump(dict(runs=[dict(name=r.name, size=c["size"], arch=c["arch"], encoder=c["encoder"], n_folds=c["n_folds"], tta_views=c["tta_views"])
                     for r, c in zip(RUNS, cfgs)],
               params=dict(t_lo=p.t_lo, t_hi=p.t_hi, rho_max=p.rho_max, link_px=p.link_px, min_area=p.min_area, weights=list(p.weights)),
               threshold=BEST_T, feature_columns=cols, oof_pq=float(best.pq_learned)),
          open("/kaggle/working/postprocess.json", "w"), indent=1)
rows_out, n_kept = [], []
for j, stem in enumerate(test_stems):
    prob = instances.blend([upsample(t[j]) for t in test_probs], p.weights)
    rho = solar.radius_map(solar.Disk(*disk_tuple("test", stem)), prob.shape)
    lab, n, parts = instances.candidates(prob, rho, p)
    if n:
        f = instances.features(lab, n, parts, prob, rho, upsample(con_of[("test", stem)]))
        keep = model.predict(f[cols]) > BEST_T
        remap = np.zeros(n + 1, np.int32); remap[1:][keep] = np.arange(1, keep.sum() + 1)
        lab, n = remap[lab], int(keep.sum())
    n_kept.append(n)
    for i, counts in enumerate(instances.encode_instances(lab, n), 1):
        rows_out.append((f"{stem}_{i}", counts))
sub = pd.DataFrame(rows_out, columns=["filament_id", "segmentation_rle"])
sub.to_csv("/kaggle/working/submission.csv", index=False)
print(f"{len(sub)} filaments on {sum(1 for k in n_kept if k)} of {len(test_stems)} test images "
      f"({np.mean(n_kept):.2f} per image)")
""")

code(r"""
# format checks: unique ids, ids map to test images, every mask decodes to 2048x2048 with positive area, no overlaps
from pycocotools import mask as mask_utils
chk = pd.read_csv("/kaggle/working/submission.csv")
assert list(chk.columns) == ["filament_id", "segmentation_rle"] and chk.filament_id.is_unique
assert set(chk.filament_id.str.rsplit("_", n=1).str[0]) <= set(test_stems)
assert not chk.segmentation_rle.str.contains('"').any()
for stem, g in chk.groupby(chk.filament_id.str.rsplit("_", n=1).str[0]):
    m = mask_utils.decode([{"size": [2048, 2048], "counts": c} for c in g.segmentation_rle])
    assert m.shape[:2] == (2048, 2048) and (m.reshape(-1, m.shape[-1]).sum(0) > 0).all() and m.sum(-1).max() <= 1
print("submission.csv passes format checks")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out_dir = ROOT / "kaggle" / "post"
out_dir.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out_dir / "solar-filament-post.ipynb")
meta = {
    "id": "kavyadhyani/solar-filament-post",
    "title": "Solar Filament Post",
    "code_file": "solar-filament-post.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": ["filament-segmentation-2026"],
    "kernel_sources": SOURCES,
    "model_sources": [],
}
json.dump(meta, open(out_dir / "kernel-metadata.json", "w"), indent=2)
print("built", out_dir)
