"""Builds the end-to-end pipeline notebook: fold weights -> test predictions -> submission.csv.

    python dev/build_pipeline_notebook.py           # notebooks/02_pipeline.ipynb (run from the repository)
    python dev/build_pipeline_notebook.py --kaggle  # kaggle/pipeline/: the same notebook as a Kaggle GPU kernel that
                                                    # mounts the network / post-processing kernels and checks the
                                                    # rebuilt submission against the submitted one
"""

import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--kaggle", action="store_true")
args = ap.parse_args()

# repository run name -> Kaggle kernel holding the same fold weights
RUNS = {"unet_effv2s_1024": "solar-filament-predict", "unet_effv2s_1536": "solar-filament-train1536"}
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# Solar filament segmentation: end-to-end pipeline

This notebook rebuilds the submitted predictions from the stored weights, without any training:

1. **Preprocessing**: limb fit (circle search on an edge map), limb-darkening-corrected contrast and
   distance-from-centre channels (`filseg.prep`, `filseg.solar`).
2. **Networks**: two U-Nets (EfficientNetV2-S encoder) trained on 5 time-grouped folds at 1024 and
   1536 px; each test image is predicted by all 10 fold models with 8-view dihedral test-time
   augmentation (`weights/<run>/fold{k}.pt`).
3. **Instances**: the two probability maps are averaged at full resolution (2048 px), thresholded
   with hysteresis, restricted to the solar disk, and fragments within 8 px are linked
   (`filseg.instances`).
4. **Keep rule**: a LightGBM model predicts each candidate's expected PQ contribution; candidates
   above the out-of-fold-tuned threshold are kept (`weights/keep_rule.txt`, `weights/postprocess.json`).
5. **Submission**: one compressed COCO RLE per filament -> `submission.csv`.

Inputs: the competition images (`MAGFiLO_1.0_Kaggle_2026/`; set `FILSEG_DATA_ROOT` if they are not in
`data/` or `/kaggle/input`). A GPU is recommended (about 45 min on one T4); the code also runs on CPU.
""")

if args.kaggle:
    pkg = {p.name: p.read_text() for p in sorted((ROOT / "src" / "filseg").glob("*.py"))}
    code(
        "# Kaggle check run: write the package, and lay out the mounted kernel outputs like the repository's weights/ folder\n"
        "import importlib.util, subprocess, sys, os, glob, json\n"
        "from pathlib import Path\n"
        "if importlib.util.find_spec('segmentation_models_pytorch') is None:\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.5.0'])\n"
        "REPO = Path('/kaggle/working/repo')\n"
        "pkg = REPO / 'src' / 'filseg'; pkg.mkdir(parents=True, exist_ok=True)\n"
        f"FILES = {pkg!r}\n"
        "for name, text in FILES.items():\n"
        "    (pkg / name).write_text(text)\n"
        f"RUNS = {RUNS!r}\n"
        "W = REPO / 'weights'; W.mkdir(exist_ok=True)\n"
        "for name, kernel in RUNS.items():\n"
        "    (W / name).mkdir(exist_ok=True)\n"
        "    for f in sorted(glob.glob(f'/kaggle/input/{kernel}/weights/fold*.pt')):\n"
        "        (W / name / Path(f).name).symlink_to(f)\n"
        "post = json.load(open('/kaggle/input/solar-filament-post/postprocess.json'))\n"
        "names = {kernel: name for name, kernel in RUNS.items()}\n"
        "for r in post['runs']:\n"
        "    r['name'] = names[r['name']]\n"
        "json.dump(post, open(W / 'postprocess.json', 'w'), indent=1)\n"
        "(W / 'keep_rule.txt').symlink_to('/kaggle/input/solar-filament-post/keep_rule.txt')\n"
        "os.environ['PYTHONPATH'] = str(REPO / 'src') + ':' + os.environ.get('PYTHONPATH', '')\n"
        "os.chdir(REPO)\n"
        "print(sorted(str(p.relative_to(REPO)) for p in W.rglob('*')))"
    )

code(r"""
import os, sys, time, json, platform
from importlib import metadata
from pathlib import Path

# repository root = the folder holding weights/postprocess.json (this notebook lives in notebooks/)
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "weights" / "postprocess.json").exists())
sys.path.insert(0, str(REPO / "src"))
os.environ["PYTHONPATH"] = str(REPO / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")  # for worker processes

import numpy as np
import pandas as pd
import torch
from filseg import data, pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("python", platform.python_version(), "| torch", torch.__version__, "| device", device,
      torch.cuda.get_device_name(0) if device.type == "cuda" else "")


def version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


print({k: version(k) for k in ("numpy", "pandas", "scipy", "opencv-python", "opencv-python-headless", "pycocotools",
                               "lightgbm", "segmentation-models-pytorch", "timm", "torch") if version(k)})
print("data:", data.DATA_ROOT)
""")

code(r"""
post, keep_model = pipeline.load_post(REPO / "weights")
print(json.dumps({k: v for k, v in post.items() if k != "feature_columns"}, indent=1))
test_stems = sorted(p.stem for p in data.TEST_IMAGES.glob("*.jpeg"))
print("test images:", len(test_stems))
""")

md("## Network predictions (10 fold models x 8 views)")
code(r"""
maps, disks = [], None
for run in post["runs"]:
    t0 = time.time()
    m, d = pipeline.probability_maps(test_stems, "test", run, REPO / "weights" / run["name"], device)
    maps.append(m)
    disks = d if disks is None else disks
    print(f"{run['name']}: maps {m.shape} in {(time.time() - t0) / 60:.1f} min")
""")

md("## Instances, keep rule and submission")
code(r"""
t0 = time.time()
cons = pipeline.contrast_maps(test_stems, "test", disks)
sub = pipeline.make_submission(test_stems, maps, disks, cons, post, keep_model)
sub.to_csv("submission.csv", index=False)
n_img = sub.filament_id.str.rsplit("_", n=1).str[0].nunique()
print(f"{len(sub)} filaments on {n_img} of {len(test_stems)} test images in {(time.time() - t0) / 60:.1f} min -> submission.csv")
""")

code(r"""
# format checks: unique ids, ids map to test images, every mask decodes to 2048 x 2048 with positive area, no overlaps
from pycocotools import mask as mask_utils
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["filament_id", "segmentation_rle"] and chk.filament_id.is_unique
assert set(chk.filament_id.str.rsplit("_", n=1).str[0]) <= set(test_stems)
for _stem, g in chk.groupby(chk.filament_id.str.rsplit("_", n=1).str[0]):
    m = mask_utils.decode([{"size": [data.H, data.W], "counts": c} for c in g.segmentation_rle])
    assert (m.reshape(-1, m.shape[-1]).sum(0) > 0).all() and m.sum(-1).max() <= 1
print("submission.csv passes format checks")
""")

if args.kaggle:
    code(r"""
# reproduction check against the submitted file (post-processing kernel output)
ref = pd.read_csv("/kaggle/input/solar-filament-post/submission.csv")
print("rows: rebuilt", len(sub), "| submitted", len(ref), "| identical rows:", len(sub.merge(ref)))
def union(df, stem):
    g = df[df.filament_id.str.rsplit("_", n=1).str[0] == stem]
    if len(g) == 0:
        return np.zeros((data.H, data.W), bool)
    return mask_utils.decode([{"size": [data.H, data.W], "counts": c} for c in g.segmentation_rle]).any(-1)
inter = uni = 0
for s in test_stems:
    a, b = union(sub, s), union(ref, s)
    inter += (a & b).sum(); uni += (a | b).sum()
print(f"pixel IoU of all filament pixels, rebuilt vs submitted: {inter / uni:.5f}")
""")

md("""
## Training (not run here)

The weights were produced on Kaggle (2 x T4) with `filseg.train.run`, e.g.

```python
from filseg.train import Config, run
run(Config(size=1024, epochs=40, seed=0))                 # weights/unet_effv2s_1024
run(Config(size=1536, epochs=60, seed=1, eval_every=10))  # weights/unet_effv2s_1536
```

and the keep rule with the post-processing notebook (`dev/build_post_notebook.py`); see the README.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
if args.kaggle:
    out_dir = ROOT / "kaggle" / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out_dir / "solar-filament-pipeline.ipynb")
    meta = {
        "id": "kavyadhyani/solar-filament-pipeline",
        "title": "Solar Filament Pipeline",
        "code_file": "solar-filament-pipeline.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": ["filament-segmentation-2026"],
        "kernel_sources": ["kavyadhyani/" + k for k in RUNS.values()] + ["kavyadhyani/solar-filament-post"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    json.dump(meta, open(out_dir / "kernel-metadata.json", "w"), indent=2)
else:
    nbf.write(nb, ROOT / "notebooks" / "02_pipeline.ipynb")
print("built", "kaggle/pipeline" if args.kaggle else "notebooks/02_pipeline.ipynb")
