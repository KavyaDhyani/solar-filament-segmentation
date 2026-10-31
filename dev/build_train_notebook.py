"""Builds the Kaggle training notebook (kaggle/train/) with the filseg package embedded.

    python dev/build_train_notebook.py --debug   # smoke run: 60 images, 2 folds x 2 epochs
    python dev/build_train_notebook.py           # full run
    python dev/build_train_notebook.py --kernel predict --source kavyadhyani/solar-filament-train \
        --config '{"init_from": "auto"}'          # re-predict with the weights of an earlier run (no training)
"""

import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--debug", action="store_true")
ap.add_argument("--config", default="{}", help="JSON overrides for filseg.train.Config")
ap.add_argument("--kernel", default="train", help="kernel slug suffix and kaggle/ subfolder")
ap.add_argument("--source", action="append", default=[], help="kernel whose output is mounted (repeatable)")
args = ap.parse_args()

pkg = {p.name: p.read_text() for p in sorted((ROOT / "src" / "filseg").glob("*.py"))}
overrides = json.loads(args.config)
if args.debug:
    overrides = {"debug": True, "epochs": 2, "folds_to_run": [0, 1], "eval_every": 1, **overrides}

cells = [
    nbf.v4.new_markdown_cell(
        "# Solar filament segmentation: grouped-fold training\n\n"
        "Trains the semantic filament network on 5 time-grouped folds, saves out-of-fold and test "
        "probability maps plus the fold weights, and prints the out-of-fold PQ of a simple "
        "post-processing grid (official matching rule). The `filseg` package is written next to "
        "the notebook by the first cell."
        + ("\n\n**DEBUG / smoke run: do not use these outputs.**" if args.debug else "")),
    nbf.v4.new_code_cell(
        "import importlib.util, subprocess, sys, os, platform\n"
        "os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')\n"
        "from pathlib import Path\n"
        "if importlib.util.find_spec('segmentation_models_pytorch') is None:\n"
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'segmentation-models-pytorch==0.5.0'])\n"
        "pkg = Path('/kaggle/working/src/filseg'); pkg.mkdir(parents=True, exist_ok=True)\n"
        f"FILES = {pkg!r}\n"
        "for name, text in FILES.items():\n"
        "    (pkg / name).write_text(text)\n"
        "sys.path.insert(0, '/kaggle/working/src')\n"
        "os.environ['PYTHONPATH'] = '/kaggle/working/src:' + os.environ.get('PYTHONPATH', '')  # spawn workers import filseg\n"
        "import torch, segmentation_models_pytorch as smp, timm\n"
        "print('python', platform.python_version(), '| torch', torch.__version__, '| smp', smp.__version__, '| timm', timm.__version__)\n"
        "print('cpus', os.cpu_count(), '| gpus', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
        "assert torch.cuda.is_available(), 'GPU required'"),
    nbf.v4.new_code_cell(
        "from filseg.train import Config, run\n"
        f"cfg = Config(**{overrides!r})\n"
        "print(cfg)\n"
        "run(cfg)"),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out_dir = ROOT / "kaggle" / args.kernel
out_dir.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out_dir / f"solar-filament-{args.kernel}.ipynb")
meta = {
    "id": f"kavyadhyani/solar-filament-{args.kernel}",
    "title": f"Solar Filament {args.kernel.title()}",
    "code_file": f"solar-filament-{args.kernel}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": ["filament-segmentation-2026"],
    "kernel_sources": args.source,
    "model_sources": [],
    # NvidiaL4 and the pinned ARC image were both refused (HTTP 400) for this kernel on 2026-10-30
    "machine_shape": "NvidiaTeslaT4",
}
json.dump(meta, open(out_dir / "kernel-metadata.json", "w"), indent=2)
print("built", "DEBUG" if args.debug else "FULL", overrides)
