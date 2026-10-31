"""Loading the MAGFiLO COCO annotations and the GONG H-alpha images.

Terminology used throughout the project:

- *stem*: one physical observation, e.g. ``20140609195854Bh`` (one JPEG file).
- *record*: one annotator's annotation set of a stem, e.g. ``040301-20140609195854Bh``.
  The official scorer treats every record as a separate ground-truth image.
- *annotator*: the six-digit prefix of a record id, ``BBBBSS`` = batch ``BBBB`` + slot ``SS``.
"""

from __future__ import annotations

import glob
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = "MAGFiLO_1.0_Kaggle_2026"


def find_data_root() -> Path:
    """$FILSEG_DATA_ROOT, else the project's data/ folder, else the Kaggle competition mount."""
    if os.environ.get("FILSEG_DATA_ROOT"):
        return Path(os.environ["FILSEG_DATA_ROOT"])
    local = PROJECT_ROOT / "data" / DATASET_DIR
    if local.exists():
        return local
    hits = sorted(glob.glob(f"/kaggle/input/*/{DATASET_DIR}") + glob.glob(f"/kaggle/input/*/*/{DATASET_DIR}"))
    if not hits:
        raise FileNotFoundError(f"{DATASET_DIR} not found; set FILSEG_DATA_ROOT")
    return Path(hits[0])


DATA_ROOT = find_data_root()
TRAIN_JSON = DATA_ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
TRAIN_IMAGES = DATA_ROOT / "train" / "train_images"
TEST_IMAGES = DATA_ROOT / "test" / "test_images"

H = W = 2048
OBSERVATORIES = {
    "Bh": "Big Bear",
    "Ch": "Cerro Tololo",
    "Lh": "Learmonth",
    "Mh": "Mauna Loa",
    "Th": "El Teide",
    "Uh": "Udaipur",
}


@lru_cache(maxsize=1)
def load_coco(path: Path = TRAIN_JSON) -> dict:
    with open(path) as f:
        return json.load(f)


def records_frame(coco: dict) -> pd.DataFrame:
    """One row per record (annotator x stem), with filament counts."""
    rows = []
    for im in coco["images"]:
        annotator, stem = im["id"].split("-", 1)
        rows.append(
            dict(
                record_id=im["id"],
                stem=stem,
                annotator=annotator,
                batch=annotator[:4],
                slot=annotator[4:],
                observatory=stem[-2:],
                time=pd.Timestamp(im["date_captured"]),
            )
        )
    rec = pd.DataFrame(rows)
    counts = pd.Series([a["image_id"] for a in coco["annotations"]]).value_counts()
    rec["n_filaments"] = rec["record_id"].map(counts).fillna(0).astype(int)
    rec["n_records_of_stem"] = rec.groupby("stem")["record_id"].transform("size")
    return rec


def polyline_length(xy: list[float]) -> float:
    p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def annotations_frame(coco: dict) -> pd.DataFrame:
    """One row per annotated filament with its geometry from the JSON (no mask decoding)."""
    rows = []
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        poly = a["segmentation"][0]
        rows.append(
            dict(
                ann_id=a["id"],
                record_id=a["image_id"],
                stem=a["image_id"].split("-", 1)[1],
                category_id=a["category_id"],
                area=a["area"],
                bbox_x=x,
                bbox_y=y,
                bbox_w=w,
                bbox_h=h,
                n_vertices=len(poly) // 2,
                spine_length=polyline_length(a["spine"]),
                n_spine_points=len(a["spine"]) // 2,
            )
        )
    return pd.DataFrame(rows)


def annotation_rle(ann: dict, height: int = H, width: int = W) -> dict:
    """Compressed COCO RLE of one annotation polygon (same path as pycocotools annToRLE)."""
    rles = mask_utils.frPyObjects(ann["segmentation"], height, width)
    return mask_utils.merge(rles)


def record_rles(coco: dict) -> dict[str, list[dict]]:
    """record_id -> list of compressed RLEs, in JSON order."""
    out: dict[str, list[dict]] = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        out[a["image_id"]].append(annotation_rle(a))
    return out


def decode(rles: list[dict]) -> np.ndarray:
    """(n, H, W) uint8 stack of masks."""
    if not rles:
        return np.zeros((0, H, W), np.uint8)
    return np.ascontiguousarray(mask_utils.decode(rles).transpose(2, 0, 1))


def encode(mask: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def load_image(stem: str, split: str = "train") -> np.ndarray:
    """Grayscale uint8 image (2048 x 2048). The JPEGs are grayscale and are read as such."""
    folder = TRAIN_IMAGES if split == "train" else TEST_IMAGES
    with Image.open(folder / f"{stem}.jpeg") as im:
        return np.asarray(im.convert("L"))
