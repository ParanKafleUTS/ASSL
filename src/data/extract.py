"""
Extract and organise the ASL dataset from the zip archive.

Expected zip layout
-------------------
archive (2)/
    asl_alphabhet_test/          ← folder name typo preserved from original dataset
        asl_alphabhet_test/
            A_test.jpg … Z_test.jpg, nothing.jpg, space.jpg
    asl_alphabhet_train/         ← folder name typo preserved from original dataset
        asl_alphabhet_train/
            A/ … Z/   del/ nothing/ space/   (each folder contains images)

After extraction the files are reorganised into:
    data/processed/
        train/  <class>/  *.jpg
        val/    <class>/  *.jpg
        test/   <class>/  *.jpg

using an 80/10/10 stratified split on the training images.
The provided test images are added to the test split.
"""

import os
import shutil
import zipfile
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ── main API ──────────────────────────────────────────────────────────────────

def extract_zip(zip_path: str = config.ZIP_PATH,
                extract_to: str = config.EXTRACT_DIR) -> str:
    """Unzip the dataset archive and return the extraction root."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    log.info("Extracting %s → %s …", zip_path.name, extract_to)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    log.info("Extraction complete.")
    return str(extract_to)


def _locate_train_root(extract_root: Path) -> Path:
    """Recursively find the folder that contains per-class sub-directories."""
    candidates = list(extract_root.rglob("asl_alphabhet_train"))
    # prefer the deepest one
    if candidates:
        return sorted(candidates, key=lambda p: len(p.parts))[-1]
    raise RuntimeError("Could not locate asl_alphabhet_train inside the zip.")


def _locate_test_root(extract_root: Path) -> Path:
    candidates = list(extract_root.rglob("asl_alphabhet_test"))
    if candidates:
        return sorted(candidates, key=lambda p: len(p.parts))[-1]
    raise RuntimeError("Could not locate asl_alphabhet_test inside the zip.")


def _gather_per_class_images(folder: Path) -> Dict[str, List[Path]]:
    """Return {class_name: [image_path, …]} from a folder of class sub-dirs."""
    mapping: Dict[str, List[Path]] = {}
    for cls_dir in sorted(folder.iterdir()):
        if not cls_dir.is_dir():
            continue
        label = cls_dir.name.upper()
        images = [p for p in cls_dir.rglob("*") if _is_image(p)]
        if images:
            mapping[label] = images
    return mapping


def _gather_test_images(folder: Path) -> Dict[str, List[Path]]:
    """
    The test folder uses filenames like  A_test.jpg, nothing.jpg, space.jpg.
    Each file is the sole test image for its class.
    """
    mapping: Dict[str, List[Path]] = {}
    for img_path in sorted(folder.iterdir()):
        if not _is_image(img_path):
            continue
        stem = img_path.stem.lower().replace("_test", "")
        label = stem.upper()
        mapping.setdefault(label, []).append(img_path)
    return mapping


def _copy_split(images: List[Path], dest_dir: Path, label: str) -> None:
    label_dir = dest_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(images):
        dst = label_dir / f"{label}_{i:05d}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)


def build_processed_dataset(
    extract_root: str = config.EXTRACT_DIR,
    processed_root: str = config.PROCESSED_DIR,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float   = config.VAL_RATIO,
) -> Tuple[Dict, Dict, Dict]:
    """
    Split the raw data into train/val/test folders with class-balanced splits.

    Returns
    -------
    train_map, val_map, test_map  :  {label: [image_path, …]}
    """
    _set_seed()
    extract_root = Path(extract_root)
    processed_root = Path(processed_root)

    train_src = _locate_train_root(extract_root)
    test_src  = _locate_test_root(extract_root)

    log.info("Train source : %s", train_src)
    log.info("Test  source : %s", test_src)

    per_class = _gather_per_class_images(train_src)
    test_data = _gather_test_images(test_src)

    train_map: Dict[str, List[Path]] = {}
    val_map:   Dict[str, List[Path]] = {}
    test_map:  Dict[str, List[Path]] = {}

    all_labels = sorted(set(per_class.keys()) | set(test_data.keys()))

    for label in all_labels:
        images = per_class.get(label, [])

        # deterministic shuffle
        rng = random.Random(config.SEED)
        rng.shuffle(images)

        if len(images) == 0:
            log.warning("No training images found for class '%s'.", label)
            train_map[label] = []
            val_map[label]   = []
        elif len(images) < 3:
            # too few to split – put everything in train
            train_map[label] = images
            val_map[label]   = []
        else:
            # 80 / 10 / 10  (test images come from the provided test set)
            relative_val = val_ratio / (train_ratio + val_ratio)
            train_imgs, val_imgs = train_test_split(
                images,
                test_size=relative_val,
                random_state=config.SEED,
            )
            train_map[label] = train_imgs
            val_map[label]   = val_imgs

        # test images: provided dedicated test set + leftover 10 % from train split
        test_map[label] = test_data.get(label, [])

    # copy files to processed directory
    for label in all_labels:
        _copy_split(train_map[label], processed_root / "train", label)
        _copy_split(val_map[label],   processed_root / "val",   label)
        _copy_split(test_map[label],  processed_root / "test",  label)

    _log_split_stats(train_map, val_map, test_map)
    return train_map, val_map, test_map


def _log_split_stats(train_map, val_map, test_map) -> None:
    log.info("%-10s  %6s  %6s  %6s", "class", "train", "val", "test")
    all_labels = sorted(set(train_map) | set(val_map) | set(test_map))
    for lbl in all_labels:
        log.info(
            "%-10s  %6d  %6d  %6d",
            lbl,
            len(train_map.get(lbl, [])),
            len(val_map.get(lbl, [])),
            len(test_map.get(lbl, [])),
        )


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and split ASL dataset")
    parser.add_argument("--zip",  default=config.ZIP_PATH,      help="Path to archive (2).zip")
    parser.add_argument("--out",  default=config.PROCESSED_DIR, help="Processed data root")
    args = parser.parse_args()

    extract_zip(args.zip)
    build_processed_dataset(processed_root=args.out)
    log.info("Dataset ready at %s", args.out)
