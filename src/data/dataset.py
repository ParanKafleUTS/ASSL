"""
PyTorch Dataset classes for the ASL recognition pipeline.

Datasets
--------
ASLImageDataset   – standard image classification (CNN / transfer models)
ASLKeypointDataset – returns (image, keypoint_coords, label) triples
                     used by the Skeleton GCN pipeline
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.data.augmentation import (
    build_train_transform,
    build_eval_transform,
    balance_dataset,
)


# ── helpers ───────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _scan_folder(root: Path) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Walk *root/<class>/*.{jpg,…} and return (samples, classes)."""
    class_dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir()
    )
    classes = [d.name for d in class_dirs]
    label2idx = {c: i for i, c in enumerate(classes)}
    samples: List[Tuple[str, int]] = []
    for cls_dir in class_dirs:
        for img_path in cls_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(img_path), label2idx[cls_dir.name]))
    return samples, classes


# ── image dataset ─────────────────────────────────────────────────────────────

class ASLImageDataset(Dataset):
    """
    Standard classification dataset.

    Parameters
    ----------
    root        : directory containing per-class sub-folders
    transform   : callable applied to each PIL image
    balance     : if True oversample minority classes (training only)
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 balance: bool = False):
        self.root      = Path(root)
        self.transform = transform

        raw_samples, self.classes = _scan_folder(self.root)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if balance:
            per_class: dict = {}
            for path, idx in raw_samples:
                label = self.classes[idx]
                per_class.setdefault(label, []).append(Path(path))
            self.samples = balance_dataset(per_class)
        else:
            self.samples = raw_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Return a sampler that draws equal samples per class (oversamples)."""
        labels  = [lbl for _, lbl in self.samples]
        counts  = np.bincount(labels)
        weights = 1.0 / counts[labels]
        return WeightedRandomSampler(
            weights=torch.DoubleTensor(weights),
            num_samples=len(self.samples),
            replacement=True,
        )


# ── keypoint dataset ──────────────────────────────────────────────────────────

class ASLKeypointDataset(Dataset):
    """
    Returns (image_tensor, label).
    The GCN trainer extracts keypoints on-the-fly using KeypointCNN.

    If pre-computed keypoints are available (npy cache) they are loaded
    instead of running the keypoint CNN at every iteration.
    """

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 keypoint_cache_dir: Optional[str] = None,
                 balance: bool = False):
        self.image_ds = ASLImageDataset(root, transform=transform, balance=balance)
        self.classes  = self.image_ds.classes
        self.cache_dir = Path(keypoint_cache_dir) if keypoint_cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.image_ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.image_ds[idx]   # (image, label); GCN trainer handles kp extraction


# ── data-loader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    processed_root: str = config.PROCESSED_DIR,
    image_size:     int = config.IMAGE_SIZE,
    batch_size:     int = config.BATCH_SIZE,
    num_workers:    int = config.NUM_WORKERS,
    balance_train:  bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Build train / val / test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, class_names
    """
    processed_root = Path(processed_root)

    train_tf = build_train_transform(image_size)
    eval_tf  = build_eval_transform(image_size)

    train_ds = ASLImageDataset(str(processed_root / "train"),
                               transform=train_tf, balance=balance_train)
    val_ds   = ASLImageDataset(str(processed_root / "val"),
                               transform=eval_tf,  balance=False)
    test_ds  = ASLImageDataset(str(processed_root / "test"),
                               transform=eval_tf,  balance=False)

    classes = train_ds.classes

    # use WeightedRandomSampler for training so each mini-batch is balanced
    sampler = train_ds.get_weighted_sampler() if balance_train else None

    common = dict(
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader, test_loader, classes
