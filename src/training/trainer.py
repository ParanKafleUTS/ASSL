"""
Generic training loop with:
  * Early stopping (configurable patience)
  * Learning-rate scheduling (ReduceLROnPlateau)
  * Mixed-precision training (torch.cuda.amp) for GPU speed-up
  * Gradient clipping
  * Checkpoint save / resume
  * Reproducible seeding
"""

import os
import math
import random
import time
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False   # reproducibility


class EarlyStopping:
    """Stop training when validation loss does not improve."""

    def __init__(self, patience: int = config.EARLY_STOP_PATIENCE,
                 min_delta: float = 1e-4, mode: str = "min"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.best      = math.inf if mode == "min" else -math.inf
        self.counter   = 0
        self.triggered = False

    def __call__(self, metric: float) -> bool:
        improved = (
            (self.mode == "min" and metric < self.best - self.min_delta) or
            (self.mode == "max" and metric > self.best + self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ── accuracy helper ───────────────────────────────────────────────────────────

def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ── one epoch helpers ─────────────────────────────────────────────────────────

def _train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    clip_grad: float = 1.0,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc  += _accuracy(logits.detach(), labels)
        n_batches  += 1

    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches}


@torch.no_grad()
def _eval_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        total_acc  += _accuracy(logits, labels)
        n_batches  += 1

    return {"loss": total_loss / n_batches, "acc": total_acc / n_batches}


# ── main train function ───────────────────────────────────────────────────────

def train(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    model_name:    str  = "model",
    num_epochs:    int  = config.MAX_EPOCHS,
    lr:            float = config.LR_DEFAULT,
    weight_decay:  float = config.WEIGHT_DECAY,
    patience:      int  = config.EARLY_STOP_PATIENCE,
    checkpoint_dir: str = config.CHECKPOINT_DIR,
    device:        torch.device = config.DEVICE,
    label_smoothing: float = 0.1,
) -> Dict:
    """
    Full training loop.

    Returns
    -------
    history : {train_loss, train_acc, val_loss, val_acc} per epoch
    """
    set_seed()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2, min_lr=1e-6,
    )
    scaler  = GradScaler(enabled=(device.type == "cuda"))
    es      = EarlyStopping(patience=patience, mode="min")
    ckpt    = Path(checkpoint_dir) / f"{model_name}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    history: Dict = {k: [] for k in
                     ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_val_loss = math.inf

    for epoch in range(1, num_epochs + 1):
        t0  = time.time()
        tr  = _train_epoch(model, train_loader, criterion,
                           optimizer, scaler, device)
        val = _eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val["loss"])
        elapsed = time.time() - t0

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])

        log.info(
            "[%s] E%03d/%03d  tr_loss=%.4f tr_acc=%.4f  "
            "val_loss=%.4f val_acc=%.4f  lr=%.2e  %.1fs",
            model_name, epoch, num_epochs,
            tr["loss"], tr["acc"],
            val["loss"], val["acc"],
            optimizer.param_groups[0]["lr"],
            elapsed,
        )

        # save best checkpoint
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   val["loss"],
                "val_acc":    val["acc"],
            }, ckpt)

        if es(val["loss"]):
            log.info("[%s] Early stopping at epoch %d", model_name, epoch)
            break

    # restore best weights
    ckpt_data = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_data["model_state"])
    log.info("[%s] Restored best weights (val_loss=%.4f, val_acc=%.4f)",
             model_name, ckpt_data["val_loss"], ckpt_data["val_acc"])
    history["best_val_acc"] = ckpt_data["val_acc"]
    return history
