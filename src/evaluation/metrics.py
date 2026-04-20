"""
Comprehensive evaluation metrics for ASL recognition.

Metrics implemented
-------------------
* Accuracy
* Per-class F1 (macro / weighted)
* Confusion matrix
* Bootstrap 95 % confidence interval on accuracy
* McNemar's test for pairwise model comparison
* Robustness degradation under noise / blur
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import chi2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

log = logging.getLogger(__name__)


# ── prediction collection ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device = config.DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on all batches and return (predictions, true_labels).
    """
    model.eval()
    model.to(device)
    preds_list  = []
    labels_list = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        preds_list.append(preds)
        labels_list.append(labels.numpy())

    return np.concatenate(preds_list), np.concatenate(labels_list)


# ── accuracy ──────────────────────────────────────────────────────────────────

def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_accuracy_ci(
    preds:  np.ndarray,
    labels: np.ndarray,
    n:      int   = config.BOOTSTRAP_N,
    alpha:  float = config.CI_ALPHA,
    seed:   int   = config.SEED,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for accuracy.

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    point = accuracy(preds, labels)
    boot  = np.zeros(n)
    N     = len(preds)
    for i in range(n):
        idx      = rng.integers(0, N, size=N)
        boot[i]  = accuracy(preds[idx], labels[idx])

    lo = float(np.percentile(boot, 100 * (1 - alpha) / 2))
    hi = float(np.percentile(boot, 100 * (1 - (1 - alpha) / 2)))
    return point, lo, hi


# ── per-class F1 ──────────────────────────────────────────────────────────────

def per_class_f1(
    preds:   np.ndarray,
    labels:  np.ndarray,
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Return per-class F1 scores as {class_name: f1}.
    """
    num_classes = int(labels.max()) + 1
    f1_dict: Dict[str, float] = {}

    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        denom = 2 * tp + fp + fn
        f1    = (2 * tp / denom) if denom > 0 else 0.0
        name  = classes[c] if classes else str(c)
        f1_dict[name] = round(f1, 4)

    return f1_dict


def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(list(per_class_f1(preds, labels).values())))


# ── confusion matrix ──────────────────────────────────────────────────────────

def confusion_matrix(
    preds:  np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return (num_classes × num_classes) confusion matrix."""
    num_classes = int(labels.max()) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


# ── McNemar's test ────────────────────────────────────────────────────────────

def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    labels:  np.ndarray,
) -> Dict[str, float]:
    """
    McNemar's test for statistical significance between two classifiers.

    Returns
    -------
    {"chi2": statistic, "p_value": p, "significant": bool (α=0.05)}
    """
    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(( correct_a & ~correct_b).sum())
    c = int((~correct_a &  correct_b).sum())

    # continuity-corrected McNemar
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant": False}

    stat   = (abs(b - c) - 1) ** 2 / (b + c)
    p_val  = float(1 - chi2.cdf(stat, df=1))
    return {
        "chi2":        round(stat,  4),
        "p_value":     round(p_val, 4),
        "significant": p_val < 0.05,
    }


# ── pairwise McNemar for all model pairs ─────────────────────────────────────

def pairwise_mcnemar(
    predictions: Dict[str, np.ndarray],
    labels:      np.ndarray,
) -> Dict[Tuple[str, str], Dict]:
    """
    Run McNemar's test for every pair of models.

    Parameters
    ----------
    predictions : {model_name: pred_array}
    labels      : true labels

    Returns
    -------
    {("model_a", "model_b"): {"chi2": …, "p_value": …, "significant": …}}
    """
    results = {}
    for a, b in combinations(predictions.keys(), 2):
        results[(a, b)] = mcnemar_test(predictions[a], predictions[b], labels)
    return results


# ── robustness test ───────────────────────────────────────────────────────────

def robustness_report(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device = config.DEVICE,
) -> Dict[str, float]:
    """
    Measure accuracy degradation under:
      * Gaussian noise  (σ = 0.1)
      * Gaussian blur   (kernel 5×5)
      * Brightness reduction (factor = 0.5)

    Returns dict of condition → accuracy.
    """
    import torchvision.transforms.functional as TF
    import torch

    def _apply_noise(imgs):
        return torch.clamp(imgs + 0.1 * torch.randn_like(imgs), 0, 1)

    def _apply_blur(imgs):
        # box blur approximation using avg_pool
        return torch.nn.functional.avg_pool2d(imgs, kernel_size=5, stride=1, padding=2)

    def _apply_dark(imgs):
        return imgs * 0.5

    conditions = {
        "clean":       lambda x: x,
        "noise":       _apply_noise,
        "blur":        _apply_blur,
        "dark":        _apply_dark,
    }

    model.eval()
    model.to(device)
    results: Dict[str, float] = {}

    for cond_name, transform in conditions.items():
        correct = 0
        total   = 0
        with torch.no_grad():
            for images, labels in loader:
                images = transform(images.to(device))
                labels = labels.to(device)
                preds  = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += len(labels)
        results[cond_name] = round(correct / total, 4) if total > 0 else 0.0

    return results


# ── full evaluation report ────────────────────────────────────────────────────

def full_report(
    model:   nn.Module,
    loader:  DataLoader,
    classes: Optional[List[str]] = None,
    device:  torch.device = config.DEVICE,
) -> Dict:
    """
    Compute and return all evaluation metrics for a single model.
    """
    preds, labels = collect_predictions(model, loader, device)

    point, lo, hi = bootstrap_accuracy_ci(preds, labels)
    f1            = per_class_f1(preds, labels, classes)
    cm            = confusion_matrix(preds, labels)
    robustness    = robustness_report(model, loader, device)

    report = {
        "accuracy":             round(point, 4),
        "accuracy_ci_lower":    round(lo,    4),
        "accuracy_ci_upper":    round(hi,    4),
        "macro_f1":             round(macro_f1(preds, labels), 4),
        "per_class_f1":         f1,
        "confusion_matrix":     cm.tolist(),
        "robustness":           robustness,
    }

    log.info("Accuracy: %.4f  [%.4f – %.4f]", point, lo, hi)
    log.info("Macro F1: %.4f", report["macro_f1"])
    return report
