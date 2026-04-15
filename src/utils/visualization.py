"""
Visualization utilities using OpenCV and Matplotlib.

Functions
---------
draw_skeleton           – overlay the 21-keypoint hand skeleton on an image
plot_confusion_matrix   – annotated heatmap of the confusion matrix
plot_training_curves    – loss / accuracy curves for a training history
plot_per_class_f1       – bar chart of per-class F1 scores
visualise_attention     – grad-CAM for CNN models
save_augmentation_grid  – show augmentation effect on sample images
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

log = logging.getLogger(__name__)

# hand skeleton connections (wrist=0, per-finger pairs)
_SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # knuckle row
]

_FINGER_COLORS = [
    (255, 0,   0),    # wrist
    (255, 128, 0),    # thumb
    (255, 255, 0),    # index
    (0,   255, 0),    # middle
    (0,   0,   255),  # ring
    (255, 0,   255),  # pinky
]

_NODE_FINGER = [0] + [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*4   # 21 nodes


def draw_skeleton(
    image:     np.ndarray,
    keypoints: np.ndarray,
    color_connections: bool = True,
) -> np.ndarray:
    """
    Draw the 21-node hand skeleton on an OpenCV image.

    Parameters
    ----------
    image      : (H, W, 3) uint8 BGR image
    keypoints  : (21, 2) array of (x, y) in pixel coordinates
    Returns BGR image with skeleton overlay.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # normalise if coords are in [0, 1]
    kp = keypoints.copy().astype(np.float32)
    if kp.max() <= 1.0:
        kp[:, 0] *= w
        kp[:, 1] *= h
    kp = kp.astype(np.int32)

    for (i, j) in _SKELETON_CONNECTIONS:
        color = _FINGER_COLORS[_NODE_FINGER[i]] if color_connections else (0, 255, 0)
        cv2.line(img, tuple(kp[i]), tuple(kp[j]), color, 2, cv2.LINE_AA)

    for idx, (x, y) in enumerate(kp):
        color = _FINGER_COLORS[_NODE_FINGER[idx]]
        cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ── confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm:         np.ndarray,
    classes:    List[str],
    save_path:  Optional[str] = None,
    normalize:  bool = True,
    title:      str = "Confusion Matrix",
) -> None:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm / row_sums, 0).astype(float)
    else:
        cm_plot = cm.astype(float)

    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(8, len(classes))))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm_plot.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_plot[i, j]:.2f}" if normalize else str(int(cm[i, j])),
                    ha="center", va="center",
                    color="white" if cm_plot[i, j] > thresh else "black",
                    fontsize=6)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        log.info("Confusion matrix saved to %s", save_path)
    plt.close(fig)


# ── training curves ───────────────────────────────────────────────────────────

def plot_training_curves(
    history:   Dict,
    model_name: str = "model",
    save_path: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss", "Accuracy"],
    ):
        for key, label in zip(metric, ["Train", "Val"]):
            if key in history:
                ax.plot(history[key], label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model_name} – {ylabel}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(model_name)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        log.info("Training curves saved to %s", save_path)
    plt.close(fig)


# ── per-class F1 bar chart ────────────────────────────────────────────────────

def plot_per_class_f1(
    f1_dict:   Dict[str, float],
    model_name: str = "model",
    save_path: Optional[str] = None,
) -> None:
    classes = list(f1_dict.keys())
    scores  = [f1_dict[c] for c in classes]

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.5), 4))
    bars = ax.bar(classes, scores, color="steelblue")
    ax.axhline(np.mean(scores), color="red", linestyle="--",
               label=f"Mean={np.mean(scores):.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{model_name} – Per-class F1")
    ax.legend()
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        log.info("F1 chart saved to %s", save_path)
    plt.close(fig)


# ── Grad-CAM attention map ────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Map for CNN-based models.

    Parameters
    ----------
    model       : the CNN (must be in eval mode)
    target_layer: the conv layer to hook  (e.g., model.features[-1])
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model  = model
        self.grads: Optional[torch.Tensor] = None
        self.feats: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_feat)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_grad)

    def _save_feat(self, _m, _i, output):
        self.feats = output.detach()

    def _save_grad(self, _m, _gi, grad_output):
        self.grads = grad_output[0].detach()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        x : (1, 3, H, W) image tensor
        Returns: (H, W) CAM heatmap in [0, 1]
        """
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax())

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # (C, H', W') → weights
        weights = self.grads[0].mean(dim=(1, 2))   # (C,)
        cam     = (weights[:, None, None] * self.feats[0]).sum(0)  # (H', W')
        cam     = torch.relu(cam).cpu().numpy()

        # resize to input spatial dims
        h, w = x.shape[-2:]
        cam   = cv2.resize(cam, (w, h))
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)


def overlay_gradcam(
    image: np.ndarray,
    cam:   np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on a BGR image."""
    heatmap = cv2.applyColorMap(
        (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


# ── augmentation grid ─────────────────────────────────────────────────────────

def save_augmentation_grid(
    image_path: str,
    n_augmented: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """Save a grid of an original image + N augmented versions."""
    from src.data.augmentation import build_train_transform, build_eval_transform

    original = Image.open(image_path).convert("RGB")
    train_tf = build_train_transform()

    images = [build_eval_transform()(original)]
    for _ in range(n_augmented):
        images.append(train_tf(original))

    # denormalise
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    imgs = [(img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            for img in images]

    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    titles = ["Original"] + [f"Aug {i}" for i in range(1, n_augmented + 1)]
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(imgs):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120)
        log.info("Augmentation grid saved to %s", save_path)
    plt.close(fig)
