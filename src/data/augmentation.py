"""
Augmentation pipeline for ASL hand images.

Goals
-----
* Handle any hand skin tone / lighting condition.
* Improve robustness to scale, rotation, and viewpoint.
* Keep images uncorrupted (pixel values stay in [0, 255]).
* Produce a *balanced* dataset via per-class oversampling to the median class size.

Two distinct pipelines are provided:
    train_transform   – heavy augmentation for training
    eval_transform    – deterministic resize + normalise only
"""

import random
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


# ── low-level OpenCV helpers ──────────────────────────────────────────────────

def _random_brightness_contrast(img: np.ndarray,
                                 brightness: float = config.AUG_BRIGHTNESS,
                                 contrast:   float = config.AUG_CONTRAST) -> np.ndarray:
    """Apply random brightness and contrast (HSV space)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # brightness: V channel
    factor_v = 1.0 + random.uniform(-brightness, brightness)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor_v, 0, 255)
    # saturation: S channel
    factor_s = 1.0 + random.uniform(-contrast, contrast)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor_s, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _random_gaussian_blur(img: np.ndarray, prob: float = config.AUG_BLUR_PROB) -> np.ndarray:
    if random.random() < prob:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img


def _add_gaussian_noise(img: np.ndarray, prob: float = config.AUG_NOISE_PROB) -> np.ndarray:
    if random.random() < prob:
        sigma  = random.uniform(5, 25)
        noise  = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img    = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def _random_cutout(img: np.ndarray,
                   prob: float = config.AUG_CUTOUT_PROB,
                   min_size: float = 0.1,
                   max_size: float = 0.3) -> np.ndarray:
    """Zero out a random rectangular patch (simulates occlusion)."""
    if random.random() < prob:
        h, w = img.shape[:2]
        ch = int(h * random.uniform(min_size, max_size))
        cw = int(w * random.uniform(min_size, max_size))
        y1 = random.randint(0, h - ch)
        x1 = random.randint(0, w - cw)
        out = img.copy()
        out[y1:y1 + ch, x1:x1 + cw] = 0
        return out
    return img


def _simulate_skin_tone(img: np.ndarray, prob: float = 0.3) -> np.ndarray:
    """Shift hue slightly to simulate different skin tones under varying lighting."""
    if random.random() < prob:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
        shift = random.randint(-10, 10)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + shift, 0, 179)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def _random_shadow(img: np.ndarray, prob: float = 0.25) -> np.ndarray:
    """Add a random semi-transparent shadow across part of the image."""
    if random.random() < prob:
        h, w = img.shape[:2]
        top_y    = random.randint(0, h // 2)
        bottom_y = random.randint(h // 2, h)
        x_left   = random.randint(0, w // 4)
        x_right  = random.randint(3 * w // 4, w)
        mask = np.zeros((h, w), dtype=np.float32)
        pts  = np.array([[x_left, top_y], [x_right, top_y],
                          [x_right, bottom_y], [x_left, bottom_y]], np.int32)
        cv2.fillPoly(mask, [pts], 1.0)
        factor = random.uniform(0.4, 0.8)
        out    = img.astype(np.float32)
        out    = out * (1 - mask[:, :, None] * (1 - factor))
        return np.clip(out, 0, 255).astype(np.uint8)
    return img


def opencv_augment(img_bgr: np.ndarray) -> np.ndarray:
    """
    Apply the full OpenCV-based augmentation chain.
    Input/output: uint8 BGR numpy array.
    """
    img = _random_brightness_contrast(img_bgr)
    img = _simulate_skin_tone(img)
    img = _random_shadow(img)
    img = _random_gaussian_blur(img)
    img = _add_gaussian_noise(img)
    img = _random_cutout(img)
    return img


# ── torchvision transform pipelines ───────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


class OpenCVAugmentTransform:
    """Callable that applies opencv_augment on a PIL image (for use in transforms)."""

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        bgr = opencv_augment(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


def build_train_transform(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    """Heavy augmentation pipeline for training."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        OpenCVAugmentTransform(),
        T.RandomHorizontalFlip(p=config.AUG_FLIP_PROB),
        T.RandomRotation(degrees=config.AUG_ROTATION),
        T.RandomPerspective(distortion_scale=0.3,
                            p=config.AUG_PERSPECTIVE_PROB),
        T.ColorJitter(
            brightness=config.AUG_BRIGHTNESS,
            contrast=config.AUG_CONTRAST,
            saturation=config.AUG_SATURATION,
            hue=config.AUG_HUE,
        ),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    """Deterministic pipeline for validation and test."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ── class balancing via oversampling ─────────────────────────────────────────

def balance_dataset(per_class_paths: dict) -> list:
    """
    Oversample minority classes to the median class count.

    Parameters
    ----------
    per_class_paths : {label: [path, ...]}

    Returns
    -------
    List of (path, label_index) tuples.
    """
    label_list = sorted(per_class_paths.keys())
    label2idx  = {l: i for i, l in enumerate(label_list)}
    counts     = {l: len(v) for l, v in per_class_paths.items()}
    target     = int(np.median(list(counts.values())))

    balanced = []
    rng = random.Random(config.SEED)
    for label, paths in per_class_paths.items():
        if len(paths) == 0:
            continue
        if len(paths) < target:
            # oversample with replacement
            extra  = rng.choices(paths, k=target - len(paths))
            paths  = paths + extra
        else:
            paths = paths[:target]
        for p in paths:
            balanced.append((str(p), label2idx[label]))

    rng.shuffle(balanced)
    return balanced
