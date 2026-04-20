# ASL Recognition System

GPU-accelerated American Sign Language (ASL) recognition pipeline using multiple deep learning architectures.

## Features
- **No MediaPipe** – custom CNN-based hand keypoint extraction
- **5 model architectures**: Custom CNN, MobileNetV2, EfficientNetB0, Skeleton GCN, Attention CNN
- **Weighted ensemble** with validation-tuned weights
- **Heavy augmentation** via OpenCV (skin tone, shadow, noise, blur, cutout, perspective)
- **Class-balanced training** using WeightedRandomSampler + oversampling
- **Mixed-precision training** (AMP) for GPU speed-up
- **Hyperparameter tuning** via Optuna (TPE sampler)
- **Early stopping** (patience=10)
- **Full evaluation**: Bootstrap 95% CI, McNemar's test, per-class F1, confusion matrix, robustness tests

## Dataset Structure (after extraction)

> **Note**: folder names contain a typo from the original dataset ( instead of ). The code matches these exact names.
```
archive (2)/
    asl_alphabhet_test/asl_alphabhet_test/   ← A_test.jpg … Z_test.jpg, nothing.jpg, space.jpg
    asl_alphabhet_train/asl_alphabhet_train/  ← A/ … Z/ del/ nothing/ space/ ← A/ … Z/ del/ nothing/ space/
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place archive (2).zip in the repo root, then run:
python main.py

# 3. Skip extraction if dataset already processed:
python main.py --skip-extract

# 4. Train specific models only (skip HPO):
python main.py --skip-hpo --models mobilenetv2 efficientnetb0

# 5. Full run with all models and HPO:
python main.py --zip "path/to/archive (2).zip"
```

## Project Structure
```
ASSL/
├── main.py                    # Pipeline entry point
├── config.py                  # All hyperparameters and paths
├── requirements.txt
└── src/
    ├── data/
    │   ├── extract.py         # Zip extraction + 80/10/10 stratified split
    │   ├── augmentation.py    # OpenCV + torchvision augmentation pipeline
    │   ├── dataset.py         # PyTorch Dataset + DataLoader factory
    │   └── keypoints.py       # Lightweight CNN keypoint extractor + hand adjacency
    ├── models/
    │   ├── custom_cnn.py      # 3-conv baseline CNN
    │   ├── mobilenetv2.py     # MobileNetV2 fine-tuning
    │   ├── efficientnetb0.py  # EfficientNetB0 fine-tuning
    │   ├── skeleton_gcn.py    # Skeleton Graph CNN (Kipf & Welling, 2017)
    │   ├── attention_cnn.py   # SE blocks + spatial attention (Hu et al., 2018)
    │   └── ensemble.py        # Weighted soft-voting ensemble
    ├── training/
    │   ├── trainer.py         # Training loop + early stopping + AMP
    │   └── hyperparameter.py  # Optuna HPO
    ├── evaluation/
    │   └── metrics.py         # Accuracy, bootstrap CI, McNemar, F1, robustness
    └── utils/
        └── visualization.py   # OpenCV skeleton overlay, Grad-CAM, plots
```

## Models

| Model | Input | Strategy |
|---|---|---|
| Custom CNN | 64×64 | 3 conv blocks, trained from scratch |
| MobileNetV2 | 96×96 | ImageNet pretrained, last 10 blocks fine-tuned |
| EfficientNetB0 | 96×96 | ImageNet pretrained, last 5 blocks fine-tuned |
| Skeleton GCN | 96×96 | CNN keypoints → 21-node hand graph → GCN |
| Attention CNN | 64×64 | SE channel attention + spatial attention |
| Ensemble | — | Weighted average of softmax probs |

## Evaluation

All metrics are saved in `results/summary.json` and individual plots in `results/`.

- Bootstrap 95% CI (n=1000 resamples) on test accuracy
- McNemar's test for every model pair
- Per-class F1 bar charts
- Normalised confusion matrices
- Robustness under Gaussian noise, blur, and reduced brightness

## Reproducibility
All random seeds are fixed to `42` throughout (Python, NumPy, PyTorch, CUDA).
