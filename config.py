"""
Central configuration for the ASL Recognition pipeline.
All hyper-parameters, paths and reproducibility seeds live here.
"""
import os
import torch

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
ZIP_PATH        = os.path.join(BASE_DIR, "archive (2).zip")   # default zip location
EXTRACT_DIR     = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")

for _d in (DATA_DIR, EXTRACT_DIR, PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

IMAGE_SIZE        = 96          # used by MobileNetV2 and EfficientNetB0
CNN_IMAGE_SIZE    = 64          # used by Custom CNN and Attention CNN
KEYPOINT_SIZE     = 96          # input size for keypoint CNN
NUM_KEYPOINTS     = 21          # hand landmarks (wrist + 4 joints × 5 fingers)
NUM_COORDS        = 2           # (x, y) per keypoint
KEYPOINT_DIM      = NUM_KEYPOINTS * NUM_COORDS

# Label set – discovered dynamically; placeholder for typing hints
CLASSES: list[str] = []

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
NUM_WORKERS   = 4
PIN_MEMORY    = True
MAX_EPOCHS    = 80
EARLY_STOP_PATIENCE = 10
LR_DEFAULT    = 1e-3
WEIGHT_DECAY  = 1e-4

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Augmentation ───────────────────────────────────────────────────────────────
AUG_BRIGHTNESS = 0.3
AUG_CONTRAST   = 0.3
AUG_SATURATION = 0.3
AUG_HUE        = 0.1
AUG_ROTATION   = 30          # degrees
AUG_FLIP_PROB  = 0.5
AUG_BLUR_PROB  = 0.2
AUG_NOISE_PROB = 0.2
AUG_CUTOUT_PROB = 0.3
AUG_PERSPECTIVE_PROB = 0.2

# ── Skeleton GCN ───────────────────────────────────────────────────────────────
GCN_HIDDEN_DIM   = 64
GCN_NUM_LAYERS   = 3

# ── Ensemble ───────────────────────────────────────────────────────────────────
ENSEMBLE_MODELS = ["custom_cnn", "mobilenetv2", "efficientnetb0",
                   "attention_cnn", "skeleton_gcn"]

# ── Evaluation ─────────────────────────────────────────────────────────────────
BOOTSTRAP_N = 1000
CI_ALPHA    = 0.95

# ── Hyperparameter search (Optuna) ─────────────────────────────────────────────
HPO_TRIALS   = 30
HPO_TIMEOUT  = None          # seconds; set to a number to cap search time
