"""
Main entry point for the ASL Recognition pipeline.

Steps
-----
1.  Extract the dataset zip (if not already done).
2.  Build train / val / test DataLoaders with augmentation.
3.  Optionally run hyperparameter search per model.
4.  Train all five model architectures with early stopping.
5.  Evaluate each model: accuracy, bootstrap CI, per-class F1,
    confusion matrix, McNemar's tests.
6.  Tune and evaluate the weighted ensemble.
7.  Save all results and visualisations to results/.

Usage
-----
    python main.py [--skip-extract] [--skip-hpo] [--models MODEL [MODEL ...]]

Example
-------
    python main.py
    python main.py --skip-extract --models custom_cnn attention_cnn
    python main.py --skip-hpo --models mobilenetv2 efficientnetb0
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

# ── repo-level imports ────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from src.training.trainer import set_seed
from src.data.dataset import build_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── model registry ────────────────────────────────────────────────────────────

def _build_model(name: str, num_classes: int, hparams: dict):
    if name == "custom_cnn":
        from src.models.custom_cnn import CustomCNN
        return CustomCNN(num_classes, dropout=hparams.get("dropout", 0.4))

    if name == "mobilenetv2":
        from src.models.mobilenetv2 import MobileNetV2Wrapper
        return MobileNetV2Wrapper(
            num_classes,
            unfreeze_blocks=hparams.get("unfreeze_blocks", 10),
            dropout=hparams.get("dropout", 0.3),
        )

    if name == "efficientnetb0":
        from src.models.efficientnetb0 import EfficientNetB0Wrapper
        return EfficientNetB0Wrapper(
            num_classes,
            unfreeze_blocks=hparams.get("unfreeze_blocks", 5),
            dropout=hparams.get("dropout", 0.3),
        )

    if name == "attention_cnn":
        from src.models.attention_cnn import AttentionCNN
        return AttentionCNN(num_classes, dropout=hparams.get("dropout", 0.4))

    if name == "skeleton_gcn":
        from src.models.skeleton_gcn import SkeletonGCN
        return SkeletonGCN(
            num_classes,
            hidden_dim=hparams.get("hidden_dim", config.GCN_HIDDEN_DIM),
            num_gcn_layers=hparams.get("num_gcn_layers", config.GCN_NUM_LAYERS),
            dropout=hparams.get("dropout", 0.4),
        )

    raise ValueError(f"Unknown model: {name}")


# ── pipeline stages ───────────────────────────────────────────────────────────

def stage_extract(zip_path: str) -> None:
    """Extract and organise the raw dataset."""
    from src.data.extract import extract_zip, build_processed_dataset
    extract_zip(zip_path)
    build_processed_dataset()


def stage_hpo(model_name: str,
              train_loader, val_loader,
              num_classes: int) -> dict:
    """Run Optuna HPO and return best hyper-parameter dict."""
    from src.training.hyperparameter import tune
    log.info("=== HPO: %s ===", model_name)
    best = tune(model_name, train_loader, val_loader, num_classes)
    log.info("Best params for %s: %s", model_name, best)
    return best


def stage_train(model_name, model, train_loader, val_loader,
                hparams: dict) -> dict:
    """Train model and return history."""
    from src.training.trainer import train
    log.info("=== Training: %s ===", model_name)
    history = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        model_name   = model_name,
        lr           = hparams.get("lr", config.LR_DEFAULT),
        weight_decay = hparams.get("weight_decay", config.WEIGHT_DECAY),
        label_smoothing = hparams.get("label_smoothing", 0.1),
    )
    return history


def stage_evaluate(model_name, model, test_loader,
                   classes, results_dir: Path) -> dict:
    """Full evaluation + save plots."""
    from src.evaluation.metrics import (
        full_report, collect_predictions
    )
    from src.utils.visualization import (
        plot_confusion_matrix, plot_per_class_f1, plot_training_curves
    )
    import numpy as np

    log.info("=== Evaluating: %s ===", model_name)
    report = full_report(model, test_loader, classes)

    # confusion matrix
    cm = np.array(report.pop("confusion_matrix"))
    plot_confusion_matrix(
        cm, classes,
        save_path=str(results_dir / f"{model_name}_confusion_matrix.png"),
    )
    report["confusion_matrix"] = cm.tolist()

    # per-class F1
    plot_per_class_f1(
        report["per_class_f1"],
        model_name=model_name,
        save_path=str(results_dir / f"{model_name}_per_class_f1.png"),
    )

    return report


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ASL Recognition Pipeline")
    parser.add_argument(
        "--zip", default=config.ZIP_PATH,
        help="Path to archive (2).zip",
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip zip extraction (dataset already processed)",
    )
    parser.add_argument(
        "--skip-hpo", action="store_true",
        help="Skip hyperparameter optimisation (use defaults)",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=config.ENSEMBLE_MODELS,
        choices=config.ENSEMBLE_MODELS,
        help="Which models to train",
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.BATCH_SIZE,
    )
    parser.add_argument(
        "--image-size", type=int, default=config.IMAGE_SIZE,
    )
    args = parser.parse_args()

    set_seed()
    log.info("Device: %s", config.DEVICE)

    # ── 1. extraction ──
    if not args.skip_extract:
        stage_extract(args.zip)
    else:
        log.info("Skipping extraction.")

    # ── 2. dataloaders ──
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        image_size  = args.image_size,
        batch_size  = args.batch_size,
    )
    num_classes = len(classes)
    config.CLASSES = classes
    log.info("Classes (%d): %s", num_classes, classes)

    results_dir = Path(config.RESULTS_DIR)
    all_reports:    dict = {}
    all_predictions: dict = {}
    trained_models: dict = {}

    # ── 3 & 4. HPO + training ──
    for name in args.models:
        log.info("─" * 60)

        # optional HPO
        if not args.skip_hpo:
            hparams = stage_hpo(name, train_loader, val_loader, num_classes)
        else:
            hparams = {}

        model = _build_model(name, num_classes, hparams)
        history = stage_train(name, model, train_loader, val_loader, hparams)

        # save training curves
        from src.utils.visualization import plot_training_curves
        plot_training_curves(
            history, model_name=name,
            save_path=str(results_dir / f"{name}_training_curves.png"),
        )

        # ── 5. evaluation ──
        report = stage_evaluate(name, model, test_loader, classes, results_dir)
        all_reports[name]     = report
        trained_models[name]  = model

        # collect predictions for McNemar
        from src.evaluation.metrics import collect_predictions
        preds, labels = collect_predictions(model, test_loader, config.DEVICE)
        all_predictions[name] = preds

        log.info(
            "[%s] Test accuracy=%.4f  [%.4f–%.4f]  macro_F1=%.4f",
            name,
            report["accuracy"],
            report["accuracy_ci_lower"],
            report["accuracy_ci_upper"],
            report["macro_f1"],
        )

    # ── 6. McNemar pairwise ──
    if len(all_predictions) >= 2:
        from src.evaluation.metrics import pairwise_mcnemar
        import numpy as np
        _, test_labels = collect_predictions(
            trained_models[list(trained_models.keys())[0]],
            test_loader, config.DEVICE,
        )
        mcnemar_results = pairwise_mcnemar(all_predictions, test_labels)
        all_reports["mcnemar"] = {
            f"{a}_vs_{b}": v
            for (a, b), v in mcnemar_results.items()
        }

    # ── 7. Ensemble ──
    if len(trained_models) >= 2:
        log.info("=== Ensemble ===")
        from src.models.ensemble import WeightedEnsemble, tune_ensemble_weights
        best_w = tune_ensemble_weights(trained_models, val_loader, config.DEVICE)
        log.info("Ensemble weights: %s",
                 {k: round(v, 4) for k, v in zip(trained_models.keys(), best_w)})

        ensemble = WeightedEnsemble(trained_models, weights=best_w)
        ensemble.eval()

        ens_report = stage_evaluate(
            "ensemble", ensemble, test_loader, classes, results_dir,
        )
        all_reports["ensemble"] = ens_report
        log.info(
            "[ensemble] Test accuracy=%.4f  macro_F1=%.4f",
            ens_report["accuracy"], ens_report["macro_f1"],
        )

    # ── save JSON summary ──
    summary_path = results_dir / "summary.json"
    _serialisable = {}
    for k, v in all_reports.items():
        # confusion matrix is already list of list
        _serialisable[k] = v
    with open(summary_path, "w") as f:
        json.dump(_serialisable, f, indent=2)
    log.info("Results saved to %s", summary_path)

    # ── print final table ──
    log.info("\n%-20s  %8s  %8s  %8s  %8s", "Model", "Acc", "CI_lo", "CI_hi", "F1")
    log.info("-" * 60)
    for name, rep in all_reports.items():
        if name == "mcnemar":
            continue
        log.info(
            "%-20s  %8.4f  %8.4f  %8.4f  %8.4f",
            name,
            rep.get("accuracy", 0),
            rep.get("accuracy_ci_lower", 0),
            rep.get("accuracy_ci_upper", 0),
            rep.get("macro_f1", 0),
        )


if __name__ == "__main__":
    main()
