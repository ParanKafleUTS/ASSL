"""
Optuna-based hyperparameter tuning.

Supports tuning any of the five model architectures.
Each trial trains a model with a sampled HP set and returns validation accuracy.

Usage
-----
    best_params = tune("custom_cnn", train_loader, val_loader, num_classes=29)
"""

import logging
from typing import Any, Dict

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.training.trainer import train

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── model factory ─────────────────────────────────────────────────────────────

def _build_model(model_name: str,
                 num_classes: int,
                 trial: optuna.Trial) -> nn.Module:
    if model_name == "custom_cnn":
        from src.models.custom_cnn import CustomCNN
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        return CustomCNN(num_classes=num_classes, dropout=dropout)

    elif model_name == "mobilenetv2":
        from src.models.mobilenetv2 import MobileNetV2Wrapper
        unfreeze = trial.suggest_int("unfreeze_blocks", 5, 15)
        dropout  = trial.suggest_float("dropout", 0.1, 0.5)
        return MobileNetV2Wrapper(num_classes, unfreeze_blocks=unfreeze,
                                  dropout=dropout)

    elif model_name == "efficientnetb0":
        from src.models.efficientnetb0 import EfficientNetB0Wrapper
        unfreeze = trial.suggest_int("unfreeze_blocks", 3, 7)
        dropout  = trial.suggest_float("dropout", 0.1, 0.5)
        return EfficientNetB0Wrapper(num_classes, unfreeze_blocks=unfreeze,
                                     dropout=dropout)

    elif model_name == "attention_cnn":
        from src.models.attention_cnn import AttentionCNN
        dropout = trial.suggest_float("dropout", 0.2, 0.6)
        return AttentionCNN(num_classes=num_classes, dropout=dropout)

    elif model_name == "skeleton_gcn":
        from src.models.skeleton_gcn import SkeletonGCN
        hidden_dim  = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        num_layers  = trial.suggest_int("num_gcn_layers", 2, 4)
        dropout     = trial.suggest_float("dropout", 0.2, 0.5)
        return SkeletonGCN(num_classes=num_classes,
                           hidden_dim=hidden_dim,
                           num_gcn_layers=num_layers,
                           dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── objective ─────────────────────────────────────────────────────────────────

def _make_objective(
    model_name:    str,
    num_classes:   int,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    hpo_epochs:    int = 20,
):
    def objective(trial: optuna.Trial) -> float:
        lr           = trial.suggest_float("lr",           1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        ls           = trial.suggest_float("label_smoothing", 0.0, 0.2)

        model = _build_model(model_name, num_classes, trial)

        history = train(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            model_name   = f"{model_name}_hpo_trial{trial.number}",
            num_epochs   = hpo_epochs,
            lr           = lr,
            weight_decay = weight_decay,
            patience     = 5,
            label_smoothing = ls,
        )
        return max(history["val_acc"])

    return objective


# ── public API ────────────────────────────────────────────────────────────────

def tune(
    model_name:    str,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    num_classes:   int,
    n_trials:      int = config.HPO_TRIALS,
    timeout:       Any = config.HPO_TIMEOUT,
    hpo_epochs:    int = 20,
) -> Dict[str, Any]:
    """
    Run Optuna hyperparameter search.

    Returns
    -------
    best_params : dict with the best hyperparameter values
    """
    study = optuna.create_study(
        direction     = "maximize",
        study_name    = f"{model_name}_hpo",
        sampler       = optuna.samplers.TPESampler(seed=config.SEED),
        pruner        = optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        _make_objective(model_name, num_classes,
                        train_loader, val_loader, hpo_epochs),
        n_trials = n_trials,
        timeout  = timeout,
        show_progress_bar = True,
    )

    log.info("[HPO] Best trial for %s: val_acc=%.4f  params=%s",
             model_name,
             study.best_value,
             study.best_params)

    return study.best_params
