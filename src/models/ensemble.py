"""
Weighted ensemble of all five models.

Ensemble strategy
-----------------
* Soft-voting: average the softmax probabilities from each model.
* Weights are learned on the validation set (grid search over a simplex).
* Optional: temperature scaling per model before averaging.

Usage
-----
    ensemble = WeightedEnsemble(models_dict, weights)
    logits   = ensemble(x)          # weighted average of log-probs
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class WeightedEnsemble(nn.Module):
    """
    Parameters
    ----------
    models  : dict  {name: nn.Module}
    weights : list  [w_1, w_2, …]  (will be normalised to sum to 1)
              if None, uniform weights are used
    """

    def __init__(self,
                 models: Dict[str, nn.Module],
                 weights: Optional[List[float]] = None):
        super().__init__()
        self.model_names = list(models.keys())
        self.models = nn.ModuleList(list(models.values()))

        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        w = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", w / w.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns weighted average of softmax probabilities (as log-probs).
        """
        probs_list = []
        for model in self.models:
            model.eval()
            logits = model(x)
            probs_list.append(F.softmax(logits, dim=-1))

        # stack: (num_models, B, C)
        stack  = torch.stack(probs_list, dim=0)
        w      = self.weights.view(-1, 1, 1)       # (num_models, 1, 1)
        avg    = (stack * w).sum(dim=0)             # (B, C)
        return torch.log(avg + 1e-9)               # log-probs


# ── weight tuning on validation set ──────────────────────────────────────────

def tune_ensemble_weights(
    models: Dict[str, nn.Module],
    val_loader: DataLoader,
    device: torch.device = config.DEVICE,
    n_trials: int = 500,
) -> List[float]:
    """
    Search for optimal ensemble weights by random search on the validation set.

    For each trial, sample weights uniformly from the simplex and compute
    validation accuracy.  Returns the weight vector that maximises accuracy.

    Parameters
    ----------
    models      : {name: nn.Module}  (should be in eval mode)
    val_loader  : DataLoader
    device      : torch device
    n_trials    : number of random weight vectors to try

    Returns
    -------
    best_weights : List[float]  (sums to 1, one per model)
    """
    n_models = len(models)
    model_list = list(models.values())

    # pre-collect all softmax outputs to avoid repeated forward passes
    all_probs: List[torch.Tensor] = [[] for _ in range(n_models)]
    all_labels: List[int] = []

    for m in model_list:
        m.eval()
        m.to(device)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            for i, m in enumerate(model_list):
                logits = m(images)
                all_probs[i].append(F.softmax(logits, dim=-1).cpu())
            all_labels.extend(labels.tolist())

    # (n_models, N, C)
    prob_tensors = torch.stack(
        [torch.cat(p, dim=0) for p in all_probs], dim=0
    )
    labels_tensor = torch.tensor(all_labels)

    rng = np.random.default_rng(config.SEED)
    best_acc    = -1.0
    best_weights = [1.0 / n_models] * n_models

    for _ in range(n_trials):
        # sample from simplex
        raw = rng.exponential(scale=1.0, size=n_models).astype(np.float32)
        raw /= raw.sum()
        w = torch.from_numpy(raw).view(-1, 1, 1)

        avg_probs = (prob_tensors * w).sum(dim=0)      # (N, C)
        preds     = avg_probs.argmax(dim=1)
        acc       = (preds == labels_tensor).float().mean().item()

        if acc > best_acc:
            best_acc     = acc
            best_weights = raw.tolist()

    return best_weights
