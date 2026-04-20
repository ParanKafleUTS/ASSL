"""
Custom 3-layer CNN baseline.

Architecture
------------
Conv(32) → Conv(64) → Conv(128) → GlobalAvgPool → FC(256) → FC(num_classes)

Trained from scratch; establishes a lower-bound on performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


class _ConvBlock(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, pool: bool = True):
        layers = [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        super().__init__(*layers)


class CustomCNN(nn.Module):
    """
    Simple 3-conv-layer CNN for ASL classification.

    Parameters
    ----------
    num_classes : number of output classes (default: 29)
    dropout     : dropout probability before the final FC layer
    """

    def __init__(self,
                 num_classes: int = 29,
                 dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            _ConvBlock(3,  32, pool=True),   # 64 → 32
            _ConvBlock(32, 64, pool=True),   # 32 → 16
            _ConvBlock(64, 128, pool=True),  # 16 → 8
        )

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(128, 256)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)           # (B, 128, *, *)
        x = self.pool(x).flatten(1)    # (B, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)             # (B, num_classes)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-d feature embedding (before final FC)."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return F.relu(self.fc1(x))


def build_custom_cnn(num_classes: int, dropout: float = 0.4) -> CustomCNN:
    return CustomCNN(num_classes=num_classes, dropout=dropout)
