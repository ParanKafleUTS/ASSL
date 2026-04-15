"""
Attention CNN for ASL recognition.

Combines two complementary attention mechanisms:
  * Channel attention  – Squeeze-and-Excitation (Hu et al., 2018)
  * Spatial attention  – learns where to focus spatially within a feature map

The network uses 4 conv blocks with SE + spatial attention and a global pooling
classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


# ── Squeeze-and-Excitation ────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., 2018).

    Parameters
    ----------
    channels    : number of input feature channels
    reduction   : squeeze ratio
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)         # squeeze
        e = self.fc(s).view(b, c, 1, 1)    # excite
        return x * e                        # scale


# ── Spatial Attention ─────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """
    Spatial attention: max-pool + avg-pool across channels → 2-channel map →
    7×7 conv → sigmoid mask.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        mask = torch.sigmoid(self.conv(cat))         # (B, 1, H, W)
        return x * mask


# ── Attention Conv Block ──────────────────────────────────────────────────────

class AttentionConvBlock(nn.Module):
    """Conv → BN → ReLU → Pool → SE → Spatial attention."""

    def __init__(self, in_c: int, out_c: int,
                 se_reduction: int = 16,
                 pool: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.pool     = nn.MaxPool2d(2) if pool else nn.Identity()
        self.se       = SEBlock(out_c, reduction=se_reduction)
        self.spatial  = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.se(x)
        x = self.spatial(x)
        return x


# ── Full Attention CNN ────────────────────────────────────────────────────────

class AttentionCNN(nn.Module):
    """
    4-block Attention CNN for ASL classification.

    Parameters
    ----------
    num_classes : number of output classes
    dropout     : dropout probability
    """

    def __init__(self, num_classes: int, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            AttentionConvBlock(3,   32,  pool=True),   # 64 → 32
            AttentionConvBlock(32,  64,  pool=True),   # 32 → 16
            AttentionConvBlock(64,  128, pool=True),   # 16 → 8
            AttentionConvBlock(128, 256, pool=True),   # 8  → 4
        )

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(256, 256)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-d embedding (before final FC)."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return F.relu(self.fc1(x))


def build_attention_cnn(num_classes: int, dropout: float = 0.4) -> AttentionCNN:
    return AttentionCNN(num_classes=num_classes, dropout=dropout)
