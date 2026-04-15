"""
EfficientNetB0 fine-tuned for ASL recognition.

Strategy
--------
* Load ImageNet-pretrained EfficientNetB0.
* Replace the classifier head.
* Unfreeze top blocks; keep earlier blocks frozen to preserve low-level features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def build_efficientnetb0(
    num_classes: int,
    pretrained: bool = True,
    unfreeze_blocks: int = 5,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Parameters
    ----------
    num_classes     : output classes
    pretrained      : load ImageNet weights
    unfreeze_blocks : number of EfficientNet blocks to unfreeze from the end
    dropout         : dropout before the final linear layer
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model   = efficientnet_b0(weights=weights)

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # selectively unfreeze the last N feature blocks
    features = model.features  # Sequential of 9 blocks (0..8)
    total     = len(features)
    for blk in features[max(0, total - unfreeze_blocks):]:
        for param in blk.parameters():
            param.requires_grad = True

    # replace classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


class EfficientNetB0Wrapper(nn.Module):
    """Thin wrapper exposing feature_extract."""

    def __init__(self, num_classes: int, pretrained: bool = True,
                 unfreeze_blocks: int = 5, dropout: float = 0.3):
        super().__init__()
        self.model = build_efficientnetb0(
            num_classes, pretrained, unfreeze_blocks, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1280-d pooled features before classifier."""
        x = self.model.features(x)
        x = self.model.avgpool(x)
        return x.flatten(1)
