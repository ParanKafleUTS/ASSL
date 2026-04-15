"""
MobileNetV2 fine-tuned for ASL recognition.

Strategy
--------
* Load ImageNet-pretrained MobileNetV2.
* Replace the final classifier with a new head for num_classes.
* Unfreeze the last N feature blocks for fine-tuning.
* Input size: 96×96 (acceptable for MobileNetV2, reduces FLOPS).
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


def build_mobilenetv2(
    num_classes: int,
    pretrained: bool = True,
    unfreeze_blocks: int = 10,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Parameters
    ----------
    num_classes     : output classes
    pretrained      : load ImageNet weights
    unfreeze_blocks : number of MobileNetV2 InvertedResidual blocks to unfreeze
                      (counted from the end; 0 = feature extractor frozen)
    dropout         : dropout before classifier
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model   = mobilenet_v2(weights=weights)

    # freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze last N blocks
    features = model.features  # ModuleList of 19 blocks (0..18)
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


class MobileNetV2Wrapper(nn.Module):
    """Thin wrapper to expose feature_extract."""

    def __init__(self, num_classes: int, pretrained: bool = True,
                 unfreeze_blocks: int = 10, dropout: float = 0.3):
        super().__init__()
        self.model = build_mobilenetv2(
            num_classes, pretrained, unfreeze_blocks, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1280-d pooled feature before classifier."""
        x = self.model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return x.flatten(1)
