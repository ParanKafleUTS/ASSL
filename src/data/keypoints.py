"""
CNN-based hand keypoint extractor (no MediaPipe).

Architecture
------------
A lightweight encoder (3 conv blocks) produces 21 heatmaps (one per landmark).
The final keypoint (x, y) is extracted via soft-argmax on each heatmap.

The model can be:
  * Trained end-to-end on labelled keypoint data (if available).
  * Used as a *feature extractor* on the ASL classification dataset to produce
    42-dimensional descriptors (21 × 2) that feed the Skeleton GCN.

When no ground-truth keypoints are available the network uses the heatmap
activations as *proxy keypoints* by computing soft-argmax on predicted
heatmaps. This still gives spatially-meaningful features that the GCN can use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


# ── building blocks ──────────────────────────────────────────────────────────

class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_c: int, out_c: int,
                 kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


class _DepthwiseSeparable(nn.Sequential):
    """Depthwise-separable conv (MobileNet style) for speed."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_c, in_c, 3, stride=stride,
                      padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )


# ── soft-argmax helper ────────────────────────────────────────────────────────

def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft-argmax for a batch of heatmaps.

    Parameters
    ----------
    heatmaps : (B, K, H, W)

    Returns
    -------
    coords   : (B, K, 2)   normalised to [0, 1]
    """
    B, K, H, W = heatmaps.shape
    # softmax over spatial positions
    flat   = heatmaps.view(B, K, -1)
    weight = F.softmax(flat, dim=-1).view(B, K, H, W)

    # coordinate grids in [0, 1]
    xs = torch.linspace(0, 1, W, device=heatmaps.device)
    ys = torch.linspace(0, 1, H, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)

    cx = (weight * grid_x.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # (B, K)
    cy = (weight * grid_y.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # (B, K)
    return torch.stack([cx, cy], dim=-1)  # (B, K, 2)


# ── main network ──────────────────────────────────────────────────────────────

class KeypointCNN(nn.Module):
    """
    Lightweight CNN that predicts 21 hand-landmark heatmaps.

    Input  : (B, 3, H, W)  – RGB image, H=W=96 by default
    Output :
        heatmaps : (B, 21, H/8, W/8)
        coords   : (B, 21, 2)   normalised to [0, 1]
    """

    def __init__(self,
                 in_channels:   int = 3,
                 num_keypoints: int = config.NUM_KEYPOINTS,
                 input_size:    int = config.KEYPOINT_SIZE):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.encoder = nn.Sequential(
            # stage 1 – stride 2  →  H/2
            _ConvBnRelu(in_channels, 32, stride=2),
            _DepthwiseSeparable(32, 64),

            # stage 2 – stride 2  →  H/4
            _DepthwiseSeparable(64, 128, stride=2),
            _DepthwiseSeparable(128, 128),

            # stage 3 – stride 2  →  H/8
            _DepthwiseSeparable(128, 256, stride=2),
            _DepthwiseSeparable(256, 256),
        )

        # 1×1 conv → num_keypoints heatmaps
        self.head = nn.Conv2d(256, num_keypoints, 1)

        # small decoder that also outputs a 256-d feature vector for GCN
        hmap_h = input_size // 8
        self.feat_pool = nn.AdaptiveAvgPool2d(1)
        self.feat_proj = nn.Linear(256, 256)

    def forward(self, x: torch.Tensor):
        feats     = self.encoder(x)            # (B, 256, H/8, W/8)
        heatmaps  = self.head(feats)           # (B, K, H/8, W/8)
        coords    = soft_argmax_2d(heatmaps)   # (B, K, 2)

        # global feature (for downstream tasks)
        g_feat = self.feat_pool(feats).flatten(1)   # (B, 256)
        g_feat = self.feat_proj(g_feat)              # (B, 256)

        return heatmaps, coords, g_feat

    def extract_keypoints(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method – returns flat (B, K*2) keypoint vector."""
        _, coords, _ = self.forward(x)
        return coords.flatten(1)          # (B, 42)


# ── hand topology adjacency matrix ───────────────────────────────────────────

def build_hand_adjacency(num_nodes: int = config.NUM_KEYPOINTS,
                          self_loops: bool = True) -> torch.Tensor:
    """
    Build the standard 21-node hand topology adjacency matrix.

    Wrist = 0
    Thumb  : 1–4   (CMC → MCP → IP → TIP)
    Index  : 5–8
    Middle : 9–12
    Ring   : 13–16
    Pinky  : 17–20

    Returns normalised adjacency (D^{-1/2} A D^{-1/2}).
    """
    edges = [
        # palm → each finger base
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # thumb
        (1, 2), (2, 3), (3, 4),
        # index
        (5, 6), (6, 7), (7, 8),
        # middle
        (9, 10), (10, 11), (11, 12),
        # ring
        (13, 14), (14, 15), (15, 16),
        # pinky
        (17, 18), (18, 19), (19, 20),
        # knuckle connections (MCP row)
        (5, 9), (9, 13), (13, 17),
    ]

    A = torch.zeros(num_nodes, num_nodes)
    for (i, j) in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    if self_loops:
        A += torch.eye(num_nodes)

    # symmetric normalisation
    deg  = A.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[d_inv_sqrt == float("inf")] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm
