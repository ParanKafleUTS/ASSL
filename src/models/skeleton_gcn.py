"""
Skeleton GCN for ASL recognition.

Graph Convolutional Network following Kipf & Welling (2017).
The 21-node hand graph is built in src/data/keypoints.py.

Pipeline
--------
1.  A frozen / fine-tuned KeypointCNN extracts (B, 21, 2) coordinates.
2.  Coordinates are projected to node features via a learnable linear layer.
3.  K rounds of graph convolution aggregate neighbour information.
4.  Global mean pooling → MLP → logits.

The KeypointCNN is trained jointly (optional) or used pre-trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from src.data.keypoints import KeypointCNN, build_hand_adjacency


# ── GCN layer ─────────────────────────────────────────────────────────────────

class GraphConvLayer(nn.Module):
    """
    A single graph convolution: H' = A_norm H W

    Parameters
    ----------
    in_features  : input node feature dimensionality
    out_features : output node feature dimensionality
    bias         : learnable bias term
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, N, in_features)
        adj : (N, N)  normalised adjacency
        """
        # linear transform
        support = x @ self.weight           # (B, N, out_features)
        # graph aggregation
        out = adj.unsqueeze(0) @ support    # (B, N, out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


# ── full skeleton GCN ─────────────────────────────────────────────────────────

class SkeletonGCN(nn.Module):
    """
    End-to-end model: image → keypoints → GCN → class logits.

    Parameters
    ----------
    num_classes     : ASL class count
    hidden_dim      : GCN hidden dimension
    num_gcn_layers  : number of GCN layers
    dropout         : regularisation strength
    freeze_kp_cnn   : if True, keypoint CNN weights are not updated
    """

    def __init__(self,
                 num_classes: int,
                 hidden_dim:    int   = config.GCN_HIDDEN_DIM,
                 num_gcn_layers: int  = config.GCN_NUM_LAYERS,
                 dropout:       float = 0.4,
                 freeze_kp_cnn: bool  = False):
        super().__init__()

        self.kp_cnn = KeypointCNN()
        if freeze_kp_cnn:
            for p in self.kp_cnn.parameters():
                p.requires_grad = False

        # project raw (x, y) coords to node feature vectors
        self.node_embed = nn.Linear(config.NUM_COORDS, hidden_dim)

        # stack of GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_dim  = hidden_dim
            out_dim = hidden_dim
            self.gcn_layers.append(GraphConvLayer(in_dim, out_dim))

        self.dropout = nn.Dropout(dropout)
        self.bn_list = nn.ModuleList(
            [nn.BatchNorm1d(config.NUM_KEYPOINTS) for _ in range(num_gcn_layers)]
        )

        # classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # register adjacency as buffer (moves to GPU automatically)
        adj = build_hand_adjacency(config.NUM_KEYPOINTS)
        self.register_buffer("adj", adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W)
        """
        # extract keypoints: (B, 21, 2)
        _, coords, _ = self.kp_cnn(x)

        # project to node features: (B, 21, hidden_dim)
        h = F.relu(self.node_embed(coords))

        # graph convolution rounds
        for layer, bn in zip(self.gcn_layers, self.bn_list):
            h = F.relu(layer(h, self.adj))
            h = bn(h)                       # BN over node dim
            h = self.dropout(h)

        # global mean pooling over nodes: (B, hidden_dim)
        h = h.mean(dim=1)

        # MLP head
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        return self.fc2(h)

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled graph feature before classifier."""
        _, coords, _ = self.kp_cnn(x)
        h = F.relu(self.node_embed(coords))
        for layer, bn in zip(self.gcn_layers, self.bn_list):
            h = F.relu(layer(h, self.adj))
            h = bn(h)
            h = self.dropout(h)
        return h.mean(dim=1)
