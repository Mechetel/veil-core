# -*- coding: utf-8 -*-
"""
DepthAgnosticEncoder

Encodes any number of bit-planes D (1 ≤ D ≤ data_depth) into one stego
image using a single fixed-shape backbone.  D is never baked into any
weight shape.

Architecture (inside forward)
-------------------------------
For each bit-plane i the backbone receives a 5-channel tile:
    [cover (3ch) | bit_plane_i (1ch) | normalised_index_i (1ch)]
producing a 3-channel residual.  All D tiles are processed in one batched
GPU call (no Python loop).  The residuals are summed and tanh-clipped onto
the cover.

    cover_exp  = cover.repeat_interleave(D, dim=0)        (N·D, 3, H, W)
    bits_flat  = payload.view(N·D, 1, H, W)               (N·D, 1, H, W)
    idx_maps   = [i/(max(D-1,1))] tiled to (N·D, 1, H, W)
    x          = cat([cover_exp, bits_flat, idx_maps])     (N·D, 5, H, W)
    residuals  = backbone(x)                              (N·D, 3, H, W)
    stego      = tanh(cover + residuals.view(N,D,3,H,W).sum(1))

The normalised index tells the backbone which "slot" to embed into, so
the decoder can later extract each bit-plane independently.

Training
--------
Trainer draws D randomly from [1, data_depth] each batch (via isinstance
check) so one trained model generalises to all depths at inference.

Inputs  : cover   (N, 3, H, W)  in [-1, 1]
          payload (N, D, H, W)  in {0, 1}  (D chosen by Trainer)
Output  : stego   (N, 3, H, W)  in [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEncoder


class DepthAgnosticEncoder(BaseEncoder):

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        # 3 cover + 1 bit-plane + 1 index = always 5 input channels
        self.conv1 = nn.Conv2d(5,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.out   = nn.Conv2d(32,  3, kernel_size=3, padding=1)

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        N, D, H, W = payload.shape

        # ── fold D into batch ────────────────────────────────────────────
        cover_exp = cover.repeat_interleave(D, dim=0)             # (N·D, 3, H, W)
        bits_flat = payload.view(N * D, 1, H, W)                  # (N·D, 1, H, W)

        indices  = torch.arange(D, device=cover.device, dtype=cover.dtype) / max(D - 1, 1)
        idx_maps = indices.repeat(N).view(N * D, 1, 1, 1).expand(N * D, 1, H, W)

        x = torch.cat([cover_exp, bits_flat, idx_maps], dim=1)    # (N·D, 5, H, W)

        # ── shared backbone (one GPU call for all D planes) ──────────────
        x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), inplace=True)
        residuals = self.out(x)                                    # (N·D, 3, H, W)

        # ── unfold: sum residuals per image, apply tanh ──────────────────
        residuals = residuals.view(N, D, 3, H, W).sum(dim=1)      # (N,   3, H, W)
        return torch.tanh(cover + residuals)
