# -*- coding: utf-8 -*-
"""
DepthAgnosticDecoder

Recovers D independent bit-planes from one stego image using a single
fixed-shape backbone.  D is never baked into any weight shape.

Architecture (inside forward)
-------------------------------
For each bit-plane i the backbone receives a 4-channel tile:
    [stego (3ch) | normalised_index_i (1ch)]
producing a 1-channel logit.  All D tiles are processed in one batched
GPU call (no Python loop).

    stego_exp  = stego.repeat_interleave(D, dim=0)        (N·D, 3, H, W)
    idx_maps   = [i/(max(D-1,1))] tiled to (N·D, 1, H, W)
    x          = cat([stego_exp, idx_maps])                (N·D, 4, H, W)
    logits     = backbone(x)                              (N·D, 1, H, W)
    output     = logits.view(N, D, H, W)                  (N,   D, H, W)

The normalised index matches the one used by DepthAgnosticEncoder so the
backbone can attend to the correct embedding "slot".

self.data_depth drives D at inference; set via SteganoGAN.set_depth().
During training D is synced from the encoder via Trainer._encode_decode.

Input  : stego  (N, 3, H, W)
Output : logits (N, D, H, W)  — threshold ≥ 0 to recover bits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseDecoder


class DepthAgnosticDecoder(BaseDecoder):

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        # 3 stego + 1 index = always 4 input channels
        self.conv1 = nn.Conv2d(4,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.out   = nn.Conv2d(32,  1, kernel_size=3, padding=1)

    # ------------------------------------------------------------------

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        N, _, H, W = stego.shape
        D = self.data_depth                                        # set by Trainer / set_depth()

        # ── fold D into batch ────────────────────────────────────────────
        stego_exp = stego.repeat_interleave(D, dim=0)             # (N·D, 3, H, W)

        indices  = torch.arange(D, device=stego.device, dtype=stego.dtype) / max(D - 1, 1)
        idx_maps = indices.repeat(N).view(N * D, 1, 1, 1).expand(N * D, 1, H, W)

        x = torch.cat([stego_exp, idx_maps], dim=1)               # (N·D, 4, H, W)

        # ── shared backbone (one GPU call for all D planes) ──────────────
        x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), inplace=True)
        logits = self.out(x)                                       # (N·D, 1, H, W)

        # ── unfold: stack into (N, D, H, W) ─────────────────────────────
        return logits.view(N, D, H, W)
