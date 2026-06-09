# -*- coding: utf-8 -*-
"""
Decoder architectures for SteganoGAN.

  BasicDecoder           : 3-layer CNN
  DenseDecoder           : DenseNet-style with dense skip connections
  EdgeAwareDenseDecoder  : Edge-aware decoder with lightweight DenseASPP + MSMA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..base import BaseDecoder
from ..encoders.edge_unet.attention import MSMAModule


class BasicDecoder(BaseDecoder):
    """
    3-layer CNN decoder.

    Three conv+BN+LeakyReLU layers, then a 1×1 output projection to D channels.

    Input : stego  (N, 3, H, W)
    Output: logits (N, D, H, W)  — apply threshold ≥0 for recovered bits
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.out   = nn.Conv2d(32, data_depth, kernel_size=3, padding=1)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1(self.conv1(stego)), inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), inplace=True)
        return self.out(x)


class DenseDecoder(BaseDecoder):
    """
    DenseNet-style decoder with dense skip connections.

    Each layer receives all previous feature maps concatenated, giving
    it direct access to low-level and high-level representations.

    Input : stego  (N, 3, H, W)
    Output: logits (N, D, H, W)
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        self.conv1 = nn.Conv2d(3,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.out   = nn.Conv2d(96, data_depth, kernel_size=3, padding=1)

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        x1 = F.leaky_relu(self.bn1(self.conv1(stego)), inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(torch.cat([x1, x2], dim=1))), inplace=True)
        return self.out(torch.cat([x1, x2, x3], dim=1))


class EdgeAwareDenseDecoder(BaseDecoder):
    """
    Edge-aware decoder with lightweight DenseASPP and MSMA attention.

    Learns its own edge map from the stego image to determine WHERE
    data was embedded, then uses a lightweight DenseASPP (2 dilated
    branches) + dense decode layers for extraction.

    Improvements over DenseDecoder:
      - Edge estimation branch guides extraction to edge regions
      - Lightweight DenseASPP (dilation rates 3, 6) for multi-scale context
      - MSMA attention for channel/spatial recalibration
      - 48 channels (vs 32) for more representational capacity

    Input : stego  (N, 3, H, W)
    Output: logits (N, D, H, W)

    Parameters
    ----------
    data_depth : bits per pixel (D)
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)

        ch = 48  # main feature channel width

        # ── Edge estimation branch ──────────────────────────────────────
        self.edge_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # ── Stem + MSMA (input: stego 3ch + edge_map 1ch = 4ch) ────────
        self.stem = nn.Sequential(
            nn.Conv2d(4, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.msma = MSMAModule(ch)

        # ── Lightweight DenseASPP (2 branches) ──────────────────────────
        branch_ch = 32
        self.aspp_d3 = nn.Sequential(
            nn.Conv2d(ch, branch_ch, kernel_size=3,
                      padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.aspp_d6 = nn.Sequential(
            nn.Conv2d(ch + branch_ch, branch_ch, kernel_size=3,
                      padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Reduce: ch + 2*branch_ch = 112 → ch
        self.reduce = nn.Sequential(
            nn.Conv2d(ch + 2 * branch_ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Dense decode layers ─────────────────────────────────────────
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(2 * ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(3 * ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ── Output projection ───────────────────────────────────────────
        self.out = nn.Conv2d(3 * ch, data_depth, kernel_size=3, padding=1)

    def _forward_impl(self, stego: torch.Tensor) -> torch.Tensor:
        edge_map  = self.edge_branch(stego)
        augmented = torch.cat([stego, edge_map], dim=1)
        f0 = self.msma(self.stem(augmented))
        d1 = self.aspp_d3(f0)
        d2 = self.aspp_d6(torch.cat([f0, d1], dim=1))
        f1 = self.reduce(torch.cat([f0, d1, d2], dim=1))
        h1 = self.dec1(f1)
        h2 = self.dec2(torch.cat([f1, h1], dim=1))
        h3 = self.dec3(torch.cat([f1, h1, h2], dim=1))
        return self.out(torch.cat([h1, h2, h3], dim=1))

    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        # Gradient checkpointing avoids storing dense-connection intermediates.
        # Called T times in the loss loop, so savings are T× (e.g. ~2.4 GB at T=4).
        if self.training and stego.requires_grad:
            return checkpoint(self._forward_impl, stego, use_reentrant=False)
        return self._forward_impl(stego)
