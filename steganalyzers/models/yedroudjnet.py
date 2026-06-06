# -*- coding: utf-8 -*-
"""
YedroudjNet (Yedroudj-Net) — An Efficient CNN for Spatial Steganalysis.

Reference
---------
Yedroudj et al., "Yedroudj-Net: An Efficient CNN for Spatial Steganalysis",
IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), 2018.

Adaptation
----------
The original network uses a fixed 30-filter SRM high-pass preprocessing layer
for grayscale input.  Here the SRM layer is applied **independently per
channel** (grouped convolution), and a 1×1 projection layer collapses the
30×in_channels channels back to 30 before the downstream blocks — exactly
preserving the paper's architecture for the feature-learning stage.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSteganalyzer
from ..kernels import get_srm_kernels


class YedroudjNet(BaseSteganalyzer):
    """
    Yedroudj-Net steganalyzer (RGB adaptation).

    Architecture
    ------------
    HPF        : Fixed SRM (30 × in_channels filters, grouped) → |·| → clamp → BN
    Projection : Conv(30×C → 30, 1×1) + BN + ReLU
    Block 1    : Conv(30→32,  3×3) + BN + ReLU + AvgPool(2×2)
    Block 2    : Conv(32→32,  3×3) + BN + ReLU + AvgPool(2×2)
    Block 3    : Conv(32→64,  3×3) + BN + ReLU + AvgPool(2×2)
    Block 4    : Conv(64→128, 3×3) + BN + ReLU + AvgPool(2×2)
    Block 5    : Conv(128→256, 3×3) + BN + ReLU
    GAP        : AdaptiveAvgPool(1×1)
    Classifier : FC(256 → num_classes)

    Parameters
    ----------
    in_channels : input channels (3 for RGB)
    num_classes : output logits (2 for cover/stego)
    abs_layer   : apply absolute value after HPF (suppresses sign, keeps magnitude)
    clamp_val   : upper clamp for HPF output (0 = no clamping)
    """

    def __init__(
        self,
        in_channels: int   = 3,
        num_classes: int   = 2,
        abs_layer:   bool  = True,
        clamp_val:   float = 3.0,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)
        self.abs_layer = abs_layer
        self.clamp_val = clamp_val

        # ── Fixed SRM preprocessing ─────────────────────────────────────────────
        srm_np     = get_srm_kernels()                           # (30, 1, 5, 5)
        srm_tiled  = np.tile(srm_np, (in_channels, 1, 1, 1))   # (30×C, 1, 5, 5)
        srm_tensor = torch.from_numpy(srm_tiled)

        self.hpf = nn.Conv2d(
            in_channels,
            30 * in_channels,
            kernel_size=5,
            padding=2,
            groups=in_channels,
            bias=False,
        )
        with torch.no_grad():
            self.hpf.weight.copy_(srm_tensor)
        self.hpf.weight.requires_grad_(False)   # fixed — not trainable

        self.bn0 = nn.BatchNorm2d(30 * in_channels)

        # 1×1 projection → standard 30-channel feature volume
        self.proj = nn.Sequential(
            nn.Conv2d(30 * in_channels, 30, kernel_size=1, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
        )

        # ── Feature-learning blocks ─────────────────────────────────────────────
        self.block1 = self._make_block(30,  32)
        self.block2 = self._make_block(32,  32)
        self.block3 = self._make_block(32,  64)
        self.block4 = self._make_block(64,  128)
        self.block5 = self._make_block(128, 256, pool=False)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(256, num_classes)

        self._init_weights()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, pool: bool = True) -> nn.Sequential:
        layers: list = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for name, m in self.named_modules():
            if "hpf" in name:
                continue    # keep SRM initialisation
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fixed high-pass preprocessing
        x = self.hpf(x)                                          # (N, 30×C, H, W)
        if self.abs_layer:
            x = x.abs()
        if self.clamp_val > 0:
            x = x.clamp(max=self.clamp_val)
        x = self.bn0(x)
        x = self.proj(x)                                         # (N, 30, H, W)

        # Feature learning
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.gap(x).flatten(1)
        return self.fc(x)
