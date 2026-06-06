# -*- coding: utf-8 -*-
"""
YeNet — Deep Learning Hierarchical Representations for Image Steganalysis.

Reference
---------
Ye et al., "Deep Learning Hierarchical Representations for Image
Steganalysis", IEEE Transactions on Information Forensics and Security, 2017.

Adaptation
----------
The original network operates on grayscale images and uses 30 SRM filters as
a fixed (or trainable) preprocessing layer.  This RGB adaptation applies the
same 30 SRM filters **independently per channel** (groups=in_channels) to
preserve channel-specific high-frequency residuals, yielding
30 × in_channels feature maps.  A subsequent 1×1 conv projects these to 30
channels before the five hierarchical groups, keeping the downstream
architecture identical to the paper.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSteganalyzer
from ..kernels import get_srm_kernels


class TLU(nn.Module):
    """Truncated Linear Unit: clamps activations to [−T, T].

    Proposed in Ye et al. (2017) to bound the preprocessing output while
    preserving gradient flow in the high-pass residual domain.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(-self.threshold, self.threshold)


class YeNet(BaseSteganalyzer):
    """
    YeNet steganalyzer (RGB adaptation).

    Architecture
    ------------
    HPF layer  : 30 SRM filters applied per channel (trainable, initialised
                 with SRM values) → 30×C feature maps → TLU → BN
    Projection : Conv(30×C → 30, 1×1) + BN + ReLU   [channel mixing]
    Group 1    : Conv(30→32, 3×3) + BN + ReLU  + AvgPool(3×3, s=2)
    Group 2    : Conv(32→32, 3×3) + BN + ReLU  + AvgPool(3×3, s=2)
    Group 3    : Conv(32→64, 3×3) + BN + ReLU  + AvgPool(3×3, s=2)
    Group 4    : Conv(64→128, 3×3) + BN + ReLU + AvgPool(3×3, s=2)
    Group 5    : Conv(128→256, 3×3) + BN + ReLU
    GAP        : AdaptiveAvgPool(1×1)
    Classifier : FC(256 → num_classes)

    Parameters
    ----------
    in_channels     : number of input channels (3 for RGB)
    num_classes     : number of output logits (2 for cover/stego)
    tlu_threshold   : threshold for TLU activation (default 3.0)
    srm_trainable   : if True the SRM preprocessing layer is trainable
    """

    def __init__(
        self,
        in_channels:   int   = 3,
        num_classes:   int   = 2,
        tlu_threshold: float = 3.0,
        srm_trainable: bool  = False,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)

        self.srm_trainable = srm_trainable

        # ── SRM preprocessing (grouped conv: one filter bank per channel) ──────
        srm_np = get_srm_kernels()                           # (30, 1, 5, 5)
        srm_tiled = np.tile(srm_np, (in_channels, 1, 1, 1)) # (30×C, 1, 5, 5)
        srm_tensor = torch.from_numpy(srm_tiled)

        # groups=in_channels: each channel is filtered independently
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
        if not srm_trainable:
            self.hpf.weight.requires_grad_(False)

        self.tlu = TLU(threshold=tlu_threshold)
        self.bn0 = nn.BatchNorm2d(30 * in_channels)

        # 1×1 projection: merge per-channel SRM features into 30 joint channels
        self.proj = nn.Sequential(
            nn.Conv2d(30 * in_channels, 30, kernel_size=1, bias=False),
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
        )

        # ── Hierarchical convolutional groups ───────────────────────────────────
        self.group1 = self._make_group(30,  32)
        self.group2 = self._make_group(32,  32)
        self.group3 = self._make_group(32,  64)
        self.group4 = self._make_group(64,  128)
        self.group5 = self._make_group(128, 256, pool=False)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Linear(256, num_classes)

        self._init_weights()

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_group(
        in_ch: int, out_ch: int, pool: bool = True
    ) -> nn.Sequential:
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
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocessing
        x = self.tlu(self.hpf(x))           # (N, 30×C, H, W)
        x = self.bn0(x)
        x = self.proj(x)                     # (N, 30, H, W)

        # Hierarchical groups
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)

        x = self.gap(x).flatten(1)
        return self.fc(x)
