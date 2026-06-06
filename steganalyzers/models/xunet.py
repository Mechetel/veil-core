# -*- coding: utf-8 -*-
"""
XuNet — Structural Design of Convolutional Neural Networks for Steganalysis.

Reference
---------
Xu et al., "Structural Design of Convolutional Neural Networks for
Steganalysis", IEEE Signal Processing Letters, 2016.

Adaptation
----------
Original paper uses grayscale (1-channel) input.  This implementation
handles RGB (3-channel) images by applying the Kapur–Voelz (KV) high-pass
filter independently to each channel (grouped convolution), preserving all
per-channel residual information before the learnable layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSteganalyzer


class XuNet(BaseSteganalyzer):
    """
    XuNet steganalyzer (RGB adaptation).

    Architecture
    ------------
    Preprocessing : Fixed KV high-pass filter applied per channel
    Group 1       : Conv(3→8, 5×5)  → |·| → BN → tanh + AvgPool(5, s=2)
    Group 2       : Conv(8→16, 5×5)        → BN → tanh + AvgPool(5, s=2)
    Group 3       : Conv(16→32, 1×1)       → BN → relu + AvgPool(5, s=2)
    Group 4       : Conv(32→64, 1×1)       → BN → relu + AvgPool(5, s=2)
    Group 5       : Conv(64→128, 1×1)      → BN → relu + AdaptiveAvgPool(1)
    Classifier    : FC(128 → 2)

    Parameters
    ----------
    in_channels : input channels (default 3 for RGB)
    num_classes : number of output logits (default 2)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)

        # Kapur–Voelz (KV) high-pass filter — registered as a non-trainable buffer
        kv = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=torch.float32) / 12.0
        # Shape: (1, 1, 5, 5) — will be tiled across groups in forward()
        self.register_buffer("kv_kernel", kv.view(1, 1, 5, 5))

        # ── Learnable layers ────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(in_channels, 8,   kernel_size=5, padding=2, bias=False)
        self.bn1   = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8,  16,  kernel_size=5, padding=2, bias=False)
        self.bn2   = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32,  kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64,  kernel_size=1, bias=False)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn5   = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128, num_classes)

        self._init_weights()

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _hpf(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the KV filter channel-independently (grouped convolution)."""
        N, C, H, W = x.shape
        # Tile the single-channel kernel for all input channels
        kernel = self.kv_kernel.expand(C, 1, 5, 5)   # (C, 1, 5, 5)
        return F.conv2d(x, kernel, padding=2, groups=C)

    # ── Weight initialisation ──────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward pass ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocessing: per-channel high-pass filter
        x = self._hpf(x)                                       # (N, C, H, W)

        # Group 1 — absolute value non-linearity (models symmetric noise)
        x = torch.abs(self.conv1(x))
        x = torch.tanh(self.bn1(x))
        x = F.avg_pool2d(x, kernel_size=5, stride=2, padding=2)

        # Group 2
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=5, stride=2, padding=2)

        # Group 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=5, stride=2, padding=2)

        # Group 4
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, kernel_size=5, stride=2, padding=2)

        # Group 5
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Classifier
        x = x.flatten(1)
        return self.fc(x)
