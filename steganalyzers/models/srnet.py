# -*- coding: utf-8 -*-
"""
SRNet — Deep Residual Network for Steganalysis of Digital Images.

Reference
---------
Boroumand et al., "Deep Residual Network for Steganalysis of Digital
Images", IEEE Transactions on Information Forensics and Security, 2019.

Adaptation
----------
The original network was designed for single-channel (grayscale) images.
For RGB images the first Type1 layer input channels are changed from 1 to 3.
All other architectural choices remain identical.

Layer types
-----------
Type1 : Conv(3×3, p=1) + BN + ReLU
Type2 : (Conv+BN+ReLU+Conv+BN) + skip  — residual, no pooling
Type3 : Type2 body + AvgPool shortcut   — residual with downsampling
Type4 : Type2 body + skip + GAP         — residual then GlobalAvgPool,
        supports in_ch → out_ch channel expansion via 1×1 shortcut
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseSteganalyzer


# ── Building blocks ────────────────────────────────────────────────────────────

class Type1(nn.Sequential):
    """Plain Conv + BN + ReLU."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class Type2(nn.Module):
    """Residual block — no spatial downsampling."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.shortcut(x))


class Type3(nn.Module):
    """Residual block with AvgPool spatial downsampling."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.AvgPool2d(3, 2, 1),
        )
        self.pool = nn.AvgPool2d(3, 2, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.pool(self.body(x)) + self.shortcut(x))


class Type4(nn.Module):
    """Residual block + GlobalAvgPool, with optional channel expansion."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.gap(self.body(x) + self.shortcut(x)))


# ── Main model ─────────────────────────────────────────────────────────────────

class SRNet(BaseSteganalyzer):
    """
    SRNet steganalyzer (RGB adaptation).

    Architecture
    ------------
    type1s : Type1(in_channels→64), Type1(64→16)
    type2s : Type2(16→16) × 2
    type3s : Type3(16→32), Type3(32→64), Type3(64→128)
    type4  : Type4(128→256) + GAP
    dense  : Linear(256→num_classes)

    Parameters
    ----------
    in_channels : input channels — 3 for RGB (1 in the original paper)
    num_classes : output logits
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)

        # Fixed Kapur-Voelz high-pass filter (same as XuNet) — removes content,
        # amplifies noise residuals so Type1 sees the steganographic signal directly.
        kv = torch.tensor([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8,-12,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=torch.float32) / 12.0
        self.register_buffer("kv_kernel", kv.view(1, 1, 5, 5))

        self.type1s = nn.Sequential(
            Type1(in_channels, 64),
            Type1(64, 16),
        )
        self.type2s = nn.Sequential(
            Type2(16, 16),
            Type2(16, 16),
        )
        self.type3s = nn.Sequential(
            Type3(16,  32),
            Type3(32,  64),
            Type3(64, 128),
        )
        self.type4 = Type4(128, 256)
        self.dense = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _hpf(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        kernel = self.kv_kernel.expand(C, 1, 5, 5)
        return F.conv2d(x, kernel, padding=2, groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._hpf(x)
        x = self.type1s(x)
        x = self.type2s(x)
        x = self.type3s(x)
        x = self.type4(x)          # (N, 512, 1, 1)
        return self.dense(x.flatten(1))
