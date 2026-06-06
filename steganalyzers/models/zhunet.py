# -*- coding: utf-8 -*-
"""
ZhuNet — Depth-Wise Separable Convolutions and Multi-Level Pooling for an
Efficient Spatial CNN-Based Steganalysis.

Reference
---------
Zhu et al., "Depth-Wise Separable Convolutions and Multi-Level Pooling for
an Efficient Spatial CNN-Based Steganalysis",
IEEE Transactions on Information Forensics and Security, 2020.
GitHub: https://github.com/1204BUPT/Zhu-Net-image-steganalysis

Architecture overview
---------------------
Pre-processing layer
    Two parallel trainable SRM filter banks applied to the input:
      • 25 compact 3×3 SRM filters  (pre_Layer_3_3)
      • 5  wide    5×5 SRM filters  (pre_Layer_5_5)
    Output channels are concatenated: 25 + 5 = 30.

Depth-wise separable block × 2  (conv_Layer)
    Block 1: DepthwiseSepConv(30→30, 3×3) → |abs| → BN → ReLU
    Block 2: DepthwiseSepConv(30→30, 3×3) →        BN + residual → ReLU
    (each "DWS conv" = groups=out_ch depthwise conv + 1×1 pointwise, matching
    the original: Conv2d(30→60, 3×3, groups=30) + Conv2d(60→30, 1×1))

Feature extraction (4 Basic Blocks)
    Block 1: Conv(30→32,  3×3) + BN + ReLU + AvgPool(5, s=2, p=2)
    Block 2: Conv(32→32,  3×3) + BN + ReLU + AvgPool(5, s=2, p=2)
    Block 3: Conv(32→64,  3×3) + BN + ReLU + AvgPool(5, s=2, p=2)
    Block 4: Conv(64→128, 3×3) + BN + ReLU  (no pooling)

Spatial Pyramid Pooling (SPP)
    AdaptiveAvgPool at 3 levels: 1×1, 2×2, 4×4
    Concatenated flattened: 128×(1+4+16) = 2688 features

Classifier
    FC(2688→1024) → ReLU → FC(1024→2)

Adaptation for RGB
------------------
The original network processes grayscale (1-channel) images.  For RGB,
each SRM filter bank is applied channel-independently (grouped convolution),
then a 1×1 conv folds the per-channel features back to the original 30-channel
volume so all downstream layers are unchanged.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base    import BaseSteganalyzer
from ..kernels import get_srm_kernels_3x3, get_srm_kernels_5x5_bank


# ── Pre-processing layers ──────────────────────────────────────────────────────

class _PreLayer3x3(nn.Module):
    """
    25 trainable 3×3 SRM filters (one bank per input channel).

    For RGB input (in_channels=3) uses grouped conv so each channel is
    filtered independently, then a 1×1 conv projects back to 25 channels.
    """

    def __init__(self, in_channels: int = 1, srm_trainable: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels

        srm_np = get_srm_kernels_3x3()                               # (25, 1, 3, 3)
        tiled  = np.tile(srm_np, (in_channels, 1, 1, 1))             # (25×C, 1, 3, 3)

        self.conv = nn.Conv2d(
            in_channels, 25 * in_channels,
            kernel_size=3, padding=1,
            groups=in_channels, bias=True,
        )
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(tiled))
            nn.init.zeros_(self.conv.bias)
        if not srm_trainable:
            self.conv.weight.requires_grad_(False)

        # Fold per-channel features → 25 channels (identity when in_channels=1)
        self.proj = (
            nn.Conv2d(25 * in_channels, 25, 1, bias=False)
            if in_channels > 1 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.proj is not None:
            x = self.proj(x)
        return x                # (N, 25, H, W)


class _PreLayer5x5(nn.Module):
    """
    5 trainable 5×5 SRM filters (one bank per input channel).
    """

    def __init__(self, in_channels: int = 1, srm_trainable: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels

        srm_np = get_srm_kernels_5x5_bank()                          # (5, 1, 5, 5)
        tiled  = np.tile(srm_np, (in_channels, 1, 1, 1))             # (5×C, 1, 5, 5)

        self.conv = nn.Conv2d(
            in_channels, 5 * in_channels,
            kernel_size=5, padding=2,
            groups=in_channels, bias=True,
        )
        with torch.no_grad():
            self.conv.weight.copy_(torch.from_numpy(tiled))
            nn.init.zeros_(self.conv.bias)
        if not srm_trainable:
            self.conv.weight.requires_grad_(False)

        self.proj = (
            nn.Conv2d(5 * in_channels, 5, 1, bias=False)
            if in_channels > 1 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.proj is not None:
            x = self.proj(x)
        return x                # (N, 5, H, W)


class _PreLayer(nn.Module):
    """
    Combined pre-processing: concat(3×3 bank, 5×5 bank) → 30 channels.
    """

    def __init__(self, in_channels: int = 1, srm_trainable: bool = False) -> None:
        super().__init__()
        self.layer3x3 = _PreLayer3x3(in_channels, srm_trainable)
        self.layer5x5 = _PreLayer5x5(in_channels, srm_trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.layer3x3(x), self.layer5x5(x)], dim=1)  # (N, 30, H, W)


# ── Depth-wise separable conv block ───────────────────────────────────────────

class _DWSConv(nn.Module):
    """
    Depth-wise separable convolution: depthwise(3×3) + pointwise(1×1).

    Matches the original implementation:
        Conv2d(ch, ch*2, 3, groups=ch)  +  Conv2d(ch*2, ch, 1)
    """

    def __init__(self, channels: int, expand: int = 2) -> None:
        super().__init__()
        mid = channels * expand
        self.dw = nn.Conv2d(channels, mid, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ── Basic feature-extraction block ────────────────────────────────────────────

class _BasicBlock(nn.Module):
    """Conv(3×3) + BN + ReLU + optional AvgPool(5, s=2, p=2)."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.AvgPool2d(5, stride=2, padding=2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        if self.pool is not None:
            x = self.pool(x)
        return x


# ── Spatial Pyramid Pooling ────────────────────────────────────────────────────

class _SPPLayer(nn.Module):
    """
    3-level Spatial Pyramid Pooling: AdaptiveAvgPool at 1×1, 2×2, 4×4.

    Output size: channels × (1 + 4 + 16) = channels × 21.
    For channels=128 this gives 2688 features, matching the paper's FC input.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool4 = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        p1 = self.pool1(x).view(N, -1)    # (N, C)
        p2 = self.pool2(x).view(N, -1)    # (N, 4C)
        p4 = self.pool4(x).view(N, -1)    # (N, 16C)
        return torch.cat([p1, p2, p4], dim=1)   # (N, 21C)


# ── Main model ─────────────────────────────────────────────────────────────────

class ZhuNet(BaseSteganalyzer):
    """
    ZhuNet steganalyzer (RGB adaptation).

    Architecture
    ------------
    Pre-process : 25 trainable 3×3 SRM  + 5 trainable 5×5 SRM  → 30 ch
    DWS block 1 : DepthwiseSepConv(30→30) + |·| + BN + ReLU
    DWS block 2 : DepthwiseSepConv(30→30) + BN + residual + ReLU
    Basic blocks: (30→32 + pool), (32→32 + pool), (32→64 + pool), (64→128)
    SPP         : AdaptiveAvgPool(1×1, 2×2, 4×4) → 2688-dim vector
    Classifier  : FC(2688→1024) → ReLU → FC(1024→num_classes)

    Parameters
    ----------
    in_channels : input channels (3 for RGB)
    num_classes : output logits (2 for cover / stego)
    """

    _SPP_DIM: int = 128 * 21   # 2688

    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 srm_trainable: bool = False) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)

        # ── Pre-processing ──────────────────────────────────────────────────
        self.pre = _PreLayer(in_channels, srm_trainable)    # → (N, 30, H, W)

        # ── Depth-wise separable blocks ─────────────────────────────────────
        self.dws1 = _DWSConv(30)
        self.bn1  = nn.BatchNorm2d(30)

        self.dws2 = _DWSConv(30)
        self.bn2  = nn.BatchNorm2d(30)

        # ── Feature extraction ───────────────────────────────────────────────
        self.block1 = _BasicBlock(30,  32,  pool=True)
        self.block2 = _BasicBlock(32,  32,  pool=True)
        self.block3 = _BasicBlock(32,  64,  pool=True)
        self.block4 = _BasicBlock(64,  128, pool=False)

        # ── Spatial Pyramid Pooling ──────────────────────────────────────────
        self.spp = _SPPLayer()

        # ── Classifier ──────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self._SPP_DIM, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for name, m in self.named_modules():
            # Skip SRM pre-processing layers — already initialised with SRM values
            if "pre" in name:
                continue
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-processing (trainable SRM, frozen in the original but we allow
        # fine-tuning — controlled via optimizer param group selection)
        x = self.pre(x)                         # (N, 30, H, W)

        # DWS block 1 — absolute value emphasises symmetric noise residuals
        x_dws1 = self.dws1(x).abs()
        x = F.relu(self.bn1(x_dws1), inplace=True)

        # DWS block 2 — residual connection from after block-1 BN
        x_dws2 = self.dws2(x)
        x = F.relu(self.bn2(x_dws2) + x, inplace=True)

        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # SPP + classification
        x = self.spp(x)                         # (N, 2688)
        return self.classifier(x)
