# -*- coding: utf-8 -*-
"""
SiaStegNet — A Siamese CNN for Image Steganalysis.

Reference
---------
W. You, H. Zhang and X. Zhao, "A Siamese CNN for Image Steganalysis",
IEEE Transactions on Information Forensics and Security,
vol. 16, pp. 291-306, 2021. doi: 10.1109/TIFS.2020.3013204.

Architecture overview
---------------------
The network uses a **shared backbone** (SRNet-style) with a Siamese
comparison module.  During Siamese training, two images are passed through
the shared branch and their feature vectors are compared via element-wise
absolute difference.  At inference time the single-image classification head
is used directly.

Two forward modes
-----------------
Single-image mode (standard ALASKA2 training) ::

    logits = model(x)              # (N, 2)  cover / stego

Siamese mode (pair-based contrastive training) ::

    diff_logit = model.siamese_forward(x1, x2)   # (N, 1)  same / different

    x1, x2 : (N, 3, H, W)
    diff_logit > 0 → "different class" (cover vs stego)
    Use BCEWithLogitsLoss with label=1 for (cover, stego) pairs.

Adaptation for RGB
------------------
SRM preprocessing is applied channel-independently (groups=in_channels)
and projected to a fixed-width feature volume before the shared SRNet-style
backbone — identical to the other networks in this package.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base    import BaseSteganalyzer
from ..kernels import get_srm_kernels


# ── SRNet backbone building blocks ────────────────────────────────────────────

class _TypeI(nn.Sequential):
    """Conv + BN + ReLU (no residual)."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _TypeII(nn.Module):
    """Residual block, no spatial downsampling."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + x)


class _TypeIII(nn.Module):
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


class _TypeIV(nn.Module):
    """Residual block with GlobalAvgPool, supports channel expansion."""
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


# ── Backbone ──────────────────────────────────────────────────────────────────

class _SiameseBranch(nn.Module):
    """
    Shared feature extractor used by both Siamese branches.

    Architecture mirrors SRNet (Boroumand et al., 2019) exactly, but is
    preceded by a trainable SRM preprocessing layer + 1×1 projection.
    This ensures the same strong noise-residual sensitivity that makes
    SRNet one of the strongest spatial steganalysis backbones.
    """

    def __init__(self, in_channels: int = 3, srm_trainable: bool = False) -> None:
        super().__init__()

        # ── SRM preprocessing ─────────────────────────────────────────────────
        srm_np     = get_srm_kernels()
        srm_tiled  = np.tile(srm_np, (in_channels, 1, 1, 1))
        srm_tensor = torch.from_numpy(srm_tiled)

        self.hpf = nn.Conv2d(
            in_channels, 30 * in_channels,
            kernel_size=5, padding=2,
            groups=in_channels, bias=False,
        )
        with torch.no_grad():
            self.hpf.weight.copy_(srm_tensor)
        self.hpf.weight.requires_grad_(srm_trainable)

        self.bn0  = nn.BatchNorm2d(30 * in_channels)
        self.proj = nn.Sequential(
            nn.Conv2d(30 * in_channels, 30, 1, bias=False),  # project to 30 ch
            nn.BatchNorm2d(30),
            nn.ReLU(inplace=True),
        )

        # ── SRNet backbone (mirrors SRNet exactly) ────────────────────────────
        self.type1s = nn.Sequential(_TypeI(30, 64), _TypeI(64, 16))
        self.type2s = nn.Sequential(
            _TypeII(16), _TypeII(16), _TypeII(16), _TypeII(16), _TypeII(16),
        )
        self.type3s = nn.Sequential(
            _TypeIII(16, 16), _TypeIII(16, 64),
            _TypeIII(64, 128), _TypeIII(128, 256),
        )
        self.type4 = _TypeIV(256, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.bn0(self.hpf(x)))   # (N, 3, H, W)
        x = self.type1s(x)
        x = self.type2s(x)
        x = self.type3s(x)
        x = self.type4(x)
        return x.flatten(1)                    # (N, 512)


# ── Main model ─────────────────────────────────────────────────────────────────

class SIAStegNet(BaseSteganalyzer):
    """
    SiaStegNet — Siamese CNN for Image Steganalysis (You et al., TIFS 2021).

    Architecture
    ------------
    Shared branch : SRM (30 filters, per-channel grouped) → SRNet backbone
                    → GlobalAvgPool → 256-dim feature vector
    Single-image  : feature → FC(256 → num_classes)
    Siamese path  : |f(x1) − f(x2)| → FC(256 → 1)
                    (output is a logit; use BCEWithLogitsLoss)

    Parameters
    ----------
    in_channels   : input channels (3 for RGB)
    num_classes   : output logits for single-image mode (default 2)
    srm_trainable : fine-tune SRM filters
    """

    def __init__(
        self,
        in_channels:   int  = 3,
        num_classes:   int  = 2,
        srm_trainable: bool = False,
    ) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)

        self.branch = _SiameseBranch(in_channels, srm_trainable)

        # Single-image classification head
        self.fc_single = nn.Linear(512, num_classes)

        # Siamese comparison head
        # Receives |f(x1) − f(x2)| and outputs a single logit (different=1)
        self.fc_siamese = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
        )

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for name, m in self.named_modules():
            if "hpf" in name:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward passes ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single-image classification.

        Parameters
        ----------
        x : (N, C, H, W)

        Returns
        -------
        (N, num_classes) logits
        """
        features = self.branch(x)           # (N, 256)
        return self.fc_single(features)

    def siamese_forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Siamese pair comparison.

        Processes both images through the **shared** branch, then computes
        the absolute difference of their feature vectors and passes it
        through the Siamese comparison head.

        Parameters
        ----------
        x1 : (N, C, H, W)  — first image (e.g. suspected stego)
        x2 : (N, C, H, W)  — second image (e.g. reference cover)

        Returns
        -------
        (N, 1) logit — positive → images are from *different* classes.
        Use ``torch.nn.BCEWithLogitsLoss`` with label=1 for (cover, stego)
        pairs and label=0 for same-class pairs.
        """
        f1 = self.branch(x1)                # (N, 256)
        f2 = self.branch(x2)                # (N, 256)
        diff = torch.abs(f1 - f2)           # (N, 256)
        return self.fc_siamese(diff)        # (N, 1)
