# -*- coding: utf-8 -*-
"""
WGAN critic (discriminator) for adversarial steganography training.

  BasicCritic               : Original 3-layer WGAN critic with weight clipping
  MultiScaleEdgeAwareCritic : Multi-scale critic with spectral normalisation
                              and edge-aware input preprocessing

References:
  Arjovsky et al., "Wasserstein GAN", ICML 2017.
  Miyato et al., "Spectral Normalization for GANs", ICLR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from ..base import BaseCritic


class BasicCritic(BaseCritic):
    """
    Fully-convolutional WGAN critic.

    Produces a per-pixel score map; the caller averages it to get a scalar
    Wasserstein distance proxy (higher = more real/cover-like).

    Architecture: Conv3×3 → LeakyReLU → BN  (×2) → Conv3×3 → score map

    Input : image (N, 3, H, W)
    Output: score (N, 1, H, W)

    Parameters
    ----------
    data_depth : accepted for API compatibility; not used internally
    """

    def __init__(self, data_depth: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,  1, kernel_size=3, padding=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


def _sn_conv(in_ch: int, out_ch: int, kernel_size: int = 3,
             padding: int = 1) -> nn.Module:
    """Spectrally-normalised Conv2d (no BatchNorm needed)."""
    return spectral_norm(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
    )


class MultiScaleEdgeAwareCritic(BaseCritic):
    """
    Multi-scale edge-aware WGAN critic with spectral normalisation.

    Improvements over BasicCritic:
      - Spectral normalisation replaces weight clipping for Lipschitz constraint
      - Multi-scale discrimination (3 scales) catches pixel, medium, and structural artifacts
      - Edge channel input (Sobel magnitude) forces critic to watch edge regions
      - No BatchNorm (known to destabilise WGAN critics)

    Architecture
    ------------
    Preprocessing: cat(image, sobel_magnitude(image)) → (N, 4, H, W)

    Scale 1 (full res):  SNConv(4→48)→LReLU → SNConv(48→48)→LReLU → SNConv(48→48)→LReLU
                         score_s1: SNConv(48→1)
    Scale 2 (H/2, W/2): AvgPool(2) → SNConv(48→64)→LReLU → SNConv(64→64)→LReLU
                         score_s2: SNConv(64→1)
    Scale 3 (H/4, W/4): AvgPool(2) → SNConv(64→96)→LReLU
                         score_s3: SNConv(96→1)

    Output: mean(score_s1) + mean(score_s2) + mean(score_s3)  → (N, 1, H, W) compatible

    Parameters
    ----------
    data_depth : accepted for API compatibility; not used internally
    """

    def __init__(self, data_depth: int = 1) -> None:
        super().__init__()

        # Fixed Sobel kernels for edge extraction
        Kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        Ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("Kx", Kx)
        self.register_buffer("Ky", Ky)

        # ── Scale 1: Full resolution ────────────────────────────────────
        self.s1_conv1 = _sn_conv(4, 48)
        self.s1_conv2 = _sn_conv(48, 48)
        self.s1_conv3 = _sn_conv(48, 48)
        self.s1_score = _sn_conv(48, 1)

        # ── Scale 2: Half resolution ────────────────────────────────────
        self.pool1    = nn.AvgPool2d(2)
        self.s2_conv1 = _sn_conv(48, 64)
        self.s2_conv2 = _sn_conv(64, 64)
        self.s2_score = _sn_conv(64, 1)

        # ── Scale 3: Quarter resolution ─────────────────────────────────
        self.pool2    = nn.AvgPool2d(2)
        self.s3_conv1 = _sn_conv(64, 96)
        self.s3_score = _sn_conv(96, 1)

    def _sobel_magnitude(self, image: torch.Tensor) -> torch.Tensor:
        """Compute per-channel Sobel edge magnitude, averaged to 1 channel."""
        N, C, H, W = image.shape
        x = image.view(N * C, 1, H, W)
        Gx = F.conv2d(x, self.Kx, padding=1)
        Gy = F.conv2d(x, self.Ky, padding=1)
        mag = (Gx ** 2 + Gy ** 2 + 1e-8).sqrt().view(N, C, H, W)
        return mag.mean(dim=1, keepdim=True)  # (N, 1, H, W)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : (N, 3, H, W)

        Returns
        -------
        score : (N, 1, H, W) — compatible with BaseCritic interface
                (caller applies torch.mean() to get scalar)
        """
        # Edge-augmented input
        edges = self._sobel_magnitude(image)                    # (N, 1, H, W)
        x = torch.cat([image, edges], dim=1)                    # (N, 4, H, W)

        # Scale 1 — full resolution
        s1 = F.leaky_relu(self.s1_conv1(x), 0.2, inplace=True)
        s1 = F.leaky_relu(self.s1_conv2(s1), 0.2, inplace=True)
        s1_feat = F.leaky_relu(self.s1_conv3(s1), 0.2, inplace=True)
        score_s1 = self.s1_score(s1_feat)                       # (N, 1, H, W)

        # Scale 2 — half resolution
        s2 = self.pool1(s1_feat)
        s2 = F.leaky_relu(self.s2_conv1(s2), 0.2, inplace=True)
        s2_feat = F.leaky_relu(self.s2_conv2(s2), 0.2, inplace=True)
        score_s2 = self.s2_score(s2_feat)                       # (N, 1, H/2, W/2)

        # Scale 3 — quarter resolution
        s3 = self.pool2(s2_feat)
        s3_feat = F.leaky_relu(self.s3_conv1(s3), 0.2, inplace=True)
        score_s3 = self.s3_score(s3_feat)                       # (N, 1, H/4, W/4)

        # Multi-scale aggregation: expand smaller scales back to full res
        # and combine so the output matches BaseCritic's (N, 1, H, W) contract
        _, _, H, W = score_s1.shape
        s2_up = F.interpolate(score_s2, size=(H, W), mode="bilinear", align_corners=False)
        s3_up = F.interpolate(score_s3, size=(H, W), mode="bilinear", align_corners=False)
        return score_s1 + s2_up + s3_up
