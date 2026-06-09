# -*- coding: utf-8 -*-
"""
Lightweight learned edge detection sub-network (EdgeNet).

Inspired by HED (Holistically-Nested Edge Detection, Xie & Tu 2015),
this 3-stage network produces a single-channel edge probability map
at full input resolution.  Unlike fixed Sobel kernels, EdgeNet learns
steganography-optimal edges end-to-end through the embedding loss.

An optional Sobel-consistency regularisation provides a warm-start bias
toward traditional edges without over-constraining the detector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeNet(nn.Module):
    """
    Lightweight 3-stage learned edge detector.

    Architecture
    ------------
    Stage 1 (full res):   Conv(3→16) → BN → ReLU → Conv(16→16) → BN → ReLU
                          side_output: Conv(16→1, 1×1)
    Stage 2 (½ res):      Conv(16→32, stride=2) → BN → ReLU → Conv(32→32) → BN → ReLU
                          side_output: Conv(32→1, 1×1) → Upsample(×2)
    Stage 3 (¼ res):      Conv(32→64, stride=2) → BN → ReLU → Conv(64→64) → BN → ReLU
                          side_output: Conv(64→1, 1×1) → Upsample(×4)
    Fusion:               Conv(3→1, 1×1, no bias) on cat(side_1, side_2, side_3) → Sigmoid

    Parameters
    ----------
    None — architecture is fixed (~55K parameters).

    Input : (N, 3, H, W) cover image
    Output: (N, 1, H, W) edge probability map in [0, 1]
    """

    def __init__(self) -> None:
        super().__init__()

        # ── Stage 1 (full resolution) ──────────────────────────────────
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.side1 = nn.Conv2d(16, 1, kernel_size=1)

        # ── Stage 2 (½ resolution) ─────────────────────────────────────
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.side2 = nn.Conv2d(32, 1, kernel_size=1)

        # ── Stage 3 (¼ resolution) ─────────────────────────────────────
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.side3 = nn.Conv2d(64, 1, kernel_size=1)

        # ── Multi-scale fusion ─────────────────────────────────────────
        self.fuse = nn.Conv2d(3, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 3, H, W) cover image

        Returns
        -------
        edge_map : (N, 1, H, W) in [0, 1]
        """
        _, _, H, W = x.shape

        # Stage 1 — full resolution
        f1 = self.stage1(x)
        s1 = self.side1(f1)  # (N, 1, H, W)

        # Stage 2 — ½ resolution
        f2 = self.stage2(f1)
        s2 = F.interpolate(
            self.side2(f2), size=(H, W), mode="bilinear", align_corners=False
        )

        # Stage 3 — ¼ resolution
        f3 = self.stage3(f2)
        s3 = F.interpolate(
            self.side3(f3), size=(H, W), mode="bilinear", align_corners=False
        )

        # Fuse multi-scale side outputs
        fused = self.fuse(torch.cat([s1, s2, s3], dim=1))
        return torch.sigmoid(fused)
