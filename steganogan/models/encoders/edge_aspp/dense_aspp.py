# -*- coding: utf-8 -*-
"""
Enhanced DenseASPP feature backbone with MSMA attention.

Inspired by HCISNet (Higher-Capacity Invisible Steganographic Network),
this module extracts multi-scale features at full spatial resolution using
dilated convolutions with dense connections.  No pooling layers are used,
preserving precise spatial alignment for per-pixel embedding decisions.

MSMA attention modules are placed after the stem and after the DenseASPP
block to recalibrate features at different stages of the pipeline.
An InceptionDMK module provides multi-kernel refinement at the end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..edge_unet.attention import MSMAModule
from ..edge_unet.inception import InceptionDMKModule


class DenseASPPBackbone(nn.Module):
    """
    DenseASPP feature backbone with integrated MSMA attention.

    Architecture
    ------------
    Stem:      Conv(3→64) → BN → LReLU → Conv(64→64) → BN → LReLU
    MSMA_stem: MSMAModule(64)
    DenseASPP: 4 dilated branches (rates 3, 6, 12, 18) with dense connections
               64 → +32 → +32 → +32 → +32 = 192 channels
    Reduce:    Conv(192→out_ch, 1×1) → BN → LReLU
    MSMA_aspp: MSMAModule(out_ch)
    Inception: InceptionDMKModule(out_ch)

    Parameters
    ----------
    in_ch  : input channels (default 3 for RGB)
    out_ch : output feature channels (default 64, must be divisible by 4 for MSMA/Inception)

    Input : (N, in_ch, H, W)
    Output: (N, out_ch, H, W)
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 64) -> None:
        super().__init__()
        assert out_ch % 4 == 0, f"out_ch must be divisible by 4, got {out_ch}"

        stem_ch = 64
        branch_ch = 32

        # ── Stem ────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(stem_ch, stem_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.msma_stem = MSMAModule(stem_ch)

        # ── DenseASPP (4 dilated branches with dense connections) ──────
        # Branch 1: input = stem_ch (64)
        self.aspp_d3 = nn.Sequential(
            nn.Conv2d(stem_ch, branch_ch, kernel_size=3,
                      padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Branch 2: input = stem_ch + branch_ch (96)
        self.aspp_d6 = nn.Sequential(
            nn.Conv2d(stem_ch + branch_ch, branch_ch, kernel_size=3,
                      padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Branch 3: input = stem_ch + 2*branch_ch (128)
        self.aspp_d12 = nn.Sequential(
            nn.Conv2d(stem_ch + 2 * branch_ch, branch_ch, kernel_size=3,
                      padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Branch 4: input = stem_ch + 3*branch_ch (160)
        self.aspp_d18 = nn.Sequential(
            nn.Conv2d(stem_ch + 3 * branch_ch, branch_ch, kernel_size=3,
                      padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Total after dense concat: stem_ch + 4*branch_ch = 192
        total_ch = stem_ch + 4 * branch_ch

        # ── Channel reduction + MSMA + InceptionDMK ────────────────────
        self.reduce = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.msma_aspp = MSMAModule(out_ch)
        self.inception = InceptionDMKModule(out_ch)

    # ── Segmented helpers for gradient checkpointing ─────────────────────

    def _stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.msma_stem(self.stem(x))

    def _aspp_forward(self, f_att: torch.Tensor) -> torch.Tensor:
        """Dense ASPP block — all 7 intermediate tensors live only during
        forward; gradient checkpointing recomputes them on the backward pass
        instead of keeping them in memory (~2.7 GB saved at batch=8, 360²).
        """
        d1    = self.aspp_d3(f_att)
        cat1  = torch.cat([f_att, d1], dim=1)                   # 96 ch
        d2    = self.aspp_d6(cat1)
        cat2  = torch.cat([f_att, d1, d2], dim=1)               # 128 ch
        d3    = self.aspp_d12(cat2)
        cat3  = torch.cat([f_att, d1, d2, d3], dim=1)           # 160 ch
        d4    = self.aspp_d18(cat3)
        return torch.cat([f_att, d1, d2, d3, d4], dim=1)        # 192 ch

    def _reduce_forward(self, cat_all: torch.Tensor) -> torch.Tensor:
        return self.inception(self.msma_aspp(self.reduce(cat_all)))

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, in_ch, H, W)

        Returns
        -------
        features : (N, out_ch, H, W)
        """
        if self.training:
            # x.requires_grad may be False (cover image has no grad) but the
            # model parameters do, so we still need to store intermediate
            # activations for parameter gradients. Checkpoint by training mode
            # only — not by requires_grad on the input.
            f_att   = checkpoint(self._stem_forward,   x,       use_reentrant=False)
            cat_all = checkpoint(self._aspp_forward,   f_att,   use_reentrant=False)
            return    checkpoint(self._reduce_forward,  cat_all, use_reentrant=False)

        f_att   = self._stem_forward(x)
        cat_all = self._aspp_forward(f_att)
        return self._reduce_forward(cat_all)
