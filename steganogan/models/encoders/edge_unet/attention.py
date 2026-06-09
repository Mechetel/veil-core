# -*- coding: utf-8 -*-
"""
Multi-Scale Median Attention (MSMA) module.
Ji, Zhang, Lv – Applied Sciences 2025, Section 3.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x: torch.Tensor, groups: int = 4) -> torch.Tensor:
    """ShuffleNet-style channel shuffle across *groups*."""
    N, C, H, W = x.shape
    assert C % groups == 0, f"C={C} must be divisible by groups={groups}"
    return (
        x.view(N, groups, C // groups, H, W)
        .transpose(1, 2)
        .contiguous()
        .view(N, C, H, W)
    )


class MedianPool2d(nn.Module):
    """Global spatial median pooling → (N, C, 1, 1)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        return x.view(N, C, -1).median(dim=-1).values.view(N, C, 1, 1)


class SharedMLP(nn.Module):
    """Two-layer channel-wise MLP  C → C//r → C  (implemented as 1×1 convs)."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class MSMAModule(nn.Module):
    """
    Multi-Scale Median Attention (MSMA).

    Three sequential sub-modules:
      (a) Channel attention  — AvgPool + MaxPool + MedianPool → shared MLP → sum
      (b) Channel shuffle    — inter-group information mixing (4 groups)
      (c) Spatial attention  — hierarchical depthwise convs (5×5, 1×7, 1×11) → 1×1

    Input / Output shape: (N, C, H, W)  where C is divisible by 4.

    Parameters
    ----------
    channels  : number of input/output channels
    reduction : MLP bottleneck reduction ratio (default 4)
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()

        # (a) channel attention pooling heads
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.med_pool = MedianPool2d()
        self.channel_mlp = SharedMLP(channels, reduction)

        # (c) hierarchical depthwise spatial attention
        self.dw_5x5  = nn.Conv2d(channels, channels, kernel_size=5,
                                  padding=2, groups=channels, bias=False)
        self.dw_1x7  = nn.Conv2d(channels, channels, kernel_size=(1, 7),
                                  padding=(0, 3), groups=channels, bias=False)
        self.dw_1x11 = nn.Conv2d(channels, channels, kernel_size=(1, 11),
                                  padding=(0, 5), groups=channels, bias=False)
        self.spatial_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (a) channel attention
        Fc = (
            self.channel_mlp(self.avg_pool(x))
            + self.channel_mlp(self.max_pool(x))
            + self.channel_mlp(self.med_pool(x))
        )
        x_weighted = Fc * x                                      # F' = Fc ⊙ F

        # (b) channel shuffle
        x_shuffled = channel_shuffle(x_weighted, groups=4)

        # (c) spatial attention
        base     = self.dw_5x5(x_shuffled)
        multiscale = self.dw_1x7(base) + self.dw_1x11(base)    # multi-scale sum
        spatial_map = torch.sigmoid(self.spatial_proj(multiscale))
        return spatial_map * x_weighted                          # F'' = σ(·) ⊙ F'
