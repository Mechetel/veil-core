# -*- coding: utf-8 -*-
"""
U-Net building blocks: contracting, bottleneck, and expanding steps.
Ji, Zhang, Lv – Applied Sciences 2025, Table 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention  import MSMAModule
from .inception  import InceptionDMKModule


class ContractingBlock(nn.Module):
    """
    One U-Net contracting step.

    Conv3×3 → BN → LeakyReLU (→ optional MSMA) → MaxPool2×2

    Returns the pooled feature map AND the pre-pool skip tensor.

    Parameters
    ----------
    in_ch     : input channels
    out_ch    : output channels
    use_msma  : attach MSMA attention after activation
    """

    def __init__(self, in_ch: int, out_ch: int, use_msma: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.attn = MSMAModule(out_ch) if use_msma else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)
        if self.attn is not None:
            x = self.attn(x)
        return self.pool(x), x      # (pooled, skip)


class BottleneckBlock(nn.Module):
    """
    U-Net bottleneck (no spatial down-sampling).

    Conv3×3 → BN → LeakyReLU → optional MSMA → optional InceptionDMK

    Parameters
    ----------
    in_ch         : input channels
    out_ch        : output channels
    use_msma      : attach MSMA attention
    use_inception : attach InceptionDMK module
    """

    def __init__(self, in_ch: int, out_ch: int,
                 use_msma: bool = False,
                 use_inception: bool = False) -> None:
        super().__init__()
        self.conv      = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn        = nn.BatchNorm2d(out_ch)
        self.attn      = MSMAModule(out_ch)       if use_msma      else None
        self.inception = InceptionDMKModule(out_ch) if use_inception else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn(self.conv(x)), 0.2, inplace=True)
        if self.attn      is not None: x = self.attn(x)
        if self.inception is not None: x = self.inception(x)
        return x


class ExpandingBlock(nn.Module):
    """
    One U-Net expanding step.

    TransposedConv 5×5 (stride 2) → BN → ReLU

    Doubles the spatial resolution and halves the channel count.

    Parameters
    ----------
    in_ch  : input channels (before the skip-connection concat)
    out_ch : output channels
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch,
            kernel_size=5, stride=2, padding=2, output_padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.deconv(x)), inplace=True)
