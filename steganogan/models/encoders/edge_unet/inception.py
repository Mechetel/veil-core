# -*- coding: utf-8 -*-
"""
InceptionDMK – Inception-style Depthwise Multi-Kernel convolution.
Ji, Zhang, Lv – Applied Sciences 2025, Section 3.3.
"""

import torch
import torch.nn as nn


class InceptionDMKModule(nn.Module):
    """
    Four parallel depthwise-separable branches; each emits ``channels//4``
    feature maps that are concatenated back to ``channels``.

    Branches
    --------
    bypass  (x_pa) : 1×1 pointwise
    x_hw           : 3×3  depthwise-separable conv
    x_w            : 1×11 depthwise-separable conv  (horizontal context)
    x_h            : 11×1 depthwise-separable conv  (vertical context)

    Input / Output shape: (N, C, H, W)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_ch = channels // 4

        self.bypass = nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False)

        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      groups=channels, bias=False),
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
        )
        self.branch_1x11 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5),
                      groups=channels, bias=False),
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
        )
        self.branch_11x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0),
                      groups=channels, bias=False),
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat(
            [self.bypass(x), self.branch_3x3(x),
             self.branch_1x11(x), self.branch_11x1(x)],
            dim=1,
        )
        return self.bn(out)
