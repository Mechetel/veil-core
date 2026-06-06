# -*- coding: utf-8 -*-
"""
Dense message-fusion block.
Ji, Zhang, Lv – Applied Sciences 2025, Section 3.1.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseMessageFusion(nn.Module):
    """
    Three densely-connected conv layers that fuse U-Net output features
    with the secret message tensor M.

    At every layer both the accumulated features AND the original message M
    are re-concatenated to prevent information decay.

    Channel progression (D = data_depth, F = unet_out_ch):
      Layer 1 input: [F + D]           → 32 ch
      Layer 2 input: [32 + F + D]      → 32 ch
      Layer 3 input: [32 + 32 + F + D] → out_ch

    Parameters
    ----------
    unet_out_ch : channels output by the U-Net expanding path
    data_depth  : secret message channels (D = bits per pixel)
    out_ch      : output channels (default 32)
    """

    def __init__(self, unet_out_ch: int, data_depth: int, out_ch: int = 32) -> None:
        super().__init__()
        D, F = data_depth, unet_out_ch
        self.conv1 = nn.Conv2d(F + D,          out_ch, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch + F + D, out_ch, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch * 2 + F + D, out_ch, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(out_ch)

    def forward(self, unet_feat: torch.Tensor,
                message: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        unet_feat : (N, F, H, W)
        message   : (N, D, H, W)

        Returns
        -------
        (N, out_ch, H, W)
        """
        base = torch.cat([unet_feat, message], dim=1)       # [F+D]

        x1 = F.leaky_relu(self.bn1(self.conv1(base)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(
            torch.cat([x1, base], dim=1))), 0.2, inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(
            torch.cat([x1, x2, base], dim=1))), 0.2, inplace=True)
        return x3
