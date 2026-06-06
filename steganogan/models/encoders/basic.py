# -*- coding: utf-8 -*-
"""
Baseline SteganoGAN encoder architectures.

  BasicEncoder    : 3-layer CNN (SteganoGAN baseline)
  ResidualEncoder : BasicEncoder + identity residual skip
  DenseEncoder    : DenseNet-style with dense skip connections (SteganoGAN default)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import BaseEncoder


class BasicEncoder(BaseEncoder):
    """
    3-layer CNN encoder (SteganoGAN baseline).

    Extracts a 32-ch feature map from the cover, concatenates the secret
    payload, applies two more conv layers, and outputs the stego image
    via tanh.

    Input : cover (N,3,H,W), payload (N,D,H,W)
    Output: stego (N,3,H,W) in [-1, 1]
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        self.feature_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.feature_bn   = nn.BatchNorm2d(32)
        self.layer1_conv  = nn.Conv2d(32 + data_depth, 32, kernel_size=3, padding=1)
        self.layer1_bn    = nn.BatchNorm2d(32)
        self.layer2_conv  = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_bn    = nn.BatchNorm2d(32)
        self.output_conv  = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.feature_bn(self.feature_conv(cover)), inplace=True)
        x = F.leaky_relu(self.layer1_bn(self.layer1_conv(torch.cat([x, payload], dim=1))), inplace=True)
        x = F.leaky_relu(self.layer2_bn(self.layer2_conv(x)), inplace=True)
        return torch.tanh(self.output_conv(x))


class ResidualEncoder(BaseEncoder):
    """
    Residual encoder: BasicEncoder backbone + identity skip from cover.

    Adding the cover image as a residual provides a strong gradient path
    and biases the encoder towards minimal perturbation.

    Input : cover (N,3,H,W), payload (N,D,H,W)
    Output: stego (N,3,H,W)
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        self.feature_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.feature_bn   = nn.BatchNorm2d(32)
        self.layer1_conv  = nn.Conv2d(32 + data_depth, 32, kernel_size=3, padding=1)
        self.layer1_bn    = nn.BatchNorm2d(32)
        self.layer2_conv  = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.layer2_bn    = nn.BatchNorm2d(32)
        self.output_conv  = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.feature_bn(self.feature_conv(cover)), inplace=True)
        x = F.leaky_relu(self.layer1_bn(self.layer1_conv(torch.cat([x, payload], dim=1))), inplace=True)
        x = F.leaky_relu(self.layer2_bn(self.layer2_conv(x)), inplace=True)
        return cover + self.output_conv(x)


class DenseEncoder(BaseEncoder):
    """
    DenseNet-style encoder with dense skip connections (SteganoGAN default).

    Every conv layer receives as input the concatenation of all previous
    feature maps AND the secret payload tensor, maximising information flow.

    Input : cover (N,3,H,W), payload (N,D,H,W)
    Output: stego (N,3,H,W)
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__(data_depth)
        D = data_depth
        self.conv1 = nn.Conv2d(3,       32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32  + D, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(64  + D, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(96  + D,  3, kernel_size=3, padding=1)

    def forward(self, cover: torch.Tensor, payload: torch.Tensor) -> torch.Tensor:
        x1 = F.leaky_relu(self.bn1(self.conv1(cover)), inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(torch.cat([x1, payload], dim=1))), inplace=True)
        x3 = F.leaky_relu(self.bn3(self.conv3(torch.cat([x1, x2, payload], dim=1))), inplace=True)
        return cover + self.conv4(torch.cat([x1, x2, x3, payload], dim=1))
