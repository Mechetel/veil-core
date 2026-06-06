# -*- coding: utf-8 -*-
"""
Abstract base class for all steganalysis networks.

Every concrete steganalyzer must inherit from BaseSteganalyzer.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseSteganalyzer(nn.Module, ABC):
    """
    Abstract base class for all steganalysis (detection) networks.

    A steganalyzer takes an RGB image and produces a 2-class logit vector:
      class 0 → cover (no hidden message)
      class 1 → stego (hidden message present)

    Subclasses must implement :meth:`forward`.

    Parameters
    ----------
    in_channels : number of input image channels (default 3 for RGB)
    num_classes : number of output classes (default 2: cover / stego)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:
        super().__init__()
        self.in_channels: int  = in_channels
        self.num_classes: int  = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify *x* as cover or stego.

        Parameters
        ----------
        x : (N, C, H, W)  RGB image, typically in [0, 1] or [0, 255]

        Returns
        -------
        (N, num_classes)  logit tensor — apply softmax / argmax for labels
        """

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
