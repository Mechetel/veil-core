# -*- coding: utf-8 -*-
"""Custom dataset wrapper for steganography training."""

import math
from typing import Optional

import torchvision
from torchvision import transforms


class SteganographyDataset(torchvision.datasets.ImageFolder):
    """
    Image folder dataset with an optional hard cap on the number of samples.

    Inherits all :class:`torchvision.datasets.ImageFolder` functionality.

    Parameters
    ----------
    root      : path to the image folder (each sub-directory = one class label)
    transform : torchvision transform pipeline
    limit     : maximum number of samples to expose (default: no limit)
    """

    def __init__(
        self,
        root: str,
        transform: transforms.Compose,
        limit: float = math.inf,
    ) -> None:
        super().__init__(root, transform=transform)
        self.limit: float = limit

    def __len__(self) -> int:
        n = super().__len__()
        return n if math.isinf(self.limit) else min(n, int(self.limit))
