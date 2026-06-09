# -*- coding: utf-8 -*-
"""
Abstract base classes for all steganography model components.

Every concrete encoder, decoder, and critic must inherit from the
corresponding ABC defined here.  This enforces a consistent interface
and makes type-checking reliable across the codebase.
"""

from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for all encoder architectures.

    An encoder takes a cover image and a secret payload and produces a
    stego image that visually resembles the cover.

    Subclasses must implement :meth:`forward`.  Training-mode encoders
    may return a ``list`` of intermediate stego images (iterative encoders);
    inference-mode encoders must return a single tensor.

    Attributes
    ----------
    data_depth : bits per pixel (D)
    version    : serialisation version string (used by :meth:`upgrade_legacy`)
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__()
        self.data_depth: int = data_depth
        self.version: str = "1"

    @abstractmethod
    def forward(
        self,
        cover: torch.Tensor,
        payload: torch.Tensor,
    ) -> Union[torch.Tensor, list]:
        """
        Embed *payload* into *cover* and return the stego image(s).

        Parameters
        ----------
        cover   : (N, 3, H, W) cover image in [-1, 1]
        payload : (N, D, H, W) binary payload in {0, 1}

        Returns
        -------
        Single tensor (N, 3, H, W)  –  standard encoders
        List of T tensors            –  iterative encoders (training mode)
        """

    def upgrade_legacy(self) -> None:
        """Patch legacy checkpoints that pre-date the version attribute."""
        if not hasattr(self, "version"):
            self.version = "1"


class BaseDecoder(nn.Module, ABC):
    """
    Abstract base class for all decoder architectures.

    A decoder takes a stego image and attempts to recover the secret payload.

    Attributes
    ----------
    data_depth : bits per pixel (D) — also the number of output channels
    version    : serialisation version string
    """

    def __init__(self, data_depth: int) -> None:
        super().__init__()
        self.data_depth: int = data_depth
        self.version: str = "1"

    @abstractmethod
    def forward(self, stego: torch.Tensor) -> torch.Tensor:
        """
        Recover the payload from *stego*.

        Parameters
        ----------
        stego : (N, 3, H, W) stego image

        Returns
        -------
        (N, D, H, W) logit tensor  —  apply sigmoid / threshold for bits
        """

    def upgrade_legacy(self) -> None:
        """Patch legacy checkpoints that pre-date the version attribute."""
        if not hasattr(self, "version"):
            self.version = "1"


class BaseCritic(nn.Module, ABC):
    """
    Abstract base class for WGAN critics (discriminators).

    The critic scores images; higher scores indicate real cover images.
    The caller averages the spatial score map to get a scalar.

    Attributes
    ----------
    version : serialisation version string
    """

    def __init__(self) -> None:
        super().__init__()
        self.version: str = "1"

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Score *image*.

        Parameters
        ----------
        image : (N, 3, H, W)

        Returns
        -------
        (N, 1, H, W) score map  —  caller applies torch.mean()
        """

    def upgrade_legacy(self) -> None:
        """Patch legacy checkpoints that pre-date the version attribute."""
        if not hasattr(self, "version"):
            self.version = "1"
