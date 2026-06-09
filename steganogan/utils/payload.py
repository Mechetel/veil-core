# -*- coding: utf-8 -*-
"""Factory for generating binary payloads used in steganographic encoding."""

from typing import List

import torch

from .crypto import text_to_bits


class PayloadFactory:
    """
    Creates binary payload tensors for the encoder.

    Parameters
    ----------
    device : target device for all generated tensors
    """

    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device

    def random(self, cover: torch.Tensor, data_depth: int) -> torch.Tensor:
        """
        Sample a uniformly random binary payload matching *cover* spatial dims.

        Parameters
        ----------
        cover      : reference image  (N, C, H, W)
        data_depth : number of bit-planes D

        Returns
        -------
        Tensor of shape (N, D, H, W) with values in {0, 1}.
        """
        N, _, H, W = cover.size()
        return torch.zeros((N, data_depth, H, W), device=self.device).random_(0, 2)

    def from_text(self, width: int, height: int,
                  depth: int, text: str) -> torch.Tensor:
        """
        Build a payload tensor by bit-encoding *text* and tiling to fill
        the (depth × height × width) grid.

        Returns
        -------
        Float tensor of shape (1, D, H, W).
        """
        bits: List[int] = text_to_bits(text) + [0] * 32
        payload = bits
        while len(payload) < width * height * depth:
            payload = payload + bits
        payload = payload[:width * height * depth]
        return torch.FloatTensor(payload).view(1, depth, height, width).to(self.device)

    # ── Backward-compatible aliases ───────────────────────────────────────────

    def random_data(self, cover: torch.Tensor, data_depth: int) -> torch.Tensor:
        """Alias for :meth:`random` (backward compat)."""
        return self.random(cover, data_depth)

    def make_payload(self, width: int, height: int,
                     depth: int, text: str) -> torch.Tensor:
        """Alias for :meth:`from_text` (backward compat)."""
        return self.from_text(width, height, depth, text)
