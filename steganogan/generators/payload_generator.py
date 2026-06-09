import torch
from typing import List
from ..utils import text_to_bits


class PayloadGenerator:
    """Handles payload generation for encoding."""

    def __init__(self, device: torch.device) -> None:
        self.device: torch.device = device

    def random_data(self, cover: torch.Tensor, data_depth: int) -> torch.Tensor:
        """
        Generate a random binary payload matching the spatial dimensions of `cover`.

        Parameters
        ----------
        cover      : reference image tensor  (N, C, H, W)
        data_depth : number of bit-planes (D)

        Returns
        -------
        Binary tensor of shape (N, D, H, W) with values in {0, 1}.
        """
        N, _, H, W = cover.size()
        return torch.zeros((N, data_depth, H, W), device=self.device).random_(0, 2)

    def make_payload(self, width: int, height: int,
                     depth: int, text: str) -> torch.Tensor:
        """
        Create a payload tensor by encoding `text` as bits and tiling to fill
        the spatial grid.

        Parameters
        ----------
        width, height : spatial dimensions of the target image
        depth         : bits per pixel (D)
        text          : secret text to embed

        Returns
        -------
        Float tensor of shape (1, D, H, W).
        """
        message: List[int] = text_to_bits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message
        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width).to(self.device)
