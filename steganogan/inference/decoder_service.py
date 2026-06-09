# -*- coding: utf-8 -*-
"""Decoding service: extracts the hidden message from a stego image file."""

import os
from collections import Counter

import torch
import torch.nn as nn
from skimage.io import imread

from ..utils.crypto import bytearray_to_bits, bits_to_bytearray, bytearray_to_text


class DecoderService:
    """
    File-level decoding service.

    Loads a stego image from disk, runs the decoder, and reconstructs the
    hidden text message using majority voting over null-byte delimited segments.

    Parameters
    ----------
    decoder : trained decoder module
    device  : compute device
    verbose : print completion message
    """

    def __init__(
        self,
        decoder: nn.Module,
        device:  torch.device,
        verbose: bool = False,
    ) -> None:
        self.decoder = decoder
        self.device  = device
        self.verbose = verbose

    def decode(self, image_path: str) -> str:
        """
        Recover the hidden message from *image_path*.

        Parameters
        ----------
        image_path : path to the stego image

        Returns
        -------
        The decoded text message.

        Raises
        ------
        FileNotFoundError : if *image_path* does not exist
        ValueError        : if no valid message is found
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Stego image not found: {os.path.basename(image_path)!r}")

        image = imread(image_path, pilmode="RGB") / 255.0
        image = (
            torch.FloatTensor(image)
            .permute(2, 1, 0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            bits = (self.decoder(image).view(-1) > 0).int().cpu().numpy().tolist()

        candidates: Counter = Counter()
        for segment in bits_to_bytearray(bits).split(b"\x00\x00\x00\x00"):
            text = bytearray_to_text(bytearray(segment))
            if text:
                candidates[text] += 1

        if not candidates:
            raise ValueError(f"No valid message found in {os.path.basename(image_path)!r}.")

        message, count = candidates.most_common(1)[0]

        if self.verbose:
            print(f"Message decoded ({count} occurrences found).")

        return message

    # Backward-compat alias
    def decode_image(self, image_path: str) -> str:
        return self.decode(image_path)
