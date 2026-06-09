# -*- coding: utf-8 -*-
"""Encoding service: embeds a text message into a cover image file."""

from typing import Union

import torch
import torch.nn as nn
from imageio import imread, imwrite

from ..utils.payload import PayloadFactory


class EncoderService:
    """
    File-level encoding service.

    Loads a cover image from disk, encodes a text message into it using
    the provided encoder, and saves the resulting stego image to disk.

    Parameters
    ----------
    encoder         : trained encoder module
    payload_factory : :class:`PayloadFactory` instance
    data_depth      : bits per pixel (D)
    device          : compute device
    verbose         : print completion message
    """

    def __init__(
        self,
        encoder:         nn.Module,
        payload_factory: PayloadFactory,
        data_depth:      int,
        device:          torch.device,
        verbose:         bool = False,
    ) -> None:
        self.encoder         = encoder
        self.payload_factory = payload_factory
        self.data_depth      = data_depth
        self.device          = device
        self.verbose         = verbose

    def encode(
        self,
        cover_path:  str,
        output_path: str,
        message:     str,
    ) -> None:
        """
        Embed *message* into the image at *cover_path* and write to *output_path*.

        Parameters
        ----------
        cover_path  : path to the RGB cover image
        output_path : path where the stego PNG will be written
        message     : secret text message
        """
        cover = imread(cover_path, pilmode="RGB") / 127.5 - 1.0
        cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0)

        _, _, H, W = cover.shape
        payload = self.payload_factory.from_text(W, H, self.data_depth, message)

        cover   = cover.to(self.device)
        payload = payload.to(self.device)

        with torch.no_grad():
            raw_out = self.encoder(cover, payload)
        stego_out = raw_out[-1] if isinstance(raw_out, (list, tuple)) else raw_out

        stego = stego_out[0].clamp(-1.0, 1.0)
        stego = ((stego.permute(2, 1, 0).cpu().numpy() + 1.0) * 127.5).astype("uint8")
        imwrite(output_path, stego)

        if self.verbose:
            print(f"Stego image saved → {output_path}")

    # Backward-compat alias
    def encode_image(self, cover_path, output_path, message):
        return self.encode(cover_path, output_path, message)
