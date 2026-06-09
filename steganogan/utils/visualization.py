# -*- coding: utf-8 -*-
"""
Visual quality monitoring: encodes a fixed message into callback images
and saves a 4×2 sample grid PNG after each checkpoint epoch.
"""

import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from imageio import imread
from PIL import Image

from .payload import PayloadFactory


class SampleGridVisualizer:
    """
    Encodes a fixed text message into reference images and saves a grid PNG.

    Useful for qualitative monitoring of stego image quality across epochs
    without running a full evaluation loop.

    Parameters
    ----------
    encoder         : trained encoder module
    payload_factory : :class:`PayloadFactory` instance
    device          : compute device
    callback_dir    : directory with reference PNG images
                      (default ``data/callback_images``)
    n_images        : number of images per grid (default 8)
    """

    GRID_COLS   = 4
    GRID_ROWS   = 2
    GRID_GAP    = 20
    TARGET_SIZE = 360

    def __init__(
        self,
        encoder:         nn.Module,
        payload_factory: PayloadFactory,
        device:          torch.device,
        callback_dir:    str = os.path.join("data", "callback_images"),
        n_images:        int = 8,
    ) -> None:
        self.encoder         = encoder
        self.payload_factory = payload_factory
        self.device          = device
        self.callback_dir    = callback_dir
        self.n_images        = n_images

    # ── Public API ────────────────────────────────────────────────────────────

    def save_grid(
        self,
        output_dir: str,
        epoch:      int,
        message:    str,
        data_depth: int,
    ) -> None:
        """
        Encode *message* into up to :attr:`n_images` callback images and
        write ``grid_epoch_{epoch}.png`` to *output_dir*.
        """
        self._validate_callback_dir()
        filenames = sorted(os.listdir(self.callback_dir))[: self.n_images]
        tensors   = [self._encode_file(f, message, data_depth) for f in filenames]
        self._write_grid(tensors, output_dir, epoch)

    # Backward-compat alias
    def generate_samples(self, samples_path: str, epoch: int,
                         text_to_encode: str, data_depth: int) -> None:
        self.save_grid(samples_path, epoch, text_to_encode, data_depth)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _validate_callback_dir(self) -> None:
        if not os.path.isdir(self.callback_dir):
            os.makedirs(self.callback_dir, exist_ok=True)
            raise FileNotFoundError(
                f"Callback image directory not found: {self.callback_dir}. "
                "Please populate it with reference PNG images."
            )
        n = len(os.listdir(self.callback_dir))
        if n < self.n_images:
            raise ValueError(
                f"Expected at least {self.n_images} images in {self.callback_dir}, "
                f"found {n}."
            )

    def _encode_file(
        self, filename: str, message: str, data_depth: int
    ) -> torch.Tensor:
        path   = os.path.join(self.callback_dir, filename)
        image  = imread(path, pilmode="RGB") / 127.5 - 1.0
        tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        # Resize to TARGET_SIZE *before* encoding so the encoder always runs
        # at a fixed, predictable resolution regardless of source image size.
        # This also makes inference safe on GPUs with limited VRAM.
        cover = F.interpolate(
            tensor,
            size=(self.TARGET_SIZE, self.TARGET_SIZE),
            mode="bilinear", align_corners=False,
        ).to(self.device)
        _, _, H, W = cover.shape
        payload = self.payload_factory.from_text(W, H, data_depth, message)

        with torch.no_grad():
            raw = self.encoder(cover, payload)
        encoded = raw[-1] if isinstance(raw, (list, tuple)) else raw
        return encoded.squeeze(0).clamp(-1.0, 1.0)

    def _write_grid(
        self,
        tensors:    List[torch.Tensor],
        output_dir: str,
        epoch:      int,
    ) -> None:
        batch  = torch.stack(tensors).clamp(-1.0, 1.0)
        batch  = ((batch + 1.0) / 2.0 * 255.0).byte()
        images = [Image.fromarray(t.permute(1, 2, 0).cpu().numpy()) for t in batch]

        w, h   = images[0].size
        total_w = self.GRID_COLS * w + (self.GRID_COLS - 1) * self.GRID_GAP
        total_h = self.GRID_ROWS * h + (self.GRID_ROWS - 1) * self.GRID_GAP
        canvas  = Image.new("RGB", (total_w, total_h), (255, 255, 255))

        for idx, img in enumerate(images):
            row, col = divmod(idx, self.GRID_COLS)
            canvas.paste(img, (col * (w + self.GRID_GAP), row * (h + self.GRID_GAP)))

        canvas.save(os.path.join(output_dir, f"grid_epoch_{epoch}.png"))
