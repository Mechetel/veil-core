"""Steganography inference: encode a message into a cover, decode it back.

Uses the high-level ``SteganoGAN.encode()/decode()`` API (never raw encoder/decoder
tensor calls). Loaded models are cached per key.
"""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache

import torch

from app.ml.checkpoint import load_steganogan
from app.ml.device import get_device
from app.registry.steg import resolve


@lru_cache(maxsize=12)
def load_steg(key: str):
    entry = resolve(key)
    if not entry.available:
        raise FileNotFoundError(f"Weight missing for {key!r}: {entry.path}")
    return load_steganogan(str(entry.path), get_device())


def encode(key: str, image_bytes: bytes, message: str) -> bytes:
    """Embed *message* into the cover image bytes; return stego PNG bytes."""
    model = load_steg(key)
    with tempfile.TemporaryDirectory() as d:
        cover = os.path.join(d, "cover.png")
        stego = os.path.join(d, "stego.png")
        with open(cover, "wb") as f:
            f.write(image_bytes)
        with torch.no_grad():
            model.encode(cover, stego, message)
        with open(stego, "rb") as f:
            return f.read()


def decode(key: str, image_bytes: bytes) -> str:
    """Recover the hidden message from stego image bytes."""
    model = load_steg(key)
    with tempfile.TemporaryDirectory() as d:
        stego = os.path.join(d, "stego.png")
        with open(stego, "wb") as f:
            f.write(image_bytes)
        with torch.no_grad():
            return model.decode(stego)
