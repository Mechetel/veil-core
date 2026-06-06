# -*- coding: utf-8 -*-
"""Backward-compat shim — use steganogan.engine or import SteganoGAN directly."""
from .engine import SteganoGAN
__all__ = ["SteganoGAN"]
