# -*- coding: utf-8 -*-
"""Backward-compat shim — use steganogan.models.decoders instead."""
from .decoders import BasicDecoder, DenseDecoder
__all__ = ["BasicDecoder", "DenseDecoder"]
