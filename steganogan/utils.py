# -*- coding: utf-8 -*-
"""Backward-compat shim — use steganogan.utils instead."""
from .utils.crypto       import (MessageCodec,
                                  text_to_bits, bits_to_text,
                                  bytearray_to_bits, bits_to_bytearray,
                                  text_to_bytearray, bytearray_to_text)
from .utils.image_quality import SSIMCalculator, ssim, first_element

MessageEncoder = MessageCodec               # old name alias
__all__ = [
    "MessageEncoder", "MessageCodec",
    "text_to_bits", "bits_to_text",
    "bytearray_to_bits", "bits_to_bytearray",
    "text_to_bytearray", "bytearray_to_text",
    "SSIMCalculator", "ssim", "first_element",
]
