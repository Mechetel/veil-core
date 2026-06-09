# -*- coding: utf-8 -*-
"""
Message codec: UTF-8 text ↔ bit-list with zlib compression and
Reed–Solomon error correction (250 parity symbols by default).
"""

import zlib
from typing import List, Union

from reedsolo import RSCodec


class MessageCodec:
    """
    Bidirectional codec for secret text messages.

    Pipeline (encode):  text → UTF-8 → zlib → RS encode → bit list
    Pipeline (decode):  bit list → bytes → RS decode → zlib decompress → text

    Parameters
    ----------
    rs_symbols : Reed-Solomon parity symbol count (default 250)
    """

    def __init__(self, rs_symbols: int = 250) -> None:
        self._rs: RSCodec = RSCodec(rs_symbols)

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Encode *text* to a list of bits {0, 1}."""
        return self._bytes_to_bits(self._text_to_bytes(text))

    def decode(self, bits: List[int]) -> Union[str, bool]:
        """Decode a bit-list back to text; returns False on failure."""
        return self._bytes_to_text(self._bits_to_bytes(bits))

    def bytes_to_bits(self, data: Union[bytes, bytearray]) -> List[int]:
        """Convert raw bytes to a flat bit list (MSB first)."""
        return self._bytes_to_bits(data)

    def bits_to_bytes(self, bits: List[int]) -> bytearray:
        """Pack a flat bit-list into a bytearray (8 bits/byte, MSB first)."""
        return self._bits_to_bytes(bits)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _text_to_bytes(self, text: str) -> bytearray:
        assert isinstance(text, str), f"Expected str, got {type(text).__name__}"
        compressed = zlib.compress(text.encode("utf-8"))
        return bytearray(self._rs.encode(bytearray(compressed)))

    def _bytes_to_text(self, data: Union[bytes, bytearray]) -> Union[str, bool]:
        try:
            assert isinstance(data, (bytes, bytearray))
            decoded = self._rs.decode(data)
            payload = decoded[0] if isinstance(decoded, tuple) else decoded
            return zlib.decompress(bytes(payload)).decode("utf-8")
        except Exception:
            return False

    @staticmethod
    def _bytes_to_bits(data: Union[bytes, bytearray]) -> List[int]:
        result: List[int] = []
        for byte in data:
            result.extend(int(b) for b in bin(byte)[2:].zfill(8))
        return result

    @staticmethod
    def _bits_to_bytes(bits: List[int]) -> bytearray:
        return bytearray(
            int("".join(str(b) for b in bits[i * 8:(i + 1) * 8]), 2)
            for i in range(len(bits) // 8)
        )


# ── Module-level singleton + backward-compatible aliases ─────────────────────

_codec = MessageCodec()

def text_to_bits(text: str) -> List[int]:                        return _codec.encode(text)
def bits_to_text(bits: List[int]) -> Union[str, bool]:           return _codec.decode(bits)
def bytearray_to_bits(x: Union[bytes, bytearray]) -> List[int]:  return _codec.bytes_to_bits(x)
def bits_to_bytearray(bits: List[int]) -> bytearray:             return _codec.bits_to_bytes(bits)
def text_to_bytearray(text: str) -> bytearray:                   return _codec._text_to_bytes(text)
def bytearray_to_text(x: Union[bytes, bytearray]) -> Union[str, bool]: return _codec._bytes_to_text(x)
