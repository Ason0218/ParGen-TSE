from __future__ import annotations

"""Bit-level utilities shared across Stage C exporters/metrics.

Bit ordering convention: MSB0
- bit_start=0 points to the MSB (bit7) of data[0]
- bit_start increases left-to-right within a byte (bit7 -> bit0), then continues to the next byte
"""

import hashlib
import struct


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("ceil_div requires b > 0")
    return (int(a) + int(b) - 1) // int(b)


def extract_bits_msb0(data: bytes, bit_start: int, size_bits: int) -> int:
    """Extract an MSB0 bit-span and return it as an integer.

    The first extracted bit becomes the MSB of the returned integer.

    Raises ValueError if the requested span is out of bounds.
    """

    bit_start_i = int(bit_start)
    size_bits_i = int(size_bits)
    if bit_start_i < 0:
        raise ValueError(f"bit_start must be >= 0 (got {bit_start_i})")
    if size_bits_i <= 0:
        raise ValueError(f"size_bits must be > 0 (got {size_bits_i})")
    total_bits = len(data) * 8
    end = bit_start_i + size_bits_i
    if end > total_bits:
        raise ValueError(f"bit span out of range: [{bit_start_i}, {end}) over total_bits={total_bits}")

    value = 0
    for i in range(size_bits_i):
        bit_index = bit_start_i + i
        byte_index = bit_index // 8
        bit_in_byte_msb0 = bit_index % 8
        bit_in_byte_lsb0 = 7 - bit_in_byte_msb0
        bit = (data[byte_index] >> bit_in_byte_lsb0) & 1
        value = (value << 1) | int(bit)
    return int(value)


def safe_value_int_msb0(data: bytes, bit_start: int, size_bits: int, *, max_exact_bits: int = 64) -> int:
    """Return a JSON-safe fingerprint integer for a bit span.

    For spans up to `max_exact_bits`, return the exact integer value.
    For larger spans, return a deterministic 64-bit hash to avoid huge integers
    (which can break JSON serialization on Python >=3.11 due to digit limits).
    """

    bs = int(bit_start)
    sz = int(size_bits)
    if bs < 0 or sz <= 0:
        raise ValueError(f"invalid bit span: bit_start={bs}, size_bits={sz}")
    end = bs + sz
    total_bits = len(data) * 8
    if end > total_bits:
        raise ValueError(f"bit span out of range: [{bs}, {end}) over total_bits={total_bits}")

    max_exact = int(max_exact_bits)
    if max_exact <= 0:
        max_exact = 0
    if sz <= max_exact:
        return int(extract_bits_msb0(data, bs, sz))

    start_byte = bs // 8
    end_byte = int(ceil_div(end, 8))
    cover = data[start_byte:end_byte]
    h = hashlib.blake2b(digest_size=8)
    # Use only the *bit offset within the first covered byte* so fingerprints are
    # invariant to coordinate frames (whole packet vs sliced field bytes).
    bit_off = int(bs % 8)
    h.update(struct.pack(">ii", int(bit_off), int(sz)))
    h.update(cover)
    return int.from_bytes(h.digest(), byteorder="big", signed=False)
