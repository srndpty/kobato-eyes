"""Tests for perceptual hash utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image

from sig.phash import dhash, hamming64, phash, to_signed64


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        ((1 << 63) - 1, (1 << 63) - 1),
        (1 << 63, -(1 << 63)),
        ((1 << 64) - 1, -1),
        (1 << 64, 0),
        ((1 << 64) + 7, 7),
        (-1, -1),
        (-(1 << 64), 0),
    ],
)
def test_to_signed64_normalizes_low_64_bits(value: int, expected: int) -> None:
    """Boundary, oversized, and negative inputs retain their low 64 bits."""

    assert to_signed64(value) == expected


@pytest.mark.parametrize("hash_function", [phash, dhash])
def test_image_hash_is_signed_64bit_and_deterministic(hash_function: Callable[[Image.Image], int]) -> None:
    """The same image produces the same SQLite-safe hash on repeated calls."""

    pixels = np.arange(64 * 64 * 3, dtype=np.uint8).reshape((64, 64, 3))
    image = Image.fromarray(pixels)

    first = hash_function(image)
    second = hash_function(image)

    assert first == second
    assert -(1 << 63) <= first <= (1 << 63) - 1


def test_dhash_matches_known_bit_order() -> None:
    """A constructed image maps eight known comparison bytes in row order."""

    expected_bytes = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01]
    rows: list[list[int]] = []
    for expected_byte in expected_bytes:
        current = 128
        row = [current]
        for shift in range(7, -1, -1):
            current += 1 if expected_byte & (1 << shift) else -1
            row.append(current)
        rows.append(row)

    image = Image.fromarray(np.asarray(rows, dtype=np.uint8))

    assert dhash(image) == 0x8040201008040201 - (1 << 64)


def test_hamming64_distance_boundaries() -> None:
    """Equal values have zero distance and inverted 64-bit values differ fully."""

    assert hamming64(0, 0) == 0
    assert hamming64(0, (1 << 64) - 1) == 64
    assert hamming64(-1, 0) == 64
