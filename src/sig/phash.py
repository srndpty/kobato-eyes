"""Perceptual hashing utilities used for duplicate detection."""

from __future__ import annotations

try:
    import numpy as np  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency may be absent in tests
    np = None  # type: ignore[assignment]

try:
    from PIL import Image  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency may be absent in tests
    Image = None  # type: ignore[assignment]

try:
    import cv2  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency may be absent in tests
    cv2 = None  # type: ignore[assignment]


def _to_grayscale(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    if Image is None or np is None:  # pragma: no cover - dependency guard
        raise RuntimeError("NumPy and Pillow are required to compute perceptual hashes")
    resample = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]
    grayscale = image.convert("L").resize(size, resample)
    return np.asarray(grayscale, dtype=np.float32)


def to_signed64(value: int) -> int:
    """Wrap ``value`` to its low 64 bits and return the signed representation.

    SQLite INTEGER values are signed 64-bit numbers. Hash implementations often
    produce unsigned values, so this conversion intentionally uses modulo 2**64
    semantics. It preserves the hash bit pattern and avoids ``OverflowError``
    when persisting values whose most-significant bit is set.
    """

    normalized = int(value) & ((1 << 64) - 1)
    return normalized - (1 << 64) if normalized >= (1 << 63) else normalized


def phash(image: Image.Image) -> int:
    """Compute a pHash and return its signed 64-bit representation."""
    if cv2 is None:  # pragma: no cover - exercised when OpenCV is unavailable
        raise RuntimeError("OpenCV (cv2) is required to compute perceptual hashes")
    pixels = _to_grayscale(image, (32, 32))
    dct = cv2.dct(pixels)
    block = dct[:8, :8]
    flat = block.flatten()
    mean = flat[1:].mean() if flat.size > 1 else flat.mean()
    bits = flat > mean
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return to_signed64(value)


def dhash(image: Image.Image) -> int:
    """Compute a dHash and return its signed 64-bit representation."""
    pixels = _to_grayscale(image, (9, 8))
    diff = pixels[:, 1:] > pixels[:, :-1]
    flat = diff.flatten()
    value = 0
    for bit in flat:
        value = (value << 1) | int(bit)
    return to_signed64(value)


def hamming64(a: int, b: int) -> int:
    """Compute the Hamming distance between two 64-bit hash values."""
    mask = (1 << 64) - 1
    return int(((int(a) ^ int(b)) & mask).bit_count())


__all__ = ["phash", "dhash", "hamming64", "to_signed64"]
