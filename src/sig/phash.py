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


def _to_signed(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def phash(image: Image.Image) -> int:
    """Compute a perceptual hash (pHash) using a DCT over the image."""
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
    return _to_signed(int(value & 0xFFFFFFFFFFFFFFFF))


def dhash(image: Image.Image) -> int:
    """Compute a difference hash (dHash) comparing adjacent pixels."""
    pixels = _to_grayscale(image, (9, 8))
    diff = pixels[:, 1:] > pixels[:, :-1]
    flat = diff.flatten()
    value = 0
    for bit in flat:
        value = (value << 1) | int(bit)
    return _to_signed(int(value & 0xFFFFFFFFFFFFFFFF))


def hamming64(a: int, b: int) -> int:
    """Compute the Hamming distance between two 64-bit hash values."""
    mask = (1 << 64) - 1
    return int(((int(a) ^ int(b)) & mask).bit_count())


__all__ = ["phash", "dhash", "hamming64"]
