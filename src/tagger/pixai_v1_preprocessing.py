"""Float RGB preprocessing matching the upstream tensor resize and padding."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

INPUT_SIZE = 1008
# 原寸 RGB(uint8) を保持するため CPU メモリ上限を設ける. 100MP ≒ 300MB/枚.
MAX_SOURCE_PIXELS = 100_000_000


class SourceImageTooLargeError(ValueError):
    """Raised before decoding when the source would exceed the v1 memory budget."""


def load_source_rgb(path: str | Path, *, max_pixels: int = MAX_SOURCE_PIXELS) -> tuple[np.ndarray, tuple[int, int]]:
    """
    v1 推論入力の唯一の読み込み経路. バッチ/単一画像の両経路で同一テンソルを保証するため、
    EXIF 回転や事前縮小は行わない（公式 processor も Image.open の生画素を使う）.
    """
    with Image.open(path) as image:
        width, height = image.size
        if width * height > max_pixels:
            raise SourceImageTooLargeError(
                f"PixAI v1 source image too large: {width}x{height} exceeds {max_pixels} pixels"
            )
        return composite_rgb(image), (int(width), int(height))


def composite_rgb(image: Image.Image) -> np.ndarray:
    """Composite transparency over white before returning uint8 RGB pixels."""
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, "white")
        canvas.alpha_composite(rgba)
        image = canvas.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _resize_axis(values: np.ndarray, length: int, axis: int) -> np.ndarray:
    """
    Half-pixel bilinear + antialias. uint8 入力をそのまま受け、gather した行だけを float32 化する
    ことで原寸 float32 配列（12B/px）を作らない.
    """
    source = values.shape[axis]
    if source == length:
        return values
    scale = source / length
    support = max(scale, 1.0)
    centers = (np.arange(length, dtype=np.float64) + 0.5) * scale - 0.5
    starts = np.floor(centers - support).astype(np.int64) + 1
    indices = starts[:, None] + np.arange(int(np.ceil(support * 2)) + 1)
    weights = np.maximum(0.0, 1.0 - np.abs(indices - centers[:, None]) / support)
    weights *= (indices >= 0) & (indices < source)
    weights /= weights.sum(axis=1, keepdims=True)
    indices = indices.clip(0, source - 1)
    moved = np.moveaxis(values, axis, 0)
    result = np.empty((length, *moved.shape[1:]), dtype=np.float32)
    for index in range(length):
        result[index] = np.tensordot(weights[index].astype(np.float32), moved[indices[index]], axes=(0, 0))
    return np.moveaxis(result, 0, axis)


def prepare_rgb_batch(images: Sequence[np.ndarray]) -> np.ndarray:
    """Resize float RGB without stretching, pad black, normalize, return NCHW."""
    batch = np.full((len(images), 3, INPUT_SIZE, INPUT_SIZE), -1.0, dtype=np.float32)
    for index, image in enumerate(images):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError("PixAI v1 requires nonempty uint8 RGB images")
        height, width = image.shape[:2]
        ratio = INPUT_SIZE / max(height, width)
        new_h, new_w = max(1, int(height * ratio)), max(1, int(width * ratio))
        # リサイズは線形なので /255 を後段に移しても結果は float32 丸め誤差の範囲で一致する.
        if new_h < height:
            resized = _resize_axis(_resize_axis(image, new_h, 0), new_w, 1)
        else:
            resized = _resize_axis(_resize_axis(image, new_w, 1), new_h, 0)
        values = resized.astype(np.float32, copy=False) / np.float32(255.0)
        top, left = (INPUT_SIZE - new_h) // 2, (INPUT_SIZE - new_w) // 2
        batch[index, :, top : top + new_h, left : left + new_w] = values.transpose(2, 0, 1) * 2.0 - 1.0
    return batch
