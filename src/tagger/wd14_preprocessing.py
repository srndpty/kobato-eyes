"""Image preprocessing helpers for WD14 ONNX taggers."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def make_square_bgr(img: np.ndarray, target_size: int) -> np.ndarray:
    """Pad a BGR image to a square at least ``target_size`` pixels wide."""

    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def smart_resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize a square image using area for downscale and cubic for upscale."""

    if img.shape[0] > size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if img.shape[0] < size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


def preprocess_pil_to_bgr(image: Image.Image, height: int) -> np.ndarray:
    """Preprocess one PIL image into ``(1, H, W, 3)`` BGR float32."""

    rgba_image = image.convert("RGBA")
    new_image = Image.new("RGBA", rgba_image.size, "WHITE")
    new_image.paste(rgba_image, mask=rgba_image)
    rgb_image = new_image.convert("RGB")
    arr = np.asarray(rgb_image)
    arr = arr[:, :, ::-1]
    arr = make_square_bgr(arr, height)
    arr = smart_resize_square(arr, height)
    arr = arr.astype(np.float32)
    return np.expand_dims(arr, 0)


def preprocess_pil_to_bgr_image(image: Image.Image, height: int) -> np.ndarray:
    """Preprocess one PIL image into ``(H, W, 3)`` BGR float32."""

    image = image.convert("RGBA")
    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)
    image = new_image.convert("RGB")
    rgb = np.asarray(image)
    bgr = rgb[:, :, ::-1]
    bgr = make_square_bgr(bgr, height)
    bgr = smart_resize_square(bgr, height)
    return bgr.astype(np.float32, copy=False)


__all__ = [
    "make_square_bgr",
    "preprocess_pil_to_bgr",
    "preprocess_pil_to_bgr_image",
    "smart_resize_square",
]
