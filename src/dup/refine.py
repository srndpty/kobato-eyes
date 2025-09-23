"""Image comparison utilities for duplicate refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

from utils.image_io import safe_load_image


@dataclass(frozen=True)
class RefinementThresholds:
    """Threshold values controlling duplicate decisions."""

    ssim: float = 0.9
    orb: float = 0.15


@dataclass(frozen=True)
class RefinedMatch:
    """Result of comparing two images."""

    file_id_a: int
    file_id_b: int
    ssim: float | None
    orb_ratio: float | None
    is_duplicate: bool
    reason: str


def _prepare_image(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    fitted = ImageOps.fit(image.convert("RGB"), size, Image.Resampling.BICUBIC)
    return np.asarray(fitted, dtype=np.float32) / 255.0


def _compute_ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    gray_size = (min(img_a.width, img_b.width), min(img_a.height, img_b.height))
    if gray_size[0] == 0 or gray_size[1] == 0:
        gray_size = (max(img_a.width, img_b.width), max(img_a.height, img_b.height))
    a_gray = ImageOps.fit(img_a.convert("L"), gray_size, Image.Resampling.BICUBIC)
    b_gray = ImageOps.fit(img_b.convert("L"), gray_size, Image.Resampling.BICUBIC)
    a_arr = np.asarray(a_gray, dtype=np.float32) / 255.0
    b_arr = np.asarray(b_gray, dtype=np.float32) / 255.0
    return float(structural_similarity(a_arr, b_arr, data_range=1.0))


def _compute_orb_ratio(img_a: Image.Image, img_b: Image.Image) -> float:
    gray_a = np.asarray(img_a.convert("L"))
    gray_b = np.asarray(img_b.convert("L"))
    orb = cv2.ORB_create()
    keypoints_a, descriptors_a = orb.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(gray_b, None)
    if descriptors_a is None or descriptors_b is None or not keypoints_a or not keypoints_b:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_a, descriptors_b)
    if not matches:
        return 0.0
    max_pairs = min(len(keypoints_a), len(keypoints_b))
    return float(len(matches) / max_pairs)


def refine_pair(
    file_id_a: int,
    file_id_b: int,
    path_a: str | Path,
    path_b: str | Path,
    *,
    thresholds: RefinementThresholds | None = None,
) -> RefinedMatch | None:
    """Compare two images using SSIM and ORB metrics."""
    image_a = safe_load_image(path_a)
    image_b = safe_load_image(path_b)
    if image_a is None or image_b is None:
        return None

    cfg = thresholds or RefinementThresholds()
    ssim_value = _compute_ssim(image_a, image_b)
    orb_ratio = _compute_orb_ratio(image_a, image_b)

    reasons: list[str] = []
    is_duplicate = False
    if ssim_value >= cfg.ssim:
        reasons.append(f"ssim>={cfg.ssim}")
        is_duplicate = True
    if orb_ratio >= cfg.orb:
        reasons.append(f"orb>={cfg.orb}")
        is_duplicate = True

    reason = ", ".join(reasons) if reasons else "below thresholds"
    return RefinedMatch(
        file_id_a=file_id_a,
        file_id_b=file_id_b,
        ssim=ssim_value,
        orb_ratio=orb_ratio,
        is_duplicate=is_duplicate,
        reason=reason,
    )


__all__ = ["RefinementThresholds", "RefinedMatch", "refine_pair"]
