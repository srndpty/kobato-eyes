"""Tests for duplicate refinement stage."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance

from dup.refine import RefinedMatch, RefinementThresholds, refine_pair


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (64, 64), color=color)
    image.save(path, format="PNG")


def _make_variant(path: Path, base_path: Path, brightness: float = 1.02) -> None:
    image = Image.open(base_path).convert("RGB")
    variant = ImageEnhance.Brightness(image).enhance(brightness)
    variant.save(path, format="PNG")


def test_refine_pair_identifies_duplicate(tmp_path: Path) -> None:
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    _make_image(path_a, (200, 10, 10))
    _make_variant(path_b, path_a)

    result = refine_pair(1, 2, path_a, path_b)
    assert isinstance(result, RefinedMatch)
    assert result.is_duplicate
    assert result.ssim is not None and result.ssim > 0.95


def test_refine_pair_non_duplicate(tmp_path: Path) -> None:
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    _make_image(path_a, (0, 255, 0))
    _make_image(path_b, (0, 0, 255))

    thresholds = RefinementThresholds(ssim=0.95, orb=0.5)
    result = refine_pair(1, 3, path_a, path_b, thresholds=thresholds)
    assert isinstance(result, RefinedMatch)
    assert not result.is_duplicate
    assert result.reason == "below thresholds"
