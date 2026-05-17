"""Tests for duplicate refinement stage."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance

import dup.refine as refine_module
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


def test_refine_pair_returns_none_for_unreadable_image(tmp_path: Path) -> None:
    path_a = tmp_path / "a.png"
    broken = tmp_path / "broken.png"
    _make_image(path_a, (200, 10, 10))
    broken.write_bytes(b"not an image")

    assert refine_pair(1, 2, path_a, broken) is None


def test_refine_pair_keeps_ssim_result_when_orb_fails(monkeypatch, tmp_path: Path) -> None:
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    _make_image(path_a, (200, 10, 10))
    _make_variant(path_b, path_a)

    def fail_orb(img_a, img_b):  # noqa: ANN001
        raise RuntimeError("orb unavailable")

    monkeypatch.setattr(refine_module, "_compute_orb_ratio", fail_orb)

    result = refine_pair(1, 2, path_a, path_b)

    assert isinstance(result, RefinedMatch)
    assert result.is_duplicate
    assert result.ssim is not None
    assert result.orb_ratio is None
    assert result.reason == "ssim>=0.9"


def test_refine_pair_reports_metric_unavailable_when_all_metrics_fail(monkeypatch, tmp_path: Path) -> None:
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.png"
    _make_image(path_a, (200, 10, 10))
    _make_variant(path_b, path_a)

    monkeypatch.setattr(refine_module, "_compute_ssim", lambda img_a, img_b: (_ for _ in ()).throw(ValueError("ssim")))
    monkeypatch.setattr(
        refine_module, "_compute_orb_ratio", lambda img_a, img_b: (_ for _ in ()).throw(ValueError("orb"))
    )

    result = refine_pair(1, 2, path_a, path_b)

    assert isinstance(result, RefinedMatch)
    assert not result.is_duplicate
    assert result.ssim is None
    assert result.orb_ratio is None
    assert result.reason == "ssim unavailable, orb unavailable"
