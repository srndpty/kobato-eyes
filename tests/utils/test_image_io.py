"""Tests for image IO helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import utils.image_io as image_io
from utils.image_io import generate_thumbnail, resize_image, safe_load_image


def _create_sample_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    image = Image.new("RGB", size, color=(255, 0, 0))
    image.save(path, format="PNG")


def test_generate_thumbnail_creates_cached_file(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    cache_dir = tmp_path / "cache"
    _create_sample_image(source)

    thumbnail_path = generate_thumbnail(source, cache_dir, size=(32, 32))
    assert thumbnail_path is not None and thumbnail_path.exists()

    with Image.open(thumbnail_path) as thumb:
        width, height = thumb.size
        assert width <= 32 and height <= 32

    second_path = generate_thumbnail(source, cache_dir, size=(32, 32))
    assert second_path == thumbnail_path


def test_safe_load_image_handles_corrupted_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not-an-image")

    assert safe_load_image(broken) is None
    cache_dir = tmp_path / "cache2"
    assert generate_thumbnail(broken, cache_dir) is None


def test_safe_load_image_can_skip_large_headers_and_convert_rgb(tmp_path: Path) -> None:
    source = tmp_path / "gray.png"
    Image.new("L", (4, 3), color=128).save(source)

    skipped = safe_load_image(source, hard_skip_pixels=1)
    loaded = safe_load_image(source, rgb=True)

    assert skipped is None
    assert loaded is not None
    assert loaded.mode == "RGB"
    assert loaded.size == (4, 3)


def test_resize_image_preserves_aspect_ratio() -> None:
    image = Image.new("RGB", (100, 50), color=(1, 2, 3))

    resized = resize_image(image, (20, 20))

    assert resized.size == (20, 10)
    assert image.size == (100, 50)


def test_generate_thumbnail_returns_none_for_missing_source(tmp_path: Path) -> None:
    assert generate_thumbnail(tmp_path / "missing.png", tmp_path / "cache") is None


def test_get_thumbnail_reports_missing_qt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(image_io, "QPixmap", None)
    monkeypatch.setattr(image_io, "Qt", None)

    with pytest.raises(RuntimeError, match="QPixmap is unavailable"):
        image_io.get_thumbnail(tmp_path / "source.png")
