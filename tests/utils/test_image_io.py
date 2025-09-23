"""Tests for image IO helper functions."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from utils.image_io import generate_thumbnail, safe_load_image


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
