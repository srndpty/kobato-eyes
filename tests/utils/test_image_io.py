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


def test_generate_thumbnail_recovers_corrupt_cache_file(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    cache_dir = tmp_path / "cache"
    _create_sample_image(source)

    thumbnail_path = generate_thumbnail(source, cache_dir, size=(32, 32))
    assert thumbnail_path is not None
    thumbnail_path.write_bytes(b"broken-cache")

    regenerated_path = generate_thumbnail(source, cache_dir, size=(32, 32))

    assert regenerated_path == thumbnail_path
    with Image.open(regenerated_path) as thumb:
        assert thumb.size == (32, 32)


def test_safe_load_image_handles_corrupted_file(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not-an-image")
    original_cap = Image.MAX_IMAGE_PIXELS

    assert safe_load_image(broken) is None
    assert Image.MAX_IMAGE_PIXELS == original_cap
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


def test_safe_load_image_composites_rgba_over_white(tmp_path: Path) -> None:
    source = tmp_path / "alpha.png"
    image = Image.new("RGBA", (1, 1), (255, 0, 0, 0))
    image.save(source)

    loaded = safe_load_image(source, rgb=True)

    assert loaded is not None
    assert loaded.mode == "RGB"
    assert loaded.getpixel((0, 0)) == (255, 255, 255)


def test_safe_load_image_applies_exif_transpose(tmp_path: Path) -> None:
    source = tmp_path / "oriented.jpg"
    image = Image.new("RGB", (2, 4), (1, 2, 3))
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)

    loaded = safe_load_image(source, rgb=True)

    assert loaded is not None
    assert loaded.size == (4, 2)


def test_safe_load_image_restores_pixel_cap_after_large_skip(tmp_path: Path) -> None:
    source = tmp_path / "large-header.png"
    _create_sample_image(source, size=(8, 8))
    original_cap = Image.MAX_IMAGE_PIXELS

    skipped = safe_load_image(source, bomb_pixel_cap=123, hard_skip_pixels=1)

    assert skipped is None
    assert Image.MAX_IMAGE_PIXELS == original_cap


def test_safe_load_image_closes_source_after_rgb_conversion(monkeypatch) -> None:
    closed: list[str] = []
    converted = object()

    class FakeImage:
        size = (4, 4)
        mode = "L"

        def draft(self, mode, size) -> None:
            assert mode == "RGB"
            assert size == (4096, 4096)

        def load(self) -> None:
            return None

        def convert(self, mode):
            assert mode == "RGB"
            return converted

        def close(self) -> None:
            closed.append("source")

    monkeypatch.setattr(image_io.Image, "open", lambda path: FakeImage())

    loaded = safe_load_image("fake.png")

    assert loaded is converted
    assert closed == ["source"]


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


def test_get_thumbnail_uses_memory_cache_and_returns_copy(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    thumb = tmp_path / "thumb.webp"
    source.write_bytes(b"source")
    thumb.write_bytes(b"thumb")
    generated: list[Path] = []

    class FakePixmap:
        def __init__(self, *args: object) -> None:
            self.args = args
            self.copied = bool(args and isinstance(args[0], FakePixmap))

        def fill(self, color: object) -> None:
            return None

        def isNull(self) -> bool:  # noqa: N802 - Qt-compatible name
            return False

        def scaled(self, width: int, height: int, aspect: object, transform: object) -> "FakePixmap":
            return FakePixmap("scaled", width, height, aspect, transform)

    class FakeQt:
        class GlobalColor:
            transparent = object()

        class AspectRatioMode:
            KeepAspectRatio = object()

        class TransformationMode:
            SmoothTransformation = object()

    def fake_generate_thumbnail(source_path: Path, cache_dir: Path, *, size: tuple[int, int], format: str) -> Path:
        generated.append(Path(source_path))
        return thumb

    monkeypatch.setattr(image_io, "QPixmap", FakePixmap)
    monkeypatch.setattr(image_io, "Qt", FakeQt)
    monkeypatch.setattr(image_io, "generate_thumbnail", fake_generate_thumbnail)
    image_io._THUMB_CACHE.clear()

    first = image_io.get_thumbnail(source, 32, 32)
    second = image_io.get_thumbnail(source, 32, 32)

    assert generated == [source]
    assert isinstance(first, FakePixmap)
    assert isinstance(second, FakePixmap)
    assert second.copied is True


def test_get_thumbnail_evicts_old_memory_cache_entries(monkeypatch, tmp_path: Path) -> None:
    first_source = tmp_path / "first.png"
    second_source = tmp_path / "second.png"
    thumb = tmp_path / "thumb.webp"
    first_source.write_bytes(b"first")
    second_source.write_bytes(b"second")
    thumb.write_bytes(b"thumb")

    class FakePixmap:
        def __init__(self, *args: object) -> None:
            self.args = args

        def fill(self, color: object) -> None:
            return None

        def isNull(self) -> bool:  # noqa: N802 - Qt-compatible name
            return False

        def scaled(self, width: int, height: int, aspect: object, transform: object) -> "FakePixmap":
            return FakePixmap("scaled", width, height, aspect, transform)

    class FakeQt:
        class GlobalColor:
            transparent = object()

        class AspectRatioMode:
            KeepAspectRatio = object()

        class TransformationMode:
            SmoothTransformation = object()

    monkeypatch.setattr(image_io, "QPixmap", FakePixmap)
    monkeypatch.setattr(image_io, "Qt", FakeQt)
    monkeypatch.setattr(image_io, "generate_thumbnail", lambda *args, **kwargs: thumb)
    monkeypatch.setattr(image_io, "_THUMB_CACHE_LIMIT", 1)
    image_io._THUMB_CACHE.clear()

    image_io.get_thumbnail(first_source, 32, 32)
    image_io.get_thumbnail(second_source, 32, 32)

    keys = list(image_io._THUMB_CACHE)
    assert len(keys) == 1
    assert str(second_source.resolve()) in keys[0]
