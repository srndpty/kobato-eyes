"""Shared pytest fixtures and configuration for all tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image

from db.connection import end_quiesce, get_conn, is_quiesced
from db.schema import apply_schema


@pytest.fixture(autouse=True)
def _reset_quiesce_state() -> Generator[None, None, None]:
    """Reset process-global quiesce counter to avoid leaking between tests."""
    while is_quiesced():
        end_quiesce()
    yield
    while is_quiesced():
        end_quiesce()


@pytest.fixture
def test_rgb_image() -> Image.Image:
    """Create a simple 16x16 RGB test image."""
    return Image.new("RGB", (16, 16), color=(200, 20, 20))


@pytest.fixture
def test_image_path(tmp_path: Path, test_rgb_image: Image.Image) -> Path:
    """Save test RGB image to a temporary file and return its path."""
    image_path = tmp_path / "test_image.png"
    test_rgb_image.save(image_path, format="PNG")
    return image_path


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Create a temporary database path with schema initialized."""
    db_path = tmp_path / "test.db"
    conn = get_conn(db_path)
    apply_schema(conn)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def test_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data (AppData-like structure)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Create subdirectories
        (data_dir / "db").mkdir(parents=True, exist_ok=True)
        (data_dir / "logs").mkdir(parents=True, exist_ok=True)
        yield data_dir


@pytest.fixture
def test_batch_images(tmp_path: Path) -> list[Path]:
    """Create multiple test images (useful for batch processing tests)."""
    images = []
    for i in range(3):
        img = Image.new("RGB", (16, 16), color=(i * 80, i * 80, i * 80))
        path = tmp_path / f"test_image_{i}.png"
        img.save(path, format="PNG")
        images.append(path)
    return images
