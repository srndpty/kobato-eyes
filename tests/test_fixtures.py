"""Tests for the shared conftest fixtures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from db.connection import get_conn


def test_test_rgb_image(test_rgb_image: Image.Image) -> None:
    """Verify test_rgb_image fixture creates valid image."""
    assert test_rgb_image.size == (16, 16)
    assert test_rgb_image.mode == "RGB"


def test_test_image_path(test_image_path: Path) -> None:
    """Verify test_image_path fixture saves image to file."""
    assert test_image_path.exists()
    assert test_image_path.suffix == ".png"

    with Image.open(test_image_path) as img:
        assert img.size == (16, 16)


def test_test_db_path(test_db_path: Path) -> None:
    """Verify test_db_path fixture creates initialized database."""
    assert test_db_path.exists()

    # Verify schema is applied (tables exist)
    conn = get_conn(test_db_path)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {row[0] for row in cursor}

        # Check for expected core tables
        assert "files" in tables
        assert "file_tags" in tables
        assert "fts_files" in tables
    finally:
        conn.close()


def test_test_data_dir(test_data_dir: Path) -> None:
    """Verify test_data_dir fixture creates directory structure."""
    assert test_data_dir.exists()
    assert (test_data_dir / "db").exists()
    assert (test_data_dir / "logs").exists()


def test_test_batch_images(test_batch_images: list[Path]) -> None:
    """Verify test_batch_images fixture creates multiple images."""
    assert len(test_batch_images) == 3

    for path in test_batch_images:
        assert path.exists()
        with Image.open(path) as img:
            assert img.size == (16, 16)
