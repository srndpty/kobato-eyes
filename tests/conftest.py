"""Shared pytest fixtures and configuration for all tests."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image

from db.connection import end_quiesce, is_quiesced
from db.schema import apply_schema


@pytest.fixture(autouse=True)
def _assert_no_quiesce_leak() -> Generator[None, None, None]:
    """Detect leaked quiesce state while keeping later tests isolated."""
    while is_quiesced():
        end_quiesce()
    yield
    leaked = is_quiesced()
    while is_quiesced():
        end_quiesce()
    assert not leaked, "quiesce state leaked from test"


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
    conn = sqlite3.connect(db_path)
    try:
        apply_schema(conn)
        conn.commit()
    finally:
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


def _proto_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _proto_field(number: int, payload: bytes) -> bytes:
    return _proto_varint(number << 3 | 2) + _proto_varint(len(payload)) + payload


def build_fake_onnx(tag: str, location: str | None = None) -> bytes:
    """
    最小の ModelProto. 外部重み参照の解析対象（graph.initializer.external_data）だけを持つ.
    tag はグラフ名に入れ、内容の差し替えを表現する.
    """
    tensor = _proto_field(8, b"weight")
    if location is not None:
        entry = _proto_field(1, b"location") + _proto_field(2, location.encode())
        tensor += _proto_field(13, entry) + _proto_varint(14 << 3) + _proto_varint(1)
    graph = _proto_field(2, tag.encode()) + _proto_field(5, tensor)
    return _proto_field(7, graph)


@pytest.fixture
def fake_onnx():
    """Return the ONNX bytes builder used by v1 identity tests."""
    return build_fake_onnx
