"""Tests for :mod:`core.pipeline.loaders.PrefetchLoaderPrepared`."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

pytest.importorskip("PIL")
from PIL import Image

from core.pipeline import loaders


np = pytest.importorskip("numpy")
pytest.importorskip("cv2")


class _DummyTagger:
    """Simple tagger stub that records RGB arrays and stacks them."""

    def __init__(self) -> None:
        self.calls: List[List[np.ndarray]] = []

    def prepare_batch_from_rgb_np(self, arrs: List[np.ndarray]) -> np.ndarray:
        self.calls.append([np.asarray(a) for a in arrs])
        stacked = np.stack([np.asarray(a, dtype=np.float32) for a in arrs], axis=0)
        return stacked


def _create_png(path: Path) -> None:
    image = Image.new("RGBA", (64, 48), (32, 64, 96, 255))
    image.save(path, format="PNG")


def _create_jpeg(path: Path) -> None:
    image = Image.new("RGB", (64, 48), (160, 32, 96))
    image.save(path, format="JPEG", quality=95)


def test_prefetch_loader_pil_fallback_and_cleanup(tmp_path, monkeypatch) -> None:
    png_path = tmp_path / "sample.png"
    jpeg_path = tmp_path / "sample.jpg"
    _create_png(png_path)
    _create_jpeg(jpeg_path)

    dummy = _DummyTagger()

    def _failing_imdecode(data: np.ndarray, flags: int):  # type: ignore[override]
        return None

    class _FailingTJ:
        def decode_header(self, buf: bytes) -> tuple[int, int, int, int]:
            return (64, 48, 0, 0)

        def decode(self, buf: bytes, *, pixel_format=None, scaling_factor=None):  # type: ignore[override]
            raise RuntimeError("decode failed")

    monkeypatch.setattr(loaders.cv2, "imdecode", _failing_imdecode)
    monkeypatch.setattr(loaders, "_TJ", _FailingTJ())

    loader = loaders.PrefetchLoaderPrepared(
        [str(png_path), str(jpeg_path)],
        tagger=dummy,
        batch_size=2,
        prefetch_batches=1,
        io_workers=1,
    )

    try:
        batches = list(loader)
        assert len(batches) == 1
        paths, np_batch, sizes = batches[0]

        assert paths == [str(png_path), str(jpeg_path)]
        assert np_batch.shape[0] == 2
        assert np_batch.dtype == np.float32

        assert sizes == [(64, 48), (64, 48)]
        assert len(dummy.calls) == 1
        assert len(dummy.calls[0]) == 2
        for arr in dummy.calls[0]:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (48, 64, 3)
            assert arr.dtype == np.uint8

        assert loader.qsize() == 0
    finally:
        loader.close()

    assert loader.qsize() == 0
    assert not loader._th.is_alive()
    assert loader._stop.is_set()
