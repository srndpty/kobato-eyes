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

    monkeypatch.setattr(loaders.cv2, "imdecode", _failing_imdecode)

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


def test_prefetch_loader_propagates_producer_failures(tmp_path) -> None:
    png_path = tmp_path / "sample.png"
    _create_png(png_path)

    class _FailingTagger:
        def prepare_batch_from_rgb_np(self, arrs: List[np.ndarray]) -> np.ndarray:
            raise RuntimeError("prepare failed")

    loader = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_FailingTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
    )

    try:
        with pytest.raises(RuntimeError, match="producer failed"):
            list(loader)
    finally:
        loader.close()

    assert not loader._th.is_alive()


def test_prefetch_loader_skips_corrupt_images_without_failing_batch(tmp_path: Path) -> None:
    good_path = tmp_path / "good.png"
    corrupt_path = tmp_path / "corrupt.png"
    _create_png(good_path)
    corrupt_path.write_bytes(b"not an image")
    dummy = _DummyTagger()

    loader = loaders.PrefetchLoaderPrepared(
        [str(corrupt_path), str(good_path)],
        tagger=dummy,
        batch_size=2,
        prefetch_batches=1,
        io_workers=1,
    )

    try:
        batches = list(loader)
    finally:
        loader.close()

    assert len(batches) == 1
    paths, np_batch, sizes = batches[0]
    assert paths == [str(good_path)]
    assert np_batch.shape[0] == 1
    assert sizes == [(64, 48)]
    metrics = loader.metrics_snapshot()
    assert metrics.submitted == 2
    assert metrics.loaded == 1
    assert metrics.failed == 1
    assert metrics.batches == 1
    assert metrics.route_counts["failed"] == 1
    assert metrics.prepare_seconds >= 0.0


def test_prefetch_loader_falls_back_when_alpha_blend_runs_out_of_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_path = tmp_path / "alpha.png"
    _create_png(png_path)
    dummy = _DummyTagger()

    def _raise_memory_error(_image: np.ndarray) -> np.ndarray:
        raise MemoryError("simulated alpha blend allocation failure")

    monkeypatch.setattr(loaders, "_alpha_to_white_then_resize_bgr", _raise_memory_error)

    loader = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=dummy,
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
    )

    try:
        batches = list(loader)
    finally:
        loader.close()

    assert len(batches) == 1
    paths, np_batch, sizes = batches[0]
    assert paths == [str(png_path)]
    assert np_batch.shape[0] == 1
    assert sizes == [(64, 48)]
    assert dummy.calls[0][0].shape == (48, 64, 3)


def test_alpha_resize_composites_over_white_before_scaling() -> None:
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :, :3] = 0
    rgba[:, :, 3] = 0
    rgba[16:48, 16:48, :3] = 255
    rgba[16:48, 16:48, 3] = 255

    expected = loaders._resize_to_target_side(loaders._alpha_to_white_bgr(rgba))
    actual = loaders._alpha_to_white_then_resize_bgr(rgba)

    assert np.array_equal(actual, expected)


def test_prefetch_loader_propagates_io_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png_path = tmp_path / "sample.png"
    _create_png(png_path)

    def fail_load_one(self: loaders.PrefetchLoaderPrepared, path: str):  # noqa: ARG001
        raise RuntimeError("io worker exploded")

    monkeypatch.setattr(loaders.PrefetchLoaderPrepared, "_load_one", fail_load_one)
    loader = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
    )

    try:
        with pytest.raises(RuntimeError, match="producer failed"):
            list(loader)
        assert isinstance(loader._producer_error, RuntimeError)
        assert str(loader._producer_error) == "io worker exploded"
    finally:
        loader.close()
