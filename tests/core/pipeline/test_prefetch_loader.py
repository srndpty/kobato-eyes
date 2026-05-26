"""Tests for :mod:`core.pipeline.loaders.PrefetchLoaderPrepared`."""

from __future__ import annotations

import os
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
    assert metrics.extension_counts[".png"] == 2
    assert metrics.extension_bytes[".png"] > 0
    assert len(metrics.slow_decode_files) == 2
    assert metrics.slow_decode_files[0]["seconds"] >= metrics.slow_decode_files[-1]["seconds"]
    assert {entry["extension"] for entry in metrics.slow_decode_files} == {".png"}
    assert metrics.prepare_seconds >= 0.0


def test_prefetch_loader_limits_slow_decode_file_metrics(tmp_path: Path) -> None:
    paths = []
    for index in range(25):
        path = tmp_path / f"sample-{index}.jpg"
        _create_jpeg(path)
        paths.append(str(path))

    loader = loaders.PrefetchLoaderPrepared(
        paths,
        tagger=_DummyTagger(),
        batch_size=5,
        prefetch_batches=1,
        io_workers=1,
    )

    try:
        list(loader)
    finally:
        loader.close()

    metrics = loader.metrics_snapshot()
    assert metrics.extension_counts[".jpg"] == 25
    assert len(metrics.slow_decode_files) == 20
    assert all(entry["extension"] == ".jpg" for entry in metrics.slow_decode_files)


def test_prefetch_loader_uses_input_cache_for_png(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_EXTENSIONS", raising=False)
    png_path = tmp_path / "sample.png"
    cache_dir = tmp_path / "input-cache"
    _create_png(png_path)

    first = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
    )
    try:
        first_batches = list(first)
    finally:
        first.close()

    assert len(first_batches) == 1
    first_metrics = first.metrics_snapshot()
    assert first_metrics.input_cache_misses == 1
    assert first_metrics.input_cache_writes == 1
    assert first_metrics.route_counts["opencv"] == 1

    second = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
    )
    try:
        second_batches = list(second)
    finally:
        second.close()

    assert len(second_batches) == 1
    second_metrics = second.metrics_snapshot()
    assert second_metrics.input_cache_hits == 1
    assert second_metrics.route_counts["input_cache"] == 1
    assert second_batches[0][2] == [(64, 48)]


def test_prefetch_loader_input_cache_defaults_to_png_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_EXTENSIONS", raising=False)
    jpg_path = tmp_path / "sample.jpg"
    cache_dir = tmp_path / "input-cache"
    _create_jpeg(jpg_path)

    loader = loaders.PrefetchLoaderPrepared(
        [str(jpg_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
    )
    try:
        batches = list(loader)
    finally:
        loader.close()

    assert len(batches) == 1
    metrics = loader.metrics_snapshot()
    assert metrics.input_cache_misses == 0
    assert metrics.input_cache_writes == 0
    assert metrics.route_counts["opencv"] == 1
    assert not list(cache_dir.rglob("*.npz"))


def test_prefetch_loader_input_cache_extension_enables_jpeg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_EXTENSIONS", raising=False)
    jpg_path = tmp_path / "sample.jpg"
    cache_dir = tmp_path / "input-cache"
    _create_jpeg(jpg_path)

    first = loaders.PrefetchLoaderPrepared(
        [str(jpg_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
        input_cache_extensions={".jpg"},
    )
    try:
        list(first)
    finally:
        first.close()

    first_metrics = first.metrics_snapshot()
    assert first_metrics.input_cache_misses == 1
    assert first_metrics.input_cache_writes == 1

    second = loaders.PrefetchLoaderPrepared(
        [str(jpg_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
        input_cache_extensions={".jpg"},
    )
    try:
        list(second)
    finally:
        second.close()

    second_metrics = second.metrics_snapshot()
    assert second_metrics.input_cache_hits == 1
    assert second_metrics.route_counts["input_cache"] == 1


def test_prefetch_loader_discards_corrupt_input_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_EXTENSIONS", raising=False)
    png_path = tmp_path / "sample.png"
    cache_dir = tmp_path / "input-cache"
    _create_png(png_path)

    first = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
    )
    try:
        list(first)
    finally:
        first.close()

    cache_files = list(cache_dir.rglob("*.npz"))
    assert len(cache_files) == 1
    cache_files[0].write_bytes(b"not an npz")

    def skip_cache_write(
        self: loaders.PrefetchLoaderPrepared, cache_path: Path, rgb: np.ndarray, size: tuple[int, int]
    ) -> None:  # noqa: ARG001
        return None

    monkeypatch.setattr(loaders.PrefetchLoaderPrepared, "_write_input_cache", skip_cache_write)
    second = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
    )
    try:
        list(second)
    finally:
        second.close()

    metrics = second.metrics_snapshot()
    assert metrics.input_cache_errors == 1
    assert metrics.input_cache_hits == 0
    assert metrics.route_counts["opencv"] == 1
    assert not cache_files[0].exists()


def test_prefetch_loader_input_cache_misses_when_mtime_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_EXTENSIONS", raising=False)
    png_path = tmp_path / "sample.png"
    cache_dir = tmp_path / "input-cache"
    _create_png(png_path)

    first = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
    )
    try:
        list(first)
    finally:
        first.close()

    stat_result = png_path.stat()
    changed_ns = int(stat_result.st_mtime_ns) + 2_000_000_000
    os.utime(png_path, ns=(int(stat_result.st_atime_ns), changed_ns))

    second = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
    )
    try:
        list(second)
    finally:
        second.close()

    metrics = second.metrics_snapshot()
    assert metrics.input_cache_hits == 0
    assert metrics.input_cache_misses == 1
    assert metrics.route_counts["opencv"] == 1


def test_prefetch_loader_disables_input_cache_when_cache_dir_cannot_be_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_DIR", raising=False)
    monkeypatch.delenv("KE_TAGGER_INPUT_CACHE_EXTENSIONS", raising=False)
    png_path = tmp_path / "sample.png"
    cache_dir = tmp_path / "not-a-directory"
    _create_png(png_path)
    cache_dir.write_text("file blocks cache directory creation", encoding="utf-8")

    loader = loaders.PrefetchLoaderPrepared(
        [str(png_path)],
        tagger=_DummyTagger(),
        batch_size=1,
        prefetch_batches=1,
        io_workers=1,
        input_cache_dir=cache_dir,
    )
    try:
        batches = list(loader)
    finally:
        loader.close()

    assert len(batches) == 1
    metrics = loader.metrics_snapshot()
    assert metrics.input_cache_misses == 0
    assert metrics.input_cache_writes == 0
    assert metrics.route_counts["opencv"] == 1


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
