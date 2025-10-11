"""Tests for parallel duplicate refinement utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image

from ui.dup_refine_parallel import refine_by_pixels_parallel, refine_by_tilehash_parallel, tile_ahash_bits, tile_hamming


@dataclass
class DummyFile:
    """Minimal file metadata used for refinement tests."""

    file_id: int
    path: Path


@dataclass
class DummyEntry:
    """Cluster entry mirroring the runtime data structure."""

    file: DummyFile
    best_hamming: int | None = None


@dataclass
class DummyCluster:
    """Simple cluster object compatible with the refinement helpers."""

    files: list[DummyEntry]
    keeper_id: int


def _save_split_image(path: Path, light: int = 230, dark: int = 20, size: int = 32) -> None:
    """Create a grayscale image whose top half is bright and bottom half is dark."""

    image = Image.new("L", (size, size), color=dark)
    px = image.load()
    half = size // 2
    for y in range(half):
        for x in range(size):
            px[x, y] = light
    # ★ 回転で位置が変わる縦ストライプ（2px）を追加して非一様にする
    for y in range(size):
        px[0, y] = light
        px[1, y] = light
    image.save(path, format="PNG")


def _save_tile_variant(path: Path, source: Path, color: int = 255) -> None:
    """Create a variant that changes a single tile to trigger bit differences."""

    with Image.open(source) as image:
        variant = image.copy()
    width, height = variant.size
    grid = 4
    tile_size = width // grid  # 8 (32x32 の時)
    x0 = width - tile_size
    y0 = height - tile_size
    px = variant.load()
    # ★ タイル全体ではなく、右下タイルの左上 4x4 だけ明るくする
    for y in range(y0, y0 + tile_size // 2):
        for x in range(x0, x0 + tile_size // 2):
            px[x, y] = color
    variant.save(path, format="PNG")


def _save_rotated(path: Path, source: Path) -> None:
    """Create a rotated copy of the given image."""

    with Image.open(source) as image:
        rotated = image.transpose(Image.Transpose.ROTATE_90)
    rotated.save(path, format="PNG")


def _copy_image(path: Path, source: Path) -> None:
    """Save an exact copy of the source image to a new path."""

    with Image.open(source) as image:
        image.copy().save(path, format="PNG")


def _make_entry(path: Path, file_id: int) -> DummyEntry:
    """Helper to create a dummy cluster entry with the provided path."""

    return DummyEntry(file=DummyFile(file_id=file_id, path=path))


def test_tile_ahash_bits_consistency_and_variation(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    clone_path = tmp_path / "clone.png"
    rotated_path = tmp_path / "rotated.png"
    variant_path = tmp_path / "variant.png"

    _save_split_image(base_path)
    _copy_image(clone_path, base_path)
    _save_rotated(rotated_path, base_path)
    _save_tile_variant(variant_path, base_path)

    base_sig = tile_ahash_bits(base_path)
    clone_sig = tile_ahash_bits(clone_path)
    rotated_sig = tile_ahash_bits(rotated_path)
    variant_sig = tile_ahash_bits(variant_path)

    assert base_sig == clone_sig

    rotated_distance = tile_hamming(base_sig, rotated_sig)
    variant_distance = tile_hamming(base_sig, variant_sig)

    assert rotated_distance > 0
    assert variant_distance > 0
    assert variant_distance < rotated_distance


def test_refine_by_tilehash_parallel_threshold_and_callbacks(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    rotated_path = tmp_path / "rotated.png"
    variant_path = tmp_path / "variant.png"

    _save_split_image(base_path)
    _save_rotated(rotated_path, base_path)
    # max_bits=0 で {1,2} が残ることを確認したいので、variant は完全コピーにする
    _copy_image(variant_path, base_path)
    # 期待形: variant はタイルハッシュ完全一致、rotated は不一致
    assert tile_hamming(tile_ahash_bits(base_path), tile_ahash_bits(variant_path)) == 0
    assert tile_hamming(tile_ahash_bits(base_path), tile_ahash_bits(rotated_path)) > 0

    cluster = DummyCluster(
        files=[
            _make_entry(base_path, 1),
            _make_entry(variant_path, 2),
            _make_entry(rotated_path, 3),
        ],
        keeper_id=1,
    )

    broad = refine_by_tilehash_parallel([cluster], max_bits=64)
    assert len(broad) == 1
    assert {entry.file.file_id for entry in broad[0].files} == {1, 2, 3}

    ticks: list[tuple[int, int, int]] = []

    def tick(done: int, total: int, phase: int) -> None:
        ticks.append((phase, done, total))

    refined = refine_by_tilehash_parallel([cluster], max_bits=0, tick=tick)
    assert len(refined) == 1
    assert {entry.file.file_id for entry in refined[0].files} == {1, 2}

    phases = {phase for phase, _, _ in ticks}
    assert phases == {1, 2}
    assert ticks[-1][1:] == (1, 1)


def test_refine_by_tilehash_parallel_cancelled(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    _save_split_image(base_path)
    cluster = DummyCluster(files=[_make_entry(base_path, 1)], keeper_id=1)

    cancelled = refine_by_tilehash_parallel([cluster], is_cancelled=lambda: True)
    assert cancelled == []


def test_refine_by_pixels_parallel_filters_and_callbacks(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    duplicate_path = tmp_path / "duplicate.png"
    different_path = tmp_path / "different.png"
    missing_path = tmp_path / "missing.png"

    _save_split_image(base_path)
    _copy_image(duplicate_path, base_path)
    Image.new("L", (32, 32), color=0).save(different_path, format="PNG")

    cluster = DummyCluster(
        files=[
            _make_entry(base_path, 1),
            _make_entry(duplicate_path, 2),
            _make_entry(different_path, 3),
            _make_entry(missing_path, 4),
        ],
        keeper_id=1,
    )

    ticks: list[tuple[int, int]] = []

    def tick(done: int, total: int) -> None:
        ticks.append((done, total))

    refined = refine_by_pixels_parallel([cluster], mae_thr=0.001, tick=tick)
    assert len(refined) == 1
    assert {entry.file.file_id for entry in refined[0].files} == {1, 2}
    assert ticks == [(1, 1)]


def test_refine_by_pixels_parallel_cancelled(tmp_path: Path) -> None:
    base_path = tmp_path / "base.png"
    _save_split_image(base_path)
    cluster = DummyCluster(files=[_make_entry(base_path, 1)], keeper_id=1)

    calls: list[None] = []

    def tick(done: int, total: int) -> None:  # pragma: no cover - should not be called
        calls.append(None)

    cancelled = refine_by_pixels_parallel([cluster, cluster], is_cancelled=lambda: True, tick=tick)
    assert cancelled == []
    assert not calls
