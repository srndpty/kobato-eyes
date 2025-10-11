"""Parallel refinement helpers for duplicate detection in the UI."""

from __future__ import annotations

import logging
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger("ui.dup_refine")


def tile_ahash_bits(path: Path, grid: int = 4, tile: int = 8) -> int:
    """
    - 画像を (grid*tile)×(grid*tile) のグレイスケールに縮小
    - 各タイル (tile×tile) の平均で二値化
    - ビット列（gy, gx, ty, tx の順に走る）を little-endian で int にパック
    """
    side = grid * tile
    with Image.open(path) as im:
        # 向き補正が要らなければ次行は省略可
        im = ImageOps.exif_transpose(im)
        im = im.convert("L").resize((side, side), Image.Resampling.BILINEAR)

    arr = np.asarray(im, dtype=np.uint8)  # (side, side)
    # 形状を (gy, gx, ty, tx) に並べ替える（元の for gy->gx->tile.flatten と同じ順）
    a = arr.reshape(grid, tile, grid, tile).transpose(0, 2, 1, 3)  # (gy, gx, ty, tx)

    # タイル平均（各 gy,gx ごとに ty,tx を平均）
    means = a.mean(axis=(2, 3), keepdims=True)

    # しきい値二値化 → 1次元にフラット（順序は gy, gx, ty, tx）
    bits_bool = (a > means).reshape(-1).astype(np.uint8)

    # ビット列をパックして int へ（little-endian でそろえる）
    packed = np.packbits(bits_bool, bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")


def tile_hamming(a_bits: int, b_bits: int) -> int:
    # Python の int は任意精度なのでそのまま XOR→popcount でOK
    return (a_bits ^ b_bits).bit_count()


def _norm_path(p: Path) -> Path:
    try:
        return Path(p).resolve(strict=False)
    except Exception:
        # resolve できない時も大小文字や区切りを正規化
        return Path(os.path.normcase(os.path.abspath(str(p))))


def _format_failure_summary(
    counts: Counter[str],
    samples: dict[str, Path | None],
) -> str:
    parts: list[str] = []
    for err, count in counts.items():
        sample = samples.get(err)
        if sample is None:
            parts.append(f"{count}×{err}")
        else:
            parts.append(f"{count}×{err} (例: {sample})")
    return "; ".join(parts)


def refine_by_tilehash_parallel(
    clusters,
    grid=4,
    tile=8,
    max_bits=32,
    io_workers=None,
    tick=None,  # tick(done, total, phase:int)
    is_cancelled=None,  # is_cancelled() -> bool
):
    if is_cancelled and is_cancelled():
        return []

    # --- phase 1: 署名生成（I/O並列） ---
    all_paths = [_norm_path(e.file.path) for cl in clusters for e in cl.files]
    uniq_paths = sorted(set(all_paths), key=lambda p: (p.anchor, str(p.parent)))
    total1 = len(uniq_paths)

    if io_workers is None:
        io_workers = int(os.environ.get("KE_TILEHASH_THREADS", "0")) or min(8, (os.cpu_count() or 4) * 2)

    log.info("TileHash phase1: %d files, threads=%d", total1, io_workers)
    cache: dict[Path, int] = {}
    done = 0
    failure_counts: Counter[str] = Counter()
    failure_samples: dict[str, Path | None] = {}

    def _work(p: Path):
        return p, tile_ahash_bits(p, grid=grid, tile=tile)

    with ThreadPoolExecutor(max_workers=io_workers) as ex:
        futs = {ex.submit(_work, p): p for p in uniq_paths}
        for f in as_completed(futs):
            if is_cancelled and is_cancelled():
                for ff in futs:
                    ff.cancel()
                return []
            try:
                p, sig = f.result()
                cache[p] = sig
            except Exception as exc:  # pragma: no cover - exercised via summary logging
                path = futs.get(f)
                key = f"{type(exc).__name__}: {exc}"
                failure_counts[key] += 1
                if key not in failure_samples:
                    failure_samples[key] = path
            done += 1
            if tick and (done % 64 == 0 or done == total1):
                tick(done, total1, phase=1)

    if failure_counts:
        total_failures = sum(failure_counts.values())
        summary = _format_failure_summary(failure_counts, failure_samples)
        log.warning(
            "TileHash phase1 skipped %d file(s) due to errors: %s",
            total_failures,
            summary,
        )

    # --- phase 2: クラスタ絞り込み ---
    out = []
    total2 = len(clusters)
    get = cache.get
    hamming = tile_hamming  # ローカル参照で少し速く
    for i, cl in enumerate(clusters, 1):
        if is_cancelled and is_cancelled():
            return []
        keep = next((e for e in cl.files if e.file.file_id == cl.keeper_id), None)
        if not keep:
            continue
        base = get(_norm_path(keep.file.path))
        if base is None:
            continue

        oks = []
        for e in cl.files:
            sig = get(_norm_path(e.file.path))
            if sig is None:
                continue
            if hamming(base, sig) <= max_bits:  # ← ここを tile_hamming に
                oks.append(e)

        if len(oks) >= 2:
            out.append(type(cl)(files=oks, keeper_id=cl.keeper_id))

        if tick and (i % 16 == 0 or i == total2):
            tick(i, total2, phase=2)

    return out


def _load_small_gray(path: Path, size=128):
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _mae01(a: np.ndarray, b: np.ndarray) -> float:
    # 0..1 正規化平均絶対誤差
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))) / 255.0)


def refine_by_pixels_parallel(
    clusters,
    mae_thr=0.006,
    thumb_size=128,
    workers=None,
    tick=None,  # tick(done, total)（単一フェーズ）
    is_cancelled=None,
):
    total = len(clusters)
    if workers is None:
        workers = min(8, (os.cpu_count() or 4))  # CPU寄りなのでスレッド数は控えめでもOK

    keeper_failure_counts: Counter[str] = Counter()
    keeper_failure_samples: dict[str, Path | None] = {}
    entry_failure_counts: Counter[str] = Counter()
    entry_failure_samples: dict[str, Path | None] = {}
    keeper_lock = Lock()
    entry_lock = Lock()
    future_failure_counts: Counter[str] = Counter()
    future_failure_samples: dict[str, Path | None] = {}

    def _process_cluster(cl):
        if is_cancelled and is_cancelled():
            return None
        keep = next((e for e in cl.files if e.file.file_id == cl.keeper_id), None)
        if not keep:
            return None
        try:
            base = _load_small_gray(keep.file.path, size=thumb_size)
        except Exception as exc:
            key = f"{type(exc).__name__}: {exc}"
            with keeper_lock:
                keeper_failure_counts[key] += 1
                keeper_failure_samples.setdefault(key, keep.file.path)
            return None

        oks = []
        for e in cl.files:
            try:
                img = _load_small_gray(e.file.path, size=thumb_size)
                if _mae01(img, base) <= mae_thr:
                    oks.append(e)
            except Exception as exc:
                key = f"{type(exc).__name__}: {exc}"
                with entry_lock:
                    entry_failure_counts[key] += 1
                    entry_failure_samples.setdefault(key, e.file.path)
        if len(oks) >= 2:
            return type(cl)(files=oks, keeper_id=cl.keeper_id)
        return None

    out = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_cluster, cl): cl for cl in clusters}
        for f in as_completed(futs):
            if is_cancelled and is_cancelled():
                for ff in futs:
                    ff.cancel()
                return []
            try:
                r = f.result()
                if r is not None:
                    out.append(r)
            except Exception as exc:  # pragma: no cover - exercised via summary logging
                key = f"{type(exc).__name__}: {exc}"
                future_failure_counts[key] += 1
                future_failure_samples.setdefault(key, None)
            done += 1
            if tick and (done % 16 == 0 or done == total):
                tick(done, total)

    if future_failure_counts:
        total_failures = sum(future_failure_counts.values())
        summary = _format_failure_summary(future_failure_counts, future_failure_samples)
        log.warning(
            "Pixel MAE workers raised %d exception(s): %s",
            total_failures,
            summary,
        )

    if keeper_failure_counts:
        total_failures = sum(keeper_failure_counts.values())
        summary = _format_failure_summary(keeper_failure_counts, keeper_failure_samples)
        log.warning(
            "Pixel MAE skipped %d cluster(s) due to keeper load errors: %s",
            total_failures,
            summary,
        )

    if entry_failure_counts:
        total_failures = sum(entry_failure_counts.values())
        summary = _format_failure_summary(entry_failure_counts, entry_failure_samples)
        log.warning(
            "Pixel MAE excluded %d file(s) due to image load errors: %s",
            total_failures,
            summary,
        )

    return out
