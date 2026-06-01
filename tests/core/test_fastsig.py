"""Tests for fast signature helper functions."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image

import core.fastsig as fastsig


def test_to_signed64_wraps_unsigned_values() -> None:
    assert fastsig._to_signed64(0) == 0
    assert fastsig._to_signed64((1 << 63) - 1) == (1 << 63) - 1
    assert fastsig._to_signed64(1 << 63) == -(1 << 63)
    assert fastsig._to_signed64((1 << 64) - 1) == -1
    assert fastsig._to_signed64((1 << 64) + 7) == 7


def test_bulk_upsert_signatures_inserts_and_updates_signed_values() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE signatures (file_id INTEGER PRIMARY KEY, phash_u64 INTEGER, dhash_u64 INTEGER)")

    inserted = fastsig.bulk_upsert_signatures(conn, [(1, 10, 20), (2, 1 << 64, (1 << 64) - 1)])
    updated = fastsig.bulk_upsert_signatures(conn, [(1, 30, 40)])

    assert inserted == 2
    assert updated == 1
    rows = conn.execute("SELECT file_id, phash_u64, dhash_u64 FROM signatures ORDER BY file_id").fetchall()
    assert rows == [(1, 30, 40), (2, 0, -1)]
    assert fastsig.bulk_upsert_signatures(conn, []) == 0


def test_compute_worker_skips_missing_and_returns_hashes(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(image_path)
    monkeypatch.setattr(fastsig, "phash", lambda image: (1 << 64) - 2)
    monkeypatch.setattr(fastsig, "dhash", lambda image: 123)

    assert fastsig._compute_worker((5, str(image_path))) == (5, -2, 123)
    assert fastsig._compute_worker((6, str(tmp_path / "missing.png"))) is None


def test_compute_signatures_mp_uses_executor_and_reports_progress(monkeypatch) -> None:
    progress: list[tuple[int, int]] = []

    class FakeExecutor:
        def __init__(self, *, max_workers: int, mp_context) -> None:
            self.max_workers = max_workers
            self.mp_context = mp_context

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def map(
            self,
            func: Callable[[tuple[int, str]], tuple[int, int, int] | None],
            tasks: Iterable[tuple[int, str]],
            chunksize: int,
        ):
            assert chunksize == 3
            for task in tasks:
                yield func(task)

    monkeypatch.setattr(fastsig, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(fastsig, "_compute_worker", lambda task: None if task[0] == 2 else (task[0], 10, 20))

    result = fastsig.compute_signatures_mp(
        [(1, "a"), (2, "b"), (3, "c")],
        max_workers=1,
        chunksize=3,
        progress=lambda done, total: progress.append((done, total)),
    )

    assert result == [(1, 10, 20), (3, 10, 20)]
    assert progress == [(3, 3)]


def test_fast_fill_missing_signatures_can_skip_database(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(fastsig, "compute_signatures_mp", lambda *args, **kwargs: [(1, 2, 3)])

    result = fastsig.fast_fill_missing_signatures(str(tmp_path / "app.db"), [(1, "a")], apply_to_db=False)

    assert result == [(1, 2, 3)]


def test_compute_signatures_mp_respects_cancel_fn(monkeypatch) -> None:
    """cancel_fn が True を返した時点で早期リターンし、部分結果だけが返る。"""

    shutdown_called = []

    class CancellableFakeExecutor:
        def __init__(self, *, max_workers: int, mp_context) -> None:
            pass

        def __enter__(self) -> "CancellableFakeExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def shutdown(self, wait: bool, cancel_futures: bool) -> None:
            shutdown_called.append((wait, cancel_futures))

        def map(
            self,
            func: Callable[[tuple[int, str]], tuple[int, int, int] | None],
            tasks: Iterable[tuple[int, str]],
            chunksize: int,
        ):
            for task in tasks:
                yield func(task)

    monkeypatch.setattr(fastsig, "ProcessPoolExecutor", CancellableFakeExecutor)
    monkeypatch.setattr(fastsig, "_compute_worker", lambda task: (task[0], 10, 20))

    # 1 件目の結果を受け取った後（2 回目の cancel_fn 呼び出し時）にキャンセルが発動する
    # ※ cancel check は各結果の yield 後に行われるため「処理前キャンセル」ではなく
    #   「結果受取後・次の結果追加前」のキャンセルになる
    call_count = 0

    def cancel_after_first() -> bool:
        nonlocal call_count
        call_count += 1
        return call_count >= 2  # 2回目以降は True（キャンセル）

    result = fastsig.compute_signatures_mp(
        [(1, "a"), (2, "b"), (3, "c")],
        max_workers=1,
        chunksize=1,
        cancel_fn=cancel_after_first,
    )

    # 1 件目の結果のみが results に積まれ、2 件目の結果を受け取った時点で早期リターンする
    assert result == [(1, 10, 20)]
    assert shutdown_called == [(False, True)]


def test_fast_fill_missing_signatures_passes_cancel_fn(monkeypatch, tmp_path: Path) -> None:
    """fast_fill_missing_signatures が cancel_fn を compute_signatures_mp に転送する。"""
    received_cancel_fn = []

    def fake_compute(items, *, max_workers, chunksize, progress, cancel_fn=None):
        received_cancel_fn.append(cancel_fn)
        return []

    monkeypatch.setattr(fastsig, "compute_signatures_mp", fake_compute)

    sentinel = lambda: True  # noqa: E731
    fastsig.fast_fill_missing_signatures(
        str(tmp_path / "app.db"),
        [(1, "a")],
        apply_to_db=False,
        cancel_fn=sentinel,
    )

    assert received_cancel_fn == [sentinel]
