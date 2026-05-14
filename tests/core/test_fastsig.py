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
