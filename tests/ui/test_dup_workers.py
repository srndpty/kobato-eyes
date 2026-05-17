"""Tests for duplicate UI background workers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 core required", exc_type=ImportError)

from ui.dup_workers import DuplicateScanRequest, DuplicateScanRunnable, RefinePipelineRunnable, ThumbJob
from ui.viewmodels import DupViewModel


class _Connection:
    def close(self) -> None:
        """Close the fake connection."""


def test_duplicate_scan_worker_emits_clusters(tmp_path: Path) -> None:
    emitted: list[object] = []
    progress: list[tuple[int, int]] = []
    rows = [
        {
            "file_id": 1,
            "path": str(tmp_path / "a.png"),
            "size": 10,
            "width": 10,
            "height": 10,
            "phash_u64": 1,
        }
    ]

    view_model = DupViewModel(
        db_path=tmp_path / "library.db",
        connection_factory=lambda path: _Connection(),
        iter_files_for_dup=lambda conn, path_like: rows,
        scanner_factory=lambda config: type("Scanner", (), {"build_clusters": lambda self, files: ["cluster"]})(),
    )
    worker = DuplicateScanRunnable(view_model, view_model.db_path, DuplicateScanRequest(None, 4, None))
    worker.signals.progress.connect(lambda current, total: progress.append((current, total)))
    worker.signals.finished.connect(emitted.append)

    worker.run()

    assert emitted == [["cluster"]]
    assert progress[-1] == (1, 1)


def test_duplicate_scan_worker_emits_error(tmp_path: Path) -> None:
    errors: list[str] = []
    view_model = DupViewModel(
        db_path=tmp_path / "library.db",
        connection_factory=lambda path: (_ for _ in ()).throw(RuntimeError("db closed")),
    )
    worker = DuplicateScanRunnable(view_model, view_model.db_path, DuplicateScanRequest(None, 4, None))
    worker.signals.error.connect(errors.append)

    worker.run()

    assert errors == ["db closed"]


def test_duplicate_scan_worker_skips_bad_rows_and_finishes(tmp_path: Path) -> None:
    emitted: list[object] = []
    errors: list[str] = []
    captured_files: list[object] = []
    rows = [
        {
            "file_id": 1,
            "path": str(tmp_path / "a.png"),
            "size": 10,
            "width": 10,
            "height": 10,
            "phash_u64": 1,
        },
        {
            "file_id": 2,
            "path": str(tmp_path / "broken.png"),
            "size": 10,
            "width": 10,
            "height": 10,
            "phash_u64": "not-a-hash",
        },
        {
            "file_id": 3,
            "path": str(tmp_path / "c.png"),
            "size": 10,
            "width": 10,
            "height": 10,
            "phash_u64": 1,
        },
    ]

    class Scanner:
        def build_clusters(self, files):
            captured_files.extend(files)
            return ["cluster"]

    view_model = DupViewModel(
        db_path=tmp_path / "library.db",
        connection_factory=lambda path: _Connection(),
        iter_files_for_dup=lambda conn, path_like: rows,
        scanner_factory=lambda config: Scanner(),
    )
    worker = DuplicateScanRunnable(view_model, view_model.db_path, DuplicateScanRequest(None, 4, None))
    worker.signals.finished.connect(emitted.append)
    worker.signals.error.connect(errors.append)

    worker.run()

    assert errors == []
    assert emitted == [["cluster"]]
    assert [file.file_id for file in captured_files] == [1, 3]


def test_duplicate_scan_worker_reports_non_int_file_id(tmp_path: Path) -> None:
    errors: list[str] = []
    rows = [
        {
            "file_id": object(),
            "path": str(tmp_path / "a.png"),
            "size": 10,
            "width": 10,
            "height": 10,
            "phash_u64": 1,
        }
    ]
    view_model = DupViewModel(
        db_path=tmp_path / "library.db",
        connection_factory=lambda path: _Connection(),
        iter_files_for_dup=lambda conn, path_like: rows,
    )
    worker = DuplicateScanRunnable(view_model, view_model.db_path, DuplicateScanRequest(None, 4, None))
    worker.signals.error.connect(errors.append)

    worker.run()

    assert errors == ["expected int-compatible value, got object"]


def test_refine_worker_cancel_emits_canceled(monkeypatch) -> None:
    canceled: list[bool] = []

    def fake_refine(*args, **kwargs):
        return ["refined"]

    monkeypatch.setattr("ui.dup_workers.refine_by_tilehash_parallel", fake_refine)
    worker = RefinePipelineRunnable(["cluster"], {}, {})
    worker.signals.canceled.connect(lambda: canceled.append(True))
    worker.cancel()

    worker.run()

    assert canceled == [True]


class _DoneEmitter:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, object]] = []

    def emit(self, path: str, image: object) -> None:
        """Collect duplicate thumbnail payloads."""

        self.emitted.append((path, image))


class _ThumbSignal:
    def __init__(self) -> None:
        self.done = _DoneEmitter()


def test_duplicate_thumb_job_broken_image_emits_none(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.png"
    signal = _ThumbSignal()
    view_model = DupViewModel(
        db_path=tmp_path / "library.db",
        generate_thumbnail=lambda path, cache_dir, size, format: (_ for _ in ()).throw(OSError("broken image")),
    )
    job = ThumbJob(view_model, missing_path, (64, 64), tmp_path / "cache", signal)  # type: ignore[arg-type]

    job.run()

    assert signal.done.emitted == [(str(missing_path), None)]
