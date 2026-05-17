"""Tests for indexing QRunnable signal boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 core required", exc_type=ImportError)

from core.pipeline import IndexPhase, IndexProgress
from ui.index_tasks import IndexRunnable


class _ViewModel:
    def run_index_once(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {"scanned": 1, "tagged": 1}


def test_index_runnable_emits_error_when_runner_fails(tmp_path: Path) -> None:
    errors: list[str] = []
    finished: list[dict[str, object]] = []

    def fail_runner(
        progress_cb: Callable[[IndexProgress], None],
        is_cancelled: Callable[[], bool],
    ) -> dict[str, object]:
        assert callable(progress_cb)
        assert is_cancelled() is False
        raise RuntimeError("manual refresh failed")

    task = IndexRunnable(_ViewModel(), tmp_path / "library.db", runner=fail_runner)  # type: ignore[arg-type]
    task.signals.error.connect(errors.append)
    task.signals.finished.connect(finished.append)

    task.run()

    assert errors == ["manual refresh failed"]
    assert finished == []


def test_index_runnable_emits_error_when_pre_run_fails(tmp_path: Path) -> None:
    errors: list[str] = []

    def fail_pre_run() -> dict[str, object]:
        raise RuntimeError("retag prepare failed")

    task = IndexRunnable(_ViewModel(), tmp_path / "library.db", pre_run=fail_pre_run)  # type: ignore[arg-type]
    task.signals.error.connect(errors.append)

    task.run()

    assert errors == ["retag prepare failed"]


def test_index_runnable_merges_pre_run_stats_and_finished(tmp_path: Path) -> None:
    finished: list[dict[str, object]] = []

    def pre_run() -> dict[str, object]:
        return {"retagged_marked": 3}

    def runner(
        progress_cb: Callable[[IndexProgress], None],
        is_cancelled: Callable[[], bool],
    ) -> dict[str, object]:
        assert is_cancelled() is False
        return {"scanned": 5, "tagged": 2}

    task = IndexRunnable(_ViewModel(), tmp_path / "library.db", pre_run=pre_run, runner=runner)  # type: ignore[arg-type]
    task.signals.finished.connect(finished.append)

    task.run()

    assert finished == [{"scanned": 5, "tagged": 2, "retagged_marked": 3}]


def test_index_runnable_progress_uses_message_for_unknown_total(tmp_path: Path) -> None:
    progress: list[tuple[int, int, str]] = []

    def runner(
        progress_cb: Callable[[IndexProgress], None],
        is_cancelled: Callable[[], bool],
    ) -> dict[str, object]:
        progress_cb(IndexProgress(phase=IndexPhase.SCAN, done=0, total=-1, message="Scanning roots"))
        progress_cb(IndexProgress(phase=IndexPhase.TAG, done=1, total=2, message="ignored"))
        return {}

    task = IndexRunnable(_ViewModel(), tmp_path / "library.db", runner=runner)  # type: ignore[arg-type]
    task.signals.progress.connect(lambda done, total, label: progress.append((done, total, label)))

    task.run()

    assert progress == [(0, -1, "Scanning roots"), (1, 2, "Tag")]


def test_index_runnable_cancel_callback_is_shared_with_runner(tmp_path: Path) -> None:
    finished: list[dict[str, object]] = []

    def runner(
        progress_cb: Callable[[IndexProgress], None],
        is_cancelled: Callable[[], bool],
    ) -> dict[str, object]:
        return {"cancelled": is_cancelled()}

    task = IndexRunnable(_ViewModel(), tmp_path / "library.db", runner=runner)  # type: ignore[arg-type]
    task.signals.finished.connect(finished.append)
    task.cancel()

    task.run()

    assert finished == [{"cancelled": True}]
