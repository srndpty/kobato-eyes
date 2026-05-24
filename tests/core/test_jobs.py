"""Tests for the asynchronous job manager."""

from __future__ import annotations

import time
from typing import Iterable

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 core required", exc_type=ImportError)
from PyQt6.QtCore import QCoreApplication

from core.jobs import BatchJob, JobManager, JobPriority

pytestmark = pytest.mark.gui


@pytest.fixture(scope="module")
def qapp() -> Iterable[QCoreApplication]:
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


class DummyJob(BatchJob):
    def __init__(self, items: list[int]) -> None:
        super().__init__(items)
        self.outputs: list[int] = []

    def process_item(self, item: int, loaded: int) -> int:
        return loaded * 2

    def write_item(self, item: int, processed: int) -> None:
        self.outputs.append(processed)

    def finalize(self) -> list[int]:
        return list(self.outputs)


class FailingJob(BatchJob):
    def __init__(self) -> None:
        super().__init__([1])
        self.cleaned = False

    def process_item(self, item: int, loaded: int) -> int:
        raise RuntimeError("job failed")

    def cleanup(self) -> None:
        self.cleaned = True


class CancelledJob(BatchJob):
    def __init__(self) -> None:
        super().__init__([1])
        self.processed = False
        self.cleaned = False

    def prepare(self) -> None:
        self.cancel()

    def process_item(self, item: int, loaded: int) -> int:
        self.processed = True
        return loaded

    def cleanup(self) -> None:
        self.cleaned = True


def _wait_for(condition, app: QCoreApplication, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while not condition():
        if time.time() > deadline:
            raise TimeoutError("Timed out waiting for job completion")
        app.processEvents()
        time.sleep(0.01)


def test_job_manager_emits_signals(qapp: QCoreApplication) -> None:
    manager = JobManager(max_workers=1)
    job = DummyJob([1, 2, 3])
    signals = manager.submit(job, priority=JobPriority.BACKGROUND)

    progress_events: list[tuple[int, int]] = []
    completions: list[list[int]] = []
    errors: list[tuple[Exception, str]] = []

    signals.progress.connect(lambda done, total: progress_events.append((done, total)))
    signals.completed.connect(lambda result: completions.append(result))
    signals.error.connect(lambda exc, tb: errors.append((exc, tb)))

    _wait_for(lambda: bool(completions) or bool(errors), qapp)

    assert not errors
    assert completions == [[2, 4, 6]]
    assert progress_events == [(1, 3), (2, 3), (3, 3)]
    manager.shutdown()


class NamedJob(BatchJob):
    def __init__(self, name: str, record: list[str]) -> None:
        super().__init__([name])
        self._name = name
        self._record = record

    def load_item(self, item: str) -> str:
        self._record.append(f"load:{self._name}")
        time.sleep(0.02)
        return item

    def finalize(self) -> str:
        self._record.append(f"done:{self._name}")
        return self._name


def test_job_manager_priority(qapp: QCoreApplication) -> None:
    manager = JobManager(max_workers=1)
    record: list[str] = []
    bg_signals = manager.submit(NamedJob("background", record), priority=JobPriority.BACKGROUND)
    fg_signals = manager.submit(NamedJob("foreground", record), priority=JobPriority.FOREGROUND)

    completions: list[str] = []
    bg_signals.completed.connect(lambda result: completions.append(result))
    fg_signals.completed.connect(lambda result: completions.append(result))

    _wait_for(lambda: len(completions) == 2, qapp)

    assert record[0].startswith("load:foreground")
    assert completions[0] == "foreground"
    manager.shutdown()


def test_job_manager_emits_error_and_finished_on_job_failure(qapp: QCoreApplication) -> None:
    manager = JobManager(max_workers=1)
    job = FailingJob()
    signals = manager.submit(job, priority=JobPriority.FOREGROUND)

    completions: list[object] = []
    errors: list[tuple[Exception, str]] = []
    finished: list[None] = []

    signals.completed.connect(completions.append)
    signals.error.connect(lambda exc, tb: errors.append((exc, tb)))
    signals.finished.connect(lambda: finished.append(None))

    _wait_for(lambda: bool(finished), qapp)

    assert completions == []
    assert len(errors) == 1
    assert isinstance(errors[0][0], RuntimeError)
    assert str(errors[0][0]) == "job failed"
    assert "RuntimeError: job failed" in errors[0][1]
    assert finished == [None]
    assert job.cleaned is True
    assert not manager.has_pending_jobs()
    manager.shutdown()


def test_job_manager_emits_cancelled_and_finished_on_cooperative_cancel(qapp: QCoreApplication) -> None:
    manager = JobManager(max_workers=1)
    job = CancelledJob()
    signals = manager.submit(job, priority=JobPriority.FOREGROUND)

    completions: list[object] = []
    cancellations: list[None] = []
    errors: list[tuple[Exception, str]] = []
    finished: list[None] = []

    signals.completed.connect(completions.append)
    signals.cancelled.connect(lambda: cancellations.append(None))
    signals.error.connect(lambda exc, tb: errors.append((exc, tb)))
    signals.finished.connect(lambda: finished.append(None))

    _wait_for(lambda: bool(finished), qapp)

    assert completions == []
    assert cancellations == [None]
    assert errors == []
    assert finished == [None]
    assert job.processed is False
    assert job.cleaned is True
    assert not manager.has_pending_jobs()
    manager.shutdown()


def test_job_manager_shutdown_waits_for_queued_jobs(qapp: QCoreApplication) -> None:
    manager = JobManager(max_workers=1)
    completions: list[list[int]] = []

    first = manager.submit(DummyJob([1]), priority=JobPriority.FOREGROUND)
    second = manager.submit(DummyJob([2]), priority=JobPriority.BACKGROUND)
    first.completed.connect(lambda result: completions.append(result))
    second.completed.connect(lambda result: completions.append(result))

    assert manager.shutdown(timeout_ms=2000)
    _wait_for(lambda: not manager.has_pending_jobs(), qapp)

    assert completions == [[2], [4]]
    assert not manager.has_pending_jobs()
