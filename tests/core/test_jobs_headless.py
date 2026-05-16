"""Headless tests for :mod:`core.jobs`."""

from __future__ import annotations

import os
import sys
import threading
import time

os.environ["KOE_HEADLESS"] = "1"
sys.modules.pop("core.jobs", None)
from core.jobs import BatchJob, JobManager, JobPriority


class _RecordingJob(BatchJob):
    def __init__(self, start_event: threading.Event) -> None:
        super().__init__([1, 2])
        self._start_event = start_event
        self.outputs: list[int] = []
        self.cleaned = False

    def prepare(self) -> None:
        assert self._start_event.wait(2.0)

    def process_item(self, item: int, loaded: int) -> int:
        return loaded * 10

    def write_item(self, item: int, processed: int) -> None:
        self.outputs.append(processed)

    def finalize(self) -> list[int]:
        return list(self.outputs)

    def cleanup(self) -> None:
        self.cleaned = True


class _FailingJob(BatchJob):
    def __init__(self, start_event: threading.Event) -> None:
        super().__init__([1])
        self._start_event = start_event
        self.cleaned = False

    def prepare(self) -> None:
        assert self._start_event.wait(2.0)

    def process_item(self, item: int, loaded: int) -> int:
        raise RuntimeError("headless failure")

    def cleanup(self) -> None:
        self.cleaned = True


class _CancelAfterLoadJob(BatchJob):
    def __init__(self, start_event: threading.Event) -> None:
        super().__init__([1])
        self._start_event = start_event
        self.wrote = False
        self.cleaned = False

    def prepare(self) -> None:
        assert self._start_event.wait(2.0)

    def load_item(self, item: int) -> int:
        self.cancel()
        return item

    def write_item(self, item: int, processed: int) -> None:
        self.wrote = True

    def cleanup(self) -> None:
        self.cleaned = True


def _wait_until(condition, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise TimeoutError("condition was not reached")


def test_headless_job_manager_completes_job() -> None:
    manager = JobManager(max_workers=1)
    start_event = threading.Event()
    job = _RecordingJob(start_event)
    signals = manager.submit(job, priority=JobPriority.FOREGROUND)

    progress: list[tuple[int, int]] = []
    completions: list[list[int]] = []
    finished: list[None] = []

    signals.progress.connect(lambda done, total: progress.append((done, total)))
    signals.completed.connect(completions.append)
    signals.finished.connect(lambda: finished.append(None))

    start_event.set()
    assert manager.wait_for_done(2000)
    _wait_until(lambda: bool(finished))

    assert progress == [(1, 2), (2, 2)]
    assert completions == [[10, 20]]
    assert finished == [None]
    assert job.cleaned is True
    assert not manager.has_pending_jobs()


def test_headless_job_manager_emits_error_and_finishes() -> None:
    manager = JobManager(max_workers=1)
    start_event = threading.Event()
    job = _FailingJob(start_event)
    signals = manager.submit(job, priority=JobPriority.FOREGROUND)

    errors: list[tuple[Exception, str]] = []
    finished: list[None] = []

    signals.error.connect(lambda exc, tb: errors.append((exc, tb)))
    signals.finished.connect(lambda: finished.append(None))

    start_event.set()
    assert manager.wait_for_done(2000)
    _wait_until(lambda: bool(finished))

    assert len(errors) == 1
    assert str(errors[0][0]) == "headless failure"
    assert "RuntimeError: headless failure" in errors[0][1]
    assert finished == [None]
    assert job.cleaned is True


def test_headless_job_manager_emits_cancelled_after_load_cancel() -> None:
    manager = JobManager(max_workers=1)
    start_event = threading.Event()
    job = _CancelAfterLoadJob(start_event)
    signals = manager.submit(job, priority=JobPriority.FOREGROUND)

    cancellations: list[None] = []
    finished: list[None] = []

    signals.cancelled.connect(lambda: cancellations.append(None))
    signals.finished.connect(lambda: finished.append(None))

    start_event.set()
    assert manager.wait_for_done(2000)
    _wait_until(lambda: bool(finished))

    assert cancellations == [None]
    assert finished == [None]
    assert job.wrote is False
    assert job.cleaned is True
