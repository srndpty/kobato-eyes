"""Headless tests for :mod:`core.jobs`."""

from __future__ import annotations

import importlib
import sys
import threading
import time

import pytest


@pytest.fixture()
def headless_jobs(monkeypatch: pytest.MonkeyPatch):
    """Import ``core.jobs`` in headless mode without leaking it to later tests."""

    previous_module = sys.modules.get("core.jobs")
    monkeypatch.setenv("KOE_HEADLESS", "1")
    sys.modules.pop("core.jobs", None)
    module = importlib.import_module("core.jobs")
    try:
        yield module
    finally:
        sys.modules.pop("core.jobs", None)
        if previous_module is not None:
            sys.modules["core.jobs"] = previous_module


def _wait_until(condition, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise TimeoutError("condition was not reached")


def test_headless_job_manager_completes_job(headless_jobs) -> None:
    class _RecordingJob(headless_jobs.BatchJob):
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

    manager = headless_jobs.JobManager(max_workers=1)
    start_event = threading.Event()
    job = _RecordingJob(start_event)
    signals = manager.submit(job, priority=headless_jobs.JobPriority.FOREGROUND)

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


def test_headless_job_manager_emits_error_and_finishes(headless_jobs) -> None:
    class _FailingJob(headless_jobs.BatchJob):
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

    manager = headless_jobs.JobManager(max_workers=1)
    start_event = threading.Event()
    job = _FailingJob(start_event)
    signals = manager.submit(job, priority=headless_jobs.JobPriority.FOREGROUND)

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


def test_headless_job_manager_emits_cancelled_after_load_cancel(headless_jobs) -> None:
    class _CancelAfterLoadJob(headless_jobs.BatchJob):
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

    manager = headless_jobs.JobManager(max_workers=1)
    start_event = threading.Event()
    job = _CancelAfterLoadJob(start_event)
    signals = manager.submit(job, priority=headless_jobs.JobPriority.FOREGROUND)

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


def test_headless_job_manager_completes_empty_job(headless_jobs) -> None:
    class _EmptyJob(headless_jobs.BatchJob):
        def __init__(self, start_event: threading.Event) -> None:
            super().__init__([])
            self._start_event = start_event
            self.cleaned = False

        def prepare(self) -> None:
            assert self._start_event.wait(2.0)

        def finalize(self) -> str:
            return "empty-complete"

        def cleanup(self) -> None:
            self.cleaned = True

    manager = headless_jobs.JobManager(max_workers=1)
    start_event = threading.Event()
    job = _EmptyJob(start_event)
    signals = manager.submit(job)
    completions: list[str] = []
    progress: list[tuple[int, int]] = []
    finished: list[None] = []

    signals.completed.connect(completions.append)
    signals.progress.connect(lambda done, total: progress.append((done, total)))
    signals.finished.connect(lambda: finished.append(None))

    start_event.set()
    assert manager.shutdown(2000)
    _wait_until(lambda: bool(finished))

    assert completions == ["empty-complete"]
    assert progress == []
    assert job.cleaned is True


def test_headless_job_manager_emits_cancelled_after_prepare_cancel(headless_jobs) -> None:
    class _CancelInPrepareJob(headless_jobs.BatchJob):
        def __init__(self, start_event: threading.Event) -> None:
            super().__init__([1])
            self._start_event = start_event
            self.processed = False

        def prepare(self) -> None:
            assert self._start_event.wait(2.0)
            self.cancel()

        def process_item(self, item: int, loaded: int) -> int:
            self.processed = True
            return item

    manager = headless_jobs.JobManager(max_workers=1)
    start_event = threading.Event()
    job = _CancelInPrepareJob(start_event)
    signals = manager.submit(job)
    cancellations: list[None] = []
    finished: list[None] = []

    signals.cancelled.connect(lambda: cancellations.append(None))
    signals.finished.connect(lambda: finished.append(None))

    start_event.set()
    assert manager.wait_for_done(2000)
    _wait_until(lambda: bool(finished))

    assert cancellations == [None]
    assert job.processed is False


def test_headless_job_manager_emits_cancelled_after_process_cancel(headless_jobs) -> None:
    class _CancelAfterProcessJob(headless_jobs.BatchJob):
        def __init__(self, start_event: threading.Event) -> None:
            super().__init__([1])
            self._start_event = start_event
            self.wrote = False

        def prepare(self) -> None:
            assert self._start_event.wait(2.0)

        def process_item(self, item: int, loaded: int) -> int:
            self.cancel()
            return loaded

        def write_item(self, item: int, processed: int) -> None:
            self.wrote = True

    manager = headless_jobs.JobManager(max_workers=1)
    start_event = threading.Event()
    job = _CancelAfterProcessJob(start_event)
    signals = manager.submit(job)
    cancellations: list[None] = []
    finished: list[None] = []

    signals.cancelled.connect(lambda: cancellations.append(None))
    signals.finished.connect(lambda: finished.append(None))

    start_event.set()
    assert manager.wait_for_done(2000)
    _wait_until(lambda: bool(finished))

    assert cancellations == [None]
    assert job.wrote is False


def test_headless_job_manager_drains_pending_jobs_by_priority(headless_jobs) -> None:
    class _OrderedJob(headless_jobs.BatchJob):
        def __init__(self, name: str, order: list[str], gate: threading.Event | None = None) -> None:
            super().__init__([name])
            self._name = name
            self._order = order
            self._gate = gate

        def prepare(self) -> None:
            if self._gate is not None:
                assert self._gate.wait(2.0)

        def process_item(self, item: str, loaded: str) -> str:
            self._order.append(self._name)
            return loaded

    manager = headless_jobs.JobManager(max_workers=1)
    order: list[str] = []
    gate = threading.Event()

    first = manager.submit(_OrderedJob("first", order, gate), priority=headless_jobs.JobPriority.FOREGROUND)
    background = manager.submit(_OrderedJob("background", order), priority=headless_jobs.JobPriority.BACKGROUND)
    foreground = manager.submit(_OrderedJob("foreground", order), priority=headless_jobs.JobPriority.FOREGROUND)
    finished: list[str] = []

    first.finished.connect(lambda: finished.append("first"))
    background.finished.connect(lambda: finished.append("background"))
    foreground.finished.connect(lambda: finished.append("foreground"))

    _wait_until(lambda: manager.has_pending_jobs())
    gate.set()
    assert manager.wait_for_done(2000)
    _wait_until(lambda: len(finished) == 3)

    assert order == ["first", "foreground", "background"]
    assert finished == ["first", "foreground", "background"]
    assert not manager.has_pending_jobs()


def test_headless_job_manager_starts_next_job_after_finished_callbacks_return(headless_jobs) -> None:
    class _OrderedJob(headless_jobs.BatchJob):
        def __init__(self, name: str, events: list[str], gate: threading.Event | None = None) -> None:
            super().__init__([name])
            self._name = name
            self._events = events
            self._gate = gate

        def prepare(self) -> None:
            if self._gate is not None:
                assert self._gate.wait(2.0)

        def process_item(self, item: str, loaded: str) -> str:
            self._events.append(f"{self._name}:process")
            return loaded

    manager = headless_jobs.JobManager(max_workers=1)
    events: list[str] = []
    gate = threading.Event()

    first = manager.submit(_OrderedJob("first", events, gate), priority=headless_jobs.JobPriority.FOREGROUND)
    second = manager.submit(_OrderedJob("second", events), priority=headless_jobs.JobPriority.BACKGROUND)

    def record_first_finished() -> None:
        events.append("first:finished:start")
        time.sleep(0.05)
        events.append("first:finished:end")

    first.finished.connect(record_first_finished)
    second.finished.connect(lambda: events.append("second:finished"))

    gate.set()
    assert manager.wait_for_done(2000)
    _wait_until(lambda: "second:finished" in events)

    assert events.index("first:finished:end") < events.index("second:process")


def test_headless_job_signal_sender_and_cleanup_state(headless_jobs) -> None:
    class _OneItemJob(headless_jobs.BatchJob):
        def __init__(self, start_event: threading.Event) -> None:
            super().__init__([1])
            self._start_event = start_event

        def prepare(self) -> None:
            assert self._start_event.wait(2.0)

    manager = headless_jobs.JobManager(max_workers=1)
    start_event = threading.Event()
    signals = manager.submit(_OneItemJob(start_event))
    senders: list[object | None] = []
    finished: list[None] = []

    signals.finished.connect(lambda: (senders.append(signals.sender()), finished.append(None)))

    start_event.set()
    assert manager.wait_for_done(2000)
    _wait_until(lambda: bool(finished))

    assert senders == [signals]
    assert signals not in manager._active_signals
    assert signals not in manager._signal_to_runnable
