"""Job management helpers with optional Qt fallbacks for headless mode."""

from __future__ import annotations

import enum
import heapq
import threading
import time
import traceback
from abc import ABC
from typing import Any, Callable, Sequence

from utils.env import is_headless

if is_headless():
    _sender_local = threading.local()

    class QObject:  # type: ignore[too-many-ancestors]
        """Very small QObject stand-in used without PyQt6."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - Qt-compatible signature
            pass

        def deleteLater(self) -> None:  # noqa: D401 - Qt-compatible signature
            pass

        @staticmethod
        def _push_sender(sender: "QObject") -> None:
            stack = getattr(_sender_local, "stack", None)
            if stack is None:
                stack = []
                _sender_local.stack = stack
            stack.append(sender)

        @staticmethod
        def _pop_sender() -> None:
            stack = getattr(_sender_local, "stack", None)
            if stack:
                stack.pop()

        def sender(self) -> "QObject | None":  # noqa: D401 - Qt-compatible signature
            stack = getattr(_sender_local, "stack", None)
            if stack:
                return stack[-1]
            return None


    class QRunnable:
        """Minimal runnable base class compatible with the Qt API."""

        def run(self) -> None:  # noqa: D401 - Qt-compatible signature
            raise NotImplementedError


    class QThreadPool:
        """Thread pool that mimics the Qt API using Python threads."""

        def __init__(self) -> None:
            self._max_thread_count = max(1, threading.active_count())
            self._threads: list[threading.Thread] = []

        def setMaxThreadCount(self, count: int) -> None:
            self._max_thread_count = max(1, int(count))

        def maxThreadCount(self) -> int:
            return self._max_thread_count

        def start(self, runnable: QRunnable) -> None:
            thread = threading.Thread(target=runnable.run, daemon=True)
            self._threads.append(thread)
            thread.start()

        def waitForDone(self, timeout_ms: int = -1) -> bool:
            deadline = None
            if timeout_ms >= 0:
                deadline = time.time() + (timeout_ms / 1000.0)
            for thread in list(self._threads):
                remaining = None
                if deadline is not None:
                    remaining = max(0.0, deadline - time.time())
                thread.join(remaining)
            finished = all(not thread.is_alive() for thread in self._threads)
            if finished:
                self._threads.clear()
            return finished


    class QTimer:
        """Provide the static Qt timer helpers needed by JobManager."""

        @staticmethod
        def singleShot(interval_ms: int, callback: Callable[[], None]) -> None:
            if interval_ms <= 0:
                callback()
                return
            timer = threading.Timer(interval_ms / 1000.0, callback)
            timer.daemon = True
            timer.start()


    class _Signal:
        def __init__(self, owner: QObject) -> None:
            self._owner = owner
            self._callbacks: list[Callable[..., None]] = []

        def connect(self, callback: Callable[..., None]) -> None:
            self._callbacks.append(callback)

        def disconnect(self, callback: Callable[..., None]) -> None:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

        def emit(self, *args, **kwargs) -> None:
            for callback in list(self._callbacks):
                QObject._push_sender(self._owner)
                try:
                    callback(*args, **kwargs)
                finally:
                    QObject._pop_sender()


    class _SignalDescriptor:
        def __init__(self) -> None:
            self._name: str | None = None

        def __set_name__(self, owner: type[QObject], name: str) -> None:
            self._name = name

        def __get__(self, instance: QObject | None, owner: type[QObject]) -> _Signal:
            if instance is None:
                raise AttributeError("Signal descriptors are only available on instances")
            if self._name is None:
                raise AttributeError("Signal descriptor not initialised")
            signal = instance.__dict__.get(self._name)
            if signal is None:
                signal = _Signal(instance)
                instance.__dict__[self._name] = signal
            return signal


    def pyqtSignal(*_args, **_kwargs) -> _SignalDescriptor:  # noqa: D401 - Qt-compatible signature
        return _SignalDescriptor()


else:  # pragma: no branch - trivial import guard
    from PyQt6.QtCore import QObject, QRunnable, QThreadPool, QTimer, pyqtSignal


class JobPriority(enum.IntEnum):
    """Enumerate job priorities for the scheduler."""

    FOREGROUND = 0
    BACKGROUND = 1


class JobSignals(QObject):
    """Signals emitted during job execution."""

    progress = pyqtSignal(int, int)
    completed = pyqtSignal(object)
    error = pyqtSignal(object, str)
    finished = pyqtSignal()


class BatchJob(ABC):
    """Template for batch-oriented jobs with load/process/write stages."""

    def __init__(self, items: Sequence[Any] | None = None) -> None:
        self.items: list[Any] = list(items or [])

    def prepare(self) -> None:
        """Hook executed before the batch is processed."""

    def load_item(self, item: Any) -> Any:
        """Load resources for ``item`` (e.g. disk IO)."""
        return item

    def process_item(self, item: Any, loaded: Any) -> Any:
        """Transform the loaded payload into a result."""
        return loaded

    def write_item(self, item: Any, processed: Any) -> None:
        """Persist the processed payload (e.g. database writes)."""

    def finalize(self) -> Any:
        """Return the result that should be emitted on completion."""
        return None

    def cleanup(self) -> None:
        """Hook executed after the batch completes (success or failure)."""


class _BatchJobRunnable(QRunnable):
    """Internal runnable executing a :class:`BatchJob` instance."""

    def __init__(self, job: BatchJob) -> None:
        super().__init__()
        self.job = job
        self.signals = JobSignals()

    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        try:
            self.job.prepare()
            total = len(self.job.items)
            if total == 0:
                result = self.job.finalize()
                self.signals.completed.emit(result)
                return

            for index, item in enumerate(self.job.items, start=1):
                loaded = self.job.load_item(item)
                processed = self.job.process_item(item, loaded)
                self.job.write_item(item, processed)
                self.signals.progress.emit(index, total)

            result = self.job.finalize()
            self.signals.completed.emit(result)
        except Exception as exc:  # pragma: no cover - exercised via error signal
            tb = traceback.format_exc()
            self.signals.error.emit(exc, tb)
        finally:
            try:
                self.job.cleanup()
            finally:
                self.signals.finished.emit()


class JobManager(QObject):
    """Manage job submission and execution on a thread pool."""

    def __init__(self, *, max_workers: int | None = None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool()
        if max_workers is not None:
            self._pool.setMaxThreadCount(max_workers)
        self._pending: list[tuple[int, int, _BatchJobRunnable]] = []
        self._sequence = 0
        self._running = 0
        self._active_signals: set[JobSignals] = set()
        self._signal_to_runnable: dict[JobSignals, _BatchJobRunnable] = {}
        self._schedule_pending = False

    def submit(self, job: BatchJob, priority: JobPriority = JobPriority.BACKGROUND) -> JobSignals:
        """Queue ``job`` for execution and return its associated signals."""
        runnable = _BatchJobRunnable(job)
        signals = runnable.signals
        signals.finished.connect(self._handle_job_finished)
        self._active_signals.add(signals)
        self._signal_to_runnable[signals] = runnable
        heapq.heappush(self._pending, (int(priority), self._sequence, runnable))
        self._sequence += 1
        self._request_schedule()
        return signals

    def has_pending_jobs(self) -> bool:
        """Return whether jobs are queued or running."""
        return bool(self._pending or self._running)

    def wait_for_done(self, timeout_ms: int = -1) -> bool:
        """Block until all jobs complete or ``timeout_ms`` elapses."""
        return self._pool.waitForDone(timeout_ms)

    def shutdown(self, timeout_ms: int = -1) -> bool:
        """Wait for completion and return whether all jobs finished."""
        return self.wait_for_done(timeout_ms)

    def _request_schedule(self) -> None:
        if self._schedule_pending:
            return
        self._schedule_pending = True
        QTimer.singleShot(0, self._drain_pending)

    def _drain_pending(self) -> None:
        self._schedule_pending = False
        self._start_available_jobs()

    def _start_available_jobs(self) -> None:
        while self._pending and self._running < self._pool.maxThreadCount():
            _, _, runnable = heapq.heappop(self._pending)
            self._running += 1
            self._pool.start(runnable)

    def _handle_job_finished(self) -> None:
        sender = self.sender()
        if not isinstance(sender, JobSignals):
            return
        signals = sender
        self._signal_to_runnable.pop(signals, None)
        try:
            signals.finished.disconnect(self._handle_job_finished)
        except TypeError:
            pass
        self._active_signals.discard(signals)
        signals.deleteLater()
        self._running = max(0, self._running - 1)
        self._start_available_jobs()


__all__ = [
    "BatchJob",
    "JobManager",
    "JobPriority",
    "JobSignals",
]
