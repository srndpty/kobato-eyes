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

_HEADLESS_MODE = is_headless()

if _HEADLESS_MODE:
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
            thread.start()
            self._threads.append(thread)

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
            delay = max(0.001, interval_ms / 1000.0)
            timer = threading.Timer(delay, callback)
            timer.daemon = True
            timer.start()

    class _Signal:
        def __init__(self, owner: QObject) -> None:
            self._owner = owner
            self._callbacks: list[Callable[..., None]] = []
            self._deferred: list[Callable[[], None]] = []
            self._emitting = 0

        def connect(self, callback: Callable[..., None]) -> None:
            self._callbacks.append(callback)

        def disconnect(self, callback: Callable[..., None]) -> None:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

        def defer_after_emit(self, callback: Callable[[], None]) -> None:
            """Run ``callback`` after the current signal emission completes."""

            if self._emitting <= 0:
                callback()
                return
            self._deferred.append(callback)

        def emit(self, *args, **kwargs) -> None:
            self._emitting += 1
            try:
                for callback in list(self._callbacks):
                    QObject._push_sender(self._owner)
                    try:
                        callback(*args, **kwargs)
                    finally:
                        QObject._pop_sender()
            finally:
                self._emitting -= 1
            if self._emitting == 0 and self._deferred:
                deferred = list(self._deferred)
                self._deferred.clear()
                for callback in deferred:
                    callback()

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
    progressState = pyqtSignal(object)
    completed = pyqtSignal(object)
    cancelled = pyqtSignal()
    error = pyqtSignal(object, str)
    finished = pyqtSignal()


class BatchJob(ABC):
    """Template for batch-oriented jobs with load/process/write stages."""

    def __init__(self, items: Sequence[Any] | None = None, *, name: str | None = None) -> None:
        self.items: list[Any] = list(items or [])
        self.name = name or self.__class__.__name__
        self._cancel_event = threading.Event()
        self._progress_state_cb: Callable[[object], None] | None = None

    def cancel(self) -> None:
        """Request cooperative cancellation before the next item is processed."""

        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._cancel_event.is_set()

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

    def emit_progress_state(self, payload: object) -> None:
        """Emit a structured progress payload when a manager is running this job."""

        if self._progress_state_cb is not None:
            self._progress_state_cb(payload)

    def execute(self, progress_cb: Callable[[int, int], None]) -> Any:
        """Execute the job and return the completion payload."""

        total = len(self.items)
        if self.is_cancelled():
            raise _JobCancelled
        if total == 0:
            return self.finalize()

        for index, item in enumerate(self.items, start=1):
            if self.is_cancelled():
                raise _JobCancelled
            loaded = self.load_item(item)
            if self.is_cancelled():
                raise _JobCancelled
            processed = self.process_item(item, loaded)
            if self.is_cancelled():
                raise _JobCancelled
            self.write_item(item, processed)
            progress_cb(index, total)

        return self.finalize()

    def _bind_progress_state(self, callback: Callable[[object], None] | None) -> None:
        self._progress_state_cb = callback


class CallableJob(BatchJob):
    """Single-call job for non-batch background tasks."""

    def __init__(
        self,
        func: Callable[[Callable[[], bool], Callable[[object], None]], Any],
        *,
        cleanup: Callable[[], None] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__([], name=name)
        self._func = func
        self._cleanup = cleanup

    def execute(self, progress_cb: Callable[[int, int], None]) -> Any:
        """Execute the wrapped callable."""

        del progress_cb
        if self.is_cancelled():
            raise _JobCancelled
        return self._func(self.is_cancelled, self.emit_progress_state)

    def cleanup(self) -> None:
        """Run the optional cleanup callback."""

        if self._cleanup is not None:
            self._cleanup()


class _JobCancelled(Exception):
    """Internal sentinel used to stop job execution without reporting an error."""


class _BatchJobRunnable(QRunnable):
    """Internal runnable executing a :class:`BatchJob` instance."""

    def __init__(self, job: BatchJob) -> None:
        super().__init__()
        self.job = job
        self.signals = JobSignals()
        self.running = False
        self.started = False
        self.completed = False

    def run(self) -> None:  # noqa: D401 - QRunnable entry point
        self.started = True
        self.running = True
        self.job._bind_progress_state(self.signals.progressState.emit)
        try:
            self.job.prepare()
            result = self.job.execute(self.signals.progress.emit)
            if self.job.is_cancelled():
                self.signals.cancelled.emit()
                return
            self.signals.completed.emit(result)
        except _JobCancelled:
            self.signals.cancelled.emit()
        except Exception as exc:  # pragma: no cover - exercised via error signal
            tb = traceback.format_exc()
            self.signals.error.emit(exc, tb)
        finally:
            try:
                self.job.cleanup()
            finally:
                self.job._bind_progress_state(None)
                self.running = False
                self.completed = True
                self.signals.finished.emit()

    def cancel_without_start(self) -> None:
        """Cancel a queued job before it reaches the thread pool."""

        self.job.cancel()
        try:
            self.job.cleanup()
        finally:
            self.completed = True
            self.signals.cancelled.emit()
            self.signals.finished.emit()


class JobHandle:
    """Handle returned for managed jobs that need lifecycle control."""

    def __init__(self, manager: "JobManager", runnable: _BatchJobRunnable) -> None:
        self._manager = manager
        self._runnable = runnable
        self.signals = runnable.signals

    @property
    def name(self) -> str:
        """Return the human-readable job name."""

        return self._runnable.job.name

    @property
    def is_running(self) -> bool:
        """Return whether the job is currently executing."""

        return self._runnable.running

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._runnable.job.is_cancelled()

    def cancel(self) -> None:
        """Request cancellation, including while the job is still queued."""

        self._runnable.job.cancel()
        self._manager._cancel_pending(self._runnable)


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
        return self.submit_handle(job, priority=priority).signals

    def submit_handle(self, job: BatchJob, priority: JobPriority = JobPriority.BACKGROUND) -> JobHandle:
        """Queue ``job`` for execution and return a lifecycle handle."""

        runnable = _BatchJobRunnable(job)
        signals = runnable.signals
        signals.finished.connect(self._handle_job_finished)
        self._active_signals.add(signals)
        self._signal_to_runnable[signals] = runnable
        heapq.heappush(self._pending, (int(priority), self._sequence, runnable))
        self._sequence += 1
        if priority == JobPriority.FOREGROUND:
            self._drain_pending()
        else:
            self._request_schedule()
        return JobHandle(self, runnable)

    def has_pending_jobs(self) -> bool:
        """Return whether jobs are queued or running."""
        return bool(self._pending or self._running or self._schedule_pending)

    def wait_for_done(self, timeout_ms: int = -1) -> bool:
        """Block until all jobs complete or ``timeout_ms`` elapses."""
        if _HEADLESS_MODE:
            deadline = None if timeout_ms < 0 else time.monotonic() + (timeout_ms / 1000.0)
            while self.has_pending_jobs():
                if deadline is None:
                    slice_ms = 100
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    slice_ms = max(1, min(100, int(remaining * 1000)))
                self._pool.waitForDone(slice_ms)
                if self.has_pending_jobs():
                    time.sleep(0.001)
            return self._pool.waitForDone(0)
        return self._pool.waitForDone(timeout_ms)

    def shutdown(self, timeout_ms: int = -1) -> bool:
        """Wait for completion and return whether all jobs finished."""
        return self.wait_for_done(timeout_ms)

    def _request_schedule(self) -> None:
        if not self._pending:
            return
        if self._schedule_pending:
            return
        if self._running >= self._pool.maxThreadCount():
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

    def _cancel_pending(self, runnable: _BatchJobRunnable) -> None:
        for index, (_priority, _sequence, pending) in enumerate(self._pending):
            if pending is runnable:
                self._pending.pop(index)
                heapq.heapify(self._pending)
                runnable.cancel_without_start()
                return

    def _handle_job_finished(self) -> None:
        sender = self.sender()
        if not isinstance(sender, JobSignals):
            return
        if _HEADLESS_MODE:
            sender.finished.defer_after_emit(lambda: self._complete_finished_job(sender))
            return
        self._complete_finished_job(sender)

    def _complete_finished_job(self, signals: JobSignals) -> None:
        runnable = self._signal_to_runnable.pop(signals, None)
        try:
            signals.finished.disconnect(self._handle_job_finished)
        except TypeError:
            pass
        self._active_signals.discard(signals)
        signals.deleteLater()
        if runnable is None or runnable.started:
            self._running = max(0, self._running - 1)
        self._request_schedule()


__all__ = [
    "BatchJob",
    "CallableJob",
    "JobHandle",
    "JobManager",
    "JobPriority",
    "JobSignals",
]
