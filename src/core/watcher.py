"""Watchdog-based directory monitoring for kobato-eyes."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Callable, Iterable, Literal

if TYPE_CHECKING:
    from core.watcher import DirectoryWatcher  # or 同一モジュールなら相対 import 不要
    # from watchdog.observers import Observer

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from utils.env import is_headless
from utils.fs import from_system_path, is_hidden, to_system_path

if is_headless():

    class QObject:  # type: ignore[too-many-ancestors]
        """Simple QObject replacement when Qt is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - Qt-compatible signature
            pass

    class _Signal:
        def __init__(self) -> None:
            self._callbacks: list[Callable[..., None]] = []

        def connect(self, callback: Callable[..., None]) -> None:
            self._callbacks.append(callback)

        def emit(self, *args, **kwargs) -> None:
            for callback in list(self._callbacks):
                callback(*args, **kwargs)

        def disconnect(self, callback: Callable[..., None]) -> None:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    class QTimer:
        """Stub timer that mimics the Qt API shape."""

        def __init__(self, parent: QObject | None = None) -> None:
            self._interval = 0
            self.timeout = _Signal()

        def setInterval(self, interval: int) -> None:
            self._interval = int(interval)

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass


else:  # pragma: no branch - trivial import guard
    from PyQt6.QtCore import QObject, QTimer

FileEventType = Literal["created", "modified", "moved"]

logger = logging.getLogger(__name__)


class _ImageEventHandler(FileSystemEventHandler):
    """Dispatch file system events to the owning watcher."""

    def __init__(self, watcher: "DirectoryWatcher") -> None:  # type: ignore[name-defined]
        super().__init__()
        self._watcher = watcher

    def on_created(self, event: FileSystemEvent) -> None:  # noqa: D401
        self._watcher.process_event(event, "created")

    def on_modified(self, event: FileSystemEvent) -> None:  # noqa: D401
        self._watcher.process_event(event, "modified")

    def on_moved(self, event: FileSystemEvent) -> None:  # noqa: D401
        super().on_moved(event)
        self._watcher.process_event(event, "moved")


class DirectoryWatcher(QObject):
    """Manage watchdog observers and batch events for indexing."""

    def __init__(
        self,
        roots: Iterable[str | Path],
        *,
        callback: Callable[[list[Path]], None],
        excluded: Iterable[str | Path] | None = None,
        extensions: Iterable[str] | None = None,
        recursive: bool = True,
        batch_interval: float = 2.0,
        use_qtimer: bool = True,
        observer_factory: Callable[[], Observer] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._roots = [Path(root).expanduser() for root in roots]
        self._callback = callback
        self._excluded = {self._resolve_path(Path(path)) for path in (excluded or [])}
        self._extensions = self._normalise_extensions(extensions)
        self._recursive = recursive
        self._observer_factory = observer_factory or Observer
        self._pending: set[Path] = set()
        self._pending_lock = Lock()
        self._observers: list[Observer] = []
        self._running = False
        self._state_lock = Lock()
        self._batch_interval = max(0.1, float(batch_interval))
        self._use_qtimer = use_qtimer
        self._timer: QTimer | None = None
        if self._use_qtimer:
            self._timer = QTimer(self)
            self._timer.setInterval(int(self._batch_interval * 1000))
            self._timer.timeout.connect(self.flush_pending)

    @staticmethod
    def _normalise_extensions(extensions: Iterable[str] | None) -> set[str] | None:
        if extensions is None:
            return None
        normalised: set[str] = set()
        for ext in extensions:
            candidate = ext.strip().lower()
            if not candidate:
                continue
            if not candidate.startswith("."):
                candidate = f".{candidate}"
            normalised.add(candidate)
        return normalised or None

    def start(self) -> None:
        """Start monitoring the configured directories."""
        with self._state_lock:
            if self._running:
                return
            self._observers = []
            handler = _ImageEventHandler(self)
            for root in self._roots:
                if not root.exists() or not root.is_dir():
                    continue
                observer = self._observer_factory()
                observer.schedule(handler, to_system_path(root), recursive=self._recursive)
                try:
                    observer.start()
                except Exception:  # pragma: no cover - watchdog defensive logging
                    logger.warning("Failed to start observer for %s", root, exc_info=True)
                    continue
                self._observers.append(observer)
            self._running = True
            if self._timer is not None:
                self._timer.start()

    def stop(self) -> None:
        """Stop the observer threads and flush outstanding events."""
        observers: list[Observer]
        with self._state_lock:
            if not self._running:
                return
            observers = list(self._observers)
            self._observers.clear()
            self._running = False
        if self._timer is not None:
            self._timer.stop()
        for observer in observers:
            try:
                observer.stop()
            except Exception:  # pragma: no cover - watchdog defensive logging
                logger.warning("Failed to stop observer", exc_info=True)
        for observer in observers:
            try:
                observer.join()
            except Exception:  # pragma: no cover - watchdog defensive logging
                logger.warning("Failed to join observer", exc_info=True)

    def is_running(self) -> bool:
        """Return whether the watcher is currently active."""
        with self._state_lock:
            return self._running

    def process_event(self, event: FileSystemEvent, event_type: FileEventType) -> None:
        """Handle a watchdog event coming from any observer."""
        if event.is_directory:
            return
        if event_type == "moved":
            dest_path = getattr(event, "dest_path", None)
            candidate = from_system_path(dest_path or event.src_path)
        else:
            candidate = from_system_path(event.src_path)
        self._record_candidate(candidate)

    def notify_path(self, path: Path) -> None:
        """Public helper primarily for tests to enqueue a path manually."""
        self._record_candidate(Path(path))

    def flush_pending(self) -> None:
        """Flush accumulated events and invoke the callback once per batch."""
        with self._pending_lock:
            if not self._pending:
                return
            batch = sorted(self._pending, key=lambda item: str(item))
            self._pending.clear()
        try:
            self._callback(batch)
        except Exception:  # pragma: no cover - downstream defensive logging
            logger.warning("Failed to dispatch batch of %d paths", len(batch), exc_info=True)

    def _record_candidate(self, path: Path) -> None:
        if not self._should_track(path):
            return
        normalised = self._normalise_path(path)
        with self._pending_lock:
            self._pending.add(normalised)

    def _should_track(self, path: Path) -> bool:
        if is_hidden(path):
            return False
        resolved = self._resolve_path(path)
        for excluded in self._excluded:
            try:
                resolved.relative_to(excluded)
                return False
            except ValueError:
                continue
        if self._extensions is None:
            return True
        suffix = path.suffix.lower()
        return suffix in self._extensions

    @staticmethod
    def _resolve_path(path: Path) -> Path:
        candidate = Path(path).expanduser()
        try:
            return candidate.resolve()
        except OSError:
            try:
                return candidate.resolve(strict=False)
            except OSError:
                return candidate.absolute()

    @classmethod
    def _normalise_path(cls, path: Path) -> Path:
        return cls._resolve_path(path)


__all__ = ["DirectoryWatcher", "FileEventType"]
