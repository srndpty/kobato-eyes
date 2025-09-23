"""Watchdog-based directory monitoring for kobato-eyes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Iterable, Literal

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from utils.fs import from_system_path, is_hidden, to_system_path

FileEventType = Literal["created", "modified", "moved"]


@dataclass(frozen=True)
class FileEvent:
    """Container for file-system events relevant to image ingestion."""

    path: Path
    event_type: FileEventType


class _ImageEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        queue: "Queue[FileEvent]",
        extensions: set[str] | None,
        excluded: set[Path],
    ) -> None:
        super().__init__()
        self._queue = queue
        self._extensions = extensions
        self._excluded = excluded

    def _should_track(self, path: Path) -> bool:
        if is_hidden(path):
            return False
        for excluded in self._excluded:
            try:
                path.resolve().relative_to(excluded)
                return False
            except ValueError:
                continue
        if self._extensions is None:
            return True
        return path.suffix.lower() in self._extensions

    def _enqueue(self, event: FileSystemEvent, event_type: FileEventType) -> None:
        if event.is_directory:
            return
        candidate = from_system_path(event.src_path)
        if not self._should_track(candidate):
            return
        self._queue.put(FileEvent(path=candidate, event_type=event_type))

    def on_created(self, event: FileSystemEvent) -> None:  # noqa: D401
        self._enqueue(event, "created")

    def on_modified(self, event: FileSystemEvent) -> None:  # noqa: D401
        self._enqueue(event, "modified")

    def on_moved(self, event: FileSystemEvent) -> None:  # noqa: D401
        super().on_moved(event)
        if event.is_directory:
            return
        candidate = from_system_path(getattr(event, "dest_path", event.src_path))
        if self._should_track(candidate):
            self._queue.put(FileEvent(path=candidate, event_type="moved"))


class DirectoryWatcher:
    """Manage watchdog observers for a set of directories."""

    def __init__(
        self,
        roots: Iterable[str | Path],
        queue: "Queue[FileEvent]",
        *,
        excluded: Iterable[str | Path] | None = None,
        extensions: Iterable[str] | None = None,
        recursive: bool = True,
    ) -> None:
        self._roots = [Path(root).resolve() for root in roots]
        self._queue = queue
        self._excluded = {Path(path).resolve() for path in (excluded or [])}
        self._extensions = self._normalise_extensions(extensions)
        self._recursive = recursive
        self._observer: Observer | None = None
        self._lock = Lock()

    @staticmethod
    def _normalise_extensions(extensions: Iterable[str] | None) -> set[str] | None:
        if extensions is None:
            return None
        normalised: set[str] = set()
        for ext in extensions:
            candidate = ext.lower()
            if not candidate.startswith("."):
                candidate = f".{candidate}"
            normalised.add(candidate)
        return normalised

    def start(self) -> None:
        """Start monitoring the configured directories."""
        with self._lock:
            if self._observer is not None:
                return
            observer = Observer()
            handler = _ImageEventHandler(self._queue, self._extensions, self._excluded)
            for root in self._roots:
                if not root.exists() or not root.is_dir():
                    continue
                observer.schedule(
                    handler, to_system_path(root), recursive=self._recursive
                )
            observer.start()
            self._observer = observer

    def stop(self) -> None:
        """Stop the observer and wait for watcher threads to finish."""
        with self._lock:
            observer = self._observer
            if observer is None:
                return
            observer.stop()
            observer.join()
            self._observer = None

    def is_running(self) -> bool:
        """Return whether the watcher is currently active."""
        with self._lock:
            return self._observer is not None


__all__ = ["FileEvent", "FileEventType", "DirectoryWatcher"]
