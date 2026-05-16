"""Background worker to execute tag searches without blocking the UI."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PyQt6.QtCore import QCoreApplication, QObject, QThread, pyqtSignal, pyqtSlot

from db.repository import search_files as _search_files

logger = logging.getLogger(__name__)


def _is_deleted_qobject_error(exc: RuntimeError) -> bool:
    """Return whether Qt rejected an operation because the QObject is gone."""

    return "has been deleted" in str(exc)


@dataclass(frozen=True)
class _SearchParams:
    """Container describing a single search execution."""

    where_sql: str
    params: tuple[object, ...]
    tags_for_relevance: tuple[str, ...]
    thresholds: Mapping[int, float] | None
    order: str
    chunk_size: int
    offset: int
    max_rows: int | None


class SearchWorker(QObject):
    """Run tag searches against SQLite in a background thread."""

    chunkReady = pyqtSignal(list)
    finished = pyqtSignal(bool, bool)
    error = pyqtSignal(str)

    def __init__(
        self,
        db_path: str | Path,
        where_sql: str,
        params: Sequence[object] | tuple[object, ...],
        *,
        tags_for_relevance: Sequence[str] | None = None,
        thresholds: Mapping[int, float] | None = None,
        order: str = "mtime",
        chunk: int = 200,
        offset: int = 0,
        max_rows: int | None = None,
        chunk_delay: float = 0.0,
    ) -> None:
        super().__init__()
        normalized_chunk = max(1, int(chunk))
        normalized_offset = max(0, int(offset))
        self._params = _SearchParams(
            where_sql=str(where_sql or "1=1"),
            params=tuple(params or ()),
            tags_for_relevance=tuple(tags_for_relevance or ()),
            thresholds=thresholds,
            order=order,
            chunk_size=normalized_chunk,
            offset=normalized_offset,
            max_rows=max_rows if max_rows is None else max(0, int(max_rows)),
        )
        self._db_path = str(Path(db_path))
        self._chunk_delay = max(0.0, float(chunk_delay))
        self._cancel = threading.Event()
        self._connection: sqlite3.Connection | None = None

    @pyqtSlot()
    def run(self) -> None:
        """Execute the configured search, emitting results progressively."""

        if self._cancel.is_set():
            self._finish(False, True)
            return

        try:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
        except sqlite3.Error as exc:  # pragma: no cover - connection errors are surfaced to UI
            logger.warning("SearchWorker failed to open database %s: %s", self._db_path, exc)
            if self._cancel.is_set():
                self._finish(False, True)
            else:
                self.error.emit(str(exc))
                self._finish(False, False)
            return

        self._connection = conn
        conn.row_factory = sqlite3.Row
        conn.set_progress_handler(self._progress_handler, 10_000)

        try:
            params = self._params
            emitted = 0
            offset = params.offset

            while not self._cancel.is_set():
                if params.max_rows is not None:
                    remaining = params.max_rows - emitted
                    if remaining <= 0:
                        break
                    limit = min(params.chunk_size, remaining)
                else:
                    limit = params.chunk_size

                rows = _search_files(
                    conn,
                    params.where_sql,
                    params.params,
                    tags_for_relevance=params.tags_for_relevance,
                    thresholds=params.thresholds,
                    order=params.order,
                    limit=limit,
                    offset=offset,
                )
                if not rows:
                    break
                try:
                    self._emit_chunk(rows)
                except RuntimeError as exc:
                    if _is_deleted_qobject_error(exc):
                        logger.debug("SearchWorker stopped because the QObject was deleted")
                        self._cancel.set()
                        self._finish(False, True, ensure_thread_quit=True)
                        return
                    raise
                emitted += len(rows)
                offset += len(rows)
                if self._chunk_delay:
                    self._cancel.wait(self._chunk_delay)
                if params.max_rows is not None and emitted >= params.max_rows:
                    break
                if len(rows) < limit:
                    break
        except sqlite3.OperationalError as exc:
            if self._cancel.is_set() or "interrupted" in str(exc).lower() and self._cancel.is_set():
                self._finish(False, True)
            else:
                logger.warning("SearchWorker query failed: %s", exc)
                self.error.emit(str(exc))
                self._finish(False, False)
        except RuntimeError as exc:
            if _is_deleted_qobject_error(exc):
                logger.debug("SearchWorker stopped because the QObject was deleted")
                self._cancel.set()
                self._finish(False, True, ensure_thread_quit=True)
            else:
                raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("SearchWorker crashed")
            self.error.emit(str(exc))
            self._finish(False, False)
        else:
            is_cancelled = self._cancel.is_set()
            self._finish(not is_cancelled, is_cancelled)
        finally:
            try:
                conn.close()
            finally:
                self._connection = None

    def cancel(self) -> None:
        """Signal the worker to stop as soon as possible."""

        self._cancel.set()
        connection = self._connection
        if connection is not None:
            with suppress(sqlite3.Error):
                connection.interrupt()

    def _progress_handler(self) -> int:
        return 1 if self._cancel.is_set() else 0

    def _finish(self, ok: bool, cancelled: bool, *, ensure_thread_quit: bool = False) -> None:
        """Notify listeners and directly stop the worker thread when signals cannot."""

        emitted = self._emit_finished(ok, cancelled)
        if ensure_thread_quit or not emitted:
            self._quit_current_worker_thread()

    def _emit_finished(self, ok: bool, cancelled: bool) -> bool:
        """Emit completion unless the underlying QObject has already been deleted."""

        try:
            self.finished.emit(ok, cancelled)
        except RuntimeError as exc:
            if _is_deleted_qobject_error(exc):
                logger.debug("SearchWorker could not emit finished because the QObject was deleted")
                return False
            raise
        return True

    def _quit_current_worker_thread(self) -> None:
        """Quit the current QThread without touching the GUI main thread."""

        app = QCoreApplication.instance()
        if app is None:
            return
        current = QThread.currentThread()
        if current is None:
            return
        if current is app.thread():
            return
        current.quit()

    def _emit_chunk(self, rows: list[dict[str, object]]) -> None:
        """Emit a result chunk."""

        self.chunkReady.emit(rows)


__all__ = ["SearchWorker"]
