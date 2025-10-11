"""Background worker to execute tag searches without blocking the UI."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from db.repository import search_files as _search_files


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
    totalKnown = pyqtSignal(int)
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
        include_total: bool = True,
        count_timeout: float = 0.25,
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
        self._include_total = bool(include_total)
        self._count_timeout = max(0.0, float(count_timeout))
        self._chunk_delay = max(0.0, float(chunk_delay))
        self._cancel = threading.Event()
        self._connection: sqlite3.Connection | None = None
        self._count_deadline: float | None = None
        self._count_timed_out = False

    @pyqtSlot()
    def run(self) -> None:
        """Execute the configured search, emitting results progressively."""

        try:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
        except sqlite3.Error as exc:  # pragma: no cover - connection errors are surfaced to UI
            self.error.emit(str(exc))
            self.finished.emit(False, False)
            return

        self._connection = conn
        conn.row_factory = sqlite3.Row
        conn.set_progress_handler(self._progress_handler, 10_000)

        try:
            total = self._try_count_with_timeout(conn) if self._include_total else None
            if total is not None:
                self.totalKnown.emit(int(total))

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
                self.chunkReady.emit(rows)
                emitted += len(rows)
                offset += len(rows)
                if self._chunk_delay:
                    time.sleep(self._chunk_delay)
                if len(rows) < limit:
                    break
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower() and (self._cancel.is_set() or self._count_timed_out):
                self.finished.emit(False, True)
            else:
                self.error.emit(str(exc))
                self.finished.emit(False, False)
        except Exception as exc:  # pragma: no cover - defensive logging
            self.error.emit(str(exc))
            self.finished.emit(False, False)
        else:
            is_cancelled = self._cancel.is_set()
            self.finished.emit(not is_cancelled, is_cancelled)
        finally:
            try:
                conn.close()
            finally:
                self._connection = None
                self._count_deadline = None
                self._count_timed_out = False

    def cancel(self) -> None:
        """Signal the worker to stop as soon as possible."""

        self._cancel.set()
        connection = self._connection
        if connection is not None:
            try:
                connection.interrupt()
            except sqlite3.Error:  # pragma: no cover - best effort
                pass

    def _progress_handler(self) -> int:
        if self._cancel.is_set():
            return 1
        if self._count_deadline is not None and time.perf_counter() >= self._count_deadline:
            self._count_timed_out = True
            self._count_deadline = None
            return 1
        return 0

    def _try_count_with_timeout(self, conn: sqlite3.Connection) -> int | None:
        if self._count_timeout <= 0:
            return None
        self._count_timed_out = False
        self._count_deadline = time.perf_counter() + self._count_timeout
        cursor = conn.cursor()
        try:
            where_sql = self._params.where_sql.strip() or "1=1"
            query = (
                "SELECT COUNT(*) FROM files f "
                "WHERE (" + where_sql + ") AND f.is_present = 1"
            )
            cursor.execute(query, self._params.params)
            row = cursor.fetchone()
            if row is None:
                return 0
            return int(row[0])
        except sqlite3.OperationalError as exc:
            if "interrupted" in str(exc).lower() and self._count_timed_out:
                return None
            raise
        finally:
            self._count_deadline = None
            cursor.close()


__all__ = ["SearchWorker"]
