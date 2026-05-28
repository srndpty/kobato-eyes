"""Background service responsible for batching DB writes from tagging."""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from typing import Dict, List, Optional

from core.pipeline.contracts import DBFlush, DBItem, DBStop, DBWriteQueue
from db.connection import get_conn
from db.fts import fts_replace_rows
from services.db_writing_lifecycle import DBWritingPragmas
from services.db_writing_staging import StagingBatchWriter, StagingMerger, chunked_table, rowid_windows
from services.db_writing_standard import (
    StandardBatchWriter,
    bulk_update_files_meta_by_id_uncommitted,
    chunked,
    upsert_tags_uncommitted,
)

logger = logging.getLogger(__name__)


class DBWritingService(DBWriteQueue):
    """Thread-backed queue that persists tagging results in batches."""

    _STARTUP_WAIT_TIMEOUT_SECONDS = 5.0
    _SHUTDOWN_LOG_INTERVAL_SECONDS = 60.0

    def __init__(
        self,
        db_path: str,
        *,
        flush_chunk: int = 1024,
        fts_topk: int = 128,
        queue_size: int = 1024,
        default_tagger_sig: str | None = None,
        unsafe_fast: bool = False,
        skip_fts: bool = False,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        self._log = logging.getLogger(__name__)
        self._db_path = db_path
        self._flush_chunk = self._require_positive_int("flush_chunk", flush_chunk)
        self._fts_topk = max(0, int(fts_topk))
        queue_size = self._require_positive_int("queue_size", queue_size)
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)
        self._default_tagger_sig = default_tagger_sig
        self._unsafe_fast = bool(unsafe_fast)
        self._skip_fts = bool(skip_fts)
        self._progress_cb = progress_cb
        self._exc: BaseException | None = None
        self._exc_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._ready_evt = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, name="DBWritingService", daemon=True)
        self._started = False
        self._written = 0
        self._flush_count = 0
        self._tag_cache: Dict[str, int] = {}
        self._stage_tags_in_temp = True
        self._debug = os.environ.get("KE_DBWRITER_DEBUG") == "1"
        self._pragmas = DBWritingPragmas(self._log)
        self._standard_writer = StandardBatchWriter(fts_replace_rows)
        self._staging_writer = StagingBatchWriter()
        if os.environ.get("KE_SKIP_FTS_DURING_TAG") == "1" or self._fts_topk <= 0:
            self._skip_fts = True

    # ------------------------------------------------------------------
    # Public API (DBWriteQueue)
    # ------------------------------------------------------------------
    def start(self) -> None:
        self.raise_if_failed()
        if self._started:
            return
        if not self._thread.is_alive():
            self._started = True
            self._thread.start()
            if not self._ready_evt.wait(self._STARTUP_WAIT_TIMEOUT_SECONDS):
                self._log.warning(
                    "DBWritingService: worker startup is still pending after %.1f seconds",
                    self._STARTUP_WAIT_TIMEOUT_SECONDS,
                )
            self.raise_if_failed()

    def raise_if_failed(self) -> None:
        with self._exc_lock:
            exc = self._exc
        if exc:
            raise exc

    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
        self.raise_if_failed()
        self._queue.put(item, block=block, timeout=timeout)
        self.raise_if_failed()

    def qsize(self) -> int:
        try:
            return self._queue.qsize()
        except Exception:
            # Failure policy: queue size is diagnostic-only and must not fail
            # callers while shutdown/error propagation uses raise_if_failed().
            return -1

    def stop(self, *, flush: bool = True, wait_forever: bool = False) -> None:
        self._log.debug("DBWritingService: stop requested (flush=%s, wait_forever=%s)", flush, wait_forever)
        if flush:
            self._put_shutdown_message(DBFlush(), "queue flush sentinel")
        self._put_shutdown_message(DBStop(), "queue stop sentinel", retry_while_alive=wait_forever)
        self._stop_evt.set()

        if wait_forever:
            next_log_at = time.monotonic() + self._SHUTDOWN_LOG_INTERVAL_SECONDS
            while self._thread.is_alive():
                self._thread.join(timeout=1.0)
                now = time.monotonic()
                if now >= next_log_at:
                    self._log.warning(
                        "DBWritingService: still waiting for worker shutdown (qsize=%s, written=%s, flush_count=%s)",
                        self.qsize(),
                        self._written,
                        self._flush_count,
                    )
                    next_log_at = now + self._SHUTDOWN_LOG_INTERVAL_SECONDS
        else:
            try:
                self._thread.join(timeout=10.0)
            except RuntimeError:
                # Failure policy: joining an unstarted/current thread during
                # shutdown is best effort; worker failures are still propagated
                # by raise_if_failed() below.
                pass
        self.raise_if_failed()

    def _put_shutdown_message(self, message: object, operation: str, *, retry_while_alive: bool = False) -> None:
        """Queue shutdown messages while preserving worker failures after join."""

        next_log_at = time.monotonic() + self._SHUTDOWN_LOG_INTERVAL_SECONDS
        while True:
            try:
                self._queue.put(message, timeout=1.0)
                return
            except queue.Full as exc:
                if retry_while_alive and self._thread.is_alive():
                    now = time.monotonic()
                    if now >= next_log_at:
                        self._log.warning(
                            "DBWritingService: still waiting to enqueue %s (qsize=%s, written=%s, flush_count=%s)",
                            operation,
                            self.qsize(),
                            self._written,
                            self._flush_count,
                        )
                        next_log_at = now + self._SHUTDOWN_LOG_INTERVAL_SECONDS
                    else:
                        self._log_best_effort_failure(operation, exc, level=logging.DEBUG)
                    continue
                self._log_best_effort_failure(operation, exc, level=logging.WARNING)
                return
            except Exception as exc:
                self._log_best_effort_failure(operation, exc, level=logging.WARNING)
                return

    @staticmethod
    def _require_positive_int(name: str, value: int) -> int:
        """Return ``value`` as a positive integer or raise a clear error."""

        normalized = int(value)
        if normalized < 1:
            raise ValueError(f"{name} must be >= 1")
        return normalized

    # ------------------------------------------------------------------
    # Thread entry point and helpers
    # ------------------------------------------------------------------
    def _thread_main(self) -> None:
        try:
            conn = self._open_connection()
            try:
                self._apply_pragmas(conn)
                if self._unsafe_fast and self._stage_tags_in_temp:
                    self._create_temp_staging(conn)
                self._ready_evt.set()
                self._process_queue(conn)
                if self._unsafe_fast and self._stage_tags_in_temp:
                    self._merge_staging_into_persistent(conn)
            finally:
                if self._unsafe_fast:
                    try:
                        self._restore_normal_mode(conn)
                    except Exception as exc:  # pragma: no cover - best effort
                        # Failure policy: restoring PRAGMAs is cleanup and must
                        # not mask a primary worker failure.
                        self._log.warning("DBWritingService: restore_normal_mode failed: %s", exc)
                conn.close()
        except BaseException as exc:  # pragma: no cover - defensive catch
            # Failure policy: worker-thread failures are fatal to the service and
            # are stored for caller-side propagation.
            self._record_failure(exc)
            self._ready_evt.set()
            self._log.exception("DBWritingService crashed: %s", exc)
            self._stop_evt.set()

    def _record_failure(self, exc: BaseException) -> None:
        """Store the first worker-side failure for caller-side propagation."""

        with self._exc_lock:
            if self._exc is None:
                self._exc = exc

    def _open_connection(self) -> sqlite3.Connection:
        return get_conn(self._db_path, allow_when_quiesced=True, timeout=120.0)

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        if self._unsafe_fast:
            try:
                self._apply_unsafe_fast_pragmas(conn)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower():
                    self._log.warning("DBWritingService: EXCLUSIVE lock unavailable; falling back to WAL mode")
                    try:
                        self._restore_normal_mode(conn)
                    except Exception as restore_exc:  # pragma: no cover - best effort
                        # Failure policy: fallback cleanup should not mask the
                        # locked-mode fallback path.
                        self._log.warning(
                            "DBWritingService: restore_normal_mode before fallback failed: %s", restore_exc
                        )
                    # フォールバック：通常フローで書き込む
                    self._unsafe_fast = False
                    self._stage_tags_in_temp = False
                    self._apply_wal_pragmas(conn)
                else:
                    raise
        else:
            self._apply_wal_pragmas(conn)

    def _apply_unsafe_fast_pragmas(self, conn: sqlite3.Connection) -> None:
        self._pragmas.apply_unsafe_fast(conn)

    def _apply_wal_pragmas(self, conn: sqlite3.Connection) -> None:
        self._pragmas.apply_wal(conn)

    def _process_queue(self, conn: sqlite3.Connection) -> None:
        batch: List[DBItem] = []
        while True:
            msg = self._poll_queue()
            if msg is None:
                if batch:
                    self._flush_batch(conn, batch)
                    batch.clear()
                if self._stop_evt.is_set():
                    break
                continue
            if isinstance(msg, DBStop):
                if batch:
                    self._flush_batch(conn, batch)
                    batch.clear()
                break
            if isinstance(msg, DBFlush):
                if batch:
                    self._flush_batch(conn, batch)
                    batch.clear()
                continue
            if isinstance(msg, DBItem):
                batch.append(msg)
                if len(batch) >= self._flush_chunk:
                    self._flush_batch(conn, batch)
                    batch.clear()

    def _poll_queue(self) -> object | None:
        try:
            return self._queue.get(timeout=0.5)
        except queue.Empty:
            return None

    def _flush_batch(self, conn: sqlite3.Connection, items: Sequence[DBItem]) -> None:
        if not items:
            return
        if self._unsafe_fast and self._stage_tags_in_temp:
            self._flush_into_temp_tables(conn, items)
        else:
            self._flush_standard(conn, items)
        self._written += len(items)
        if self._debug:
            self._log.debug(
                "DBWritingService: flushed %d items (written=%d, q=%d)",
                len(items),
                self._written,
                self.qsize(),
            )

    def _flush_into_temp_tables(self, conn: sqlite3.Connection, items: Sequence[DBItem]) -> None:
        try:
            self._staging_writer.flush(conn, items, default_tagger_sig=self._default_tagger_sig)
        except Exception:
            # Failure policy: temp batch transaction failures are fatal; rollback
            # is best effort and the original failure is re-raised.
            self._rollback_safely(conn)
            raise

    def _flush_standard(self, conn: sqlite3.Connection, items: Sequence[DBItem]) -> None:
        try:
            tag_cache = self._standard_writer.flush(
                conn,
                items,
                tag_cache=self._tag_cache,
                default_tagger_sig=self._default_tagger_sig,
                skip_fts=self._skip_fts,
                fts_topk=self._fts_topk,
            )
        except Exception:
            # Failure policy: standard batch transaction failures are fatal;
            # rollback is best effort and the original failure is re-raised.
            self._rollback_safely(conn)
            raise
        self._tag_cache = tag_cache
        self._maybe_checkpoint(conn)

    def _upsert_tags_uncommitted(
        self,
        conn: sqlite3.Connection,
        tags,
    ) -> dict[str, int]:
        """Upsert tag definitions without committing the caller's transaction."""

        return upsert_tags_uncommitted(conn, tags)

    def _bulk_update_files_meta_by_id_uncommitted(
        self,
        conn: sqlite3.Connection,
        rows: Sequence[tuple[int | None, int | None, str | None, float | None, int]],
        *,
        coalesce_wh: bool = True,
    ) -> None:
        """Update file metadata without committing the caller's transaction."""

        bulk_update_files_meta_by_id_uncommitted(conn, rows, coalesce_wh=coalesce_wh)

    def _rollback_safely(self, conn: sqlite3.Connection) -> None:
        """Rollback a failed batch while preserving the original exception."""

        try:
            conn.rollback()
        except Exception as exc:  # pragma: no cover - rollback failures depend on SQLite state
            self._log_best_effort_failure("rollback after batch error", exc, level=logging.WARNING)

    def _maybe_checkpoint(self, conn: sqlite3.Connection) -> None:
        wal_size = self._wal_size_mb()
        if wal_size >= 256:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception as exc:
                self._log_best_effort_failure("checkpoint passive after large WAL", exc, level=logging.DEBUG)
            return
        self._flush_count += 1
        if (self._flush_count % 2) == 0:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception as exc:
                self._log_best_effort_failure("checkpoint passive", exc, level=logging.DEBUG)
        if (self._flush_count % 32) == 0 and self._queue.empty():
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA optimize")
            except Exception as exc:
                self._log_best_effort_failure("checkpoint truncate/optimize", exc, level=logging.DEBUG)

    def _wal_size_mb(self) -> int:
        try:
            return os.path.getsize(f"{self._db_path}-wal") // (1024 * 1024)
        except OSError:
            return 0

    def _create_temp_staging(self, conn: sqlite3.Connection) -> None:
        self._staging_writer.create_tables(conn)

    def _merge_staging_into_persistent(self, conn: sqlite3.Connection) -> None:
        merger = StagingMerger(
            self._log,
            self._emit_progress,
            lambda operation, exc: self._log_best_effort_failure(operation, exc, level=logging.DEBUG),
        )
        merger.merge(conn)

    def _emit_progress(self, kind: str, done: int, total: int) -> None:
        cb = self._progress_cb
        if cb:
            try:
                cb(kind, int(done), int(max(total, 1)))
            except Exception as exc:
                self._log_best_effort_failure("progress callback", exc, level=logging.WARNING)

    def _restore_normal_mode(self, conn: sqlite3.Connection) -> None:
        self._pragmas.restore_normal_mode(conn, self._log_best_effort_failure)

    def _log_best_effort_failure(self, operation: str, exc: BaseException, *, level: int) -> None:
        """Log cleanup/maintenance failures that must not mask the primary result."""

        self._log.log(
            level,
            "DBWritingService: best-effort %s failed: %s",
            operation,
            exc,
            exc_info=self._debug,
        )

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------
    def _chunked(self, seq: Sequence[int], size: int) -> List[List[int]]:
        return chunked(seq, size)

    def _chunked_table(self, conn: sqlite3.Connection, table: str, *, step: int) -> List[List[int]]:
        return chunked_table(conn, table, step=step)

    def _rowid_windows(self, conn: sqlite3.Connection, table: str, window: int) -> List[tuple[int, int]]:
        return rowid_windows(conn, table, window)


__all__ = ["DBWritingService"]
