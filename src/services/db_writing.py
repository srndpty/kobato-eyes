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
from db.files import bulk_update_files_meta_by_id
from db.fts import fts_replace_rows
from db.tags import upsert_tags

logger = logging.getLogger(__name__)


class DBWritingService(DBWriteQueue):
    """Thread-backed queue that persists tagging results in batches."""

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
        self._flush_chunk = int(flush_chunk)
        self._fts_topk = int(fts_topk)
        self._queue: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)
        self._default_tagger_sig = default_tagger_sig
        self._unsafe_fast = bool(unsafe_fast)
        self._skip_fts = bool(skip_fts)
        self._progress_cb = progress_cb
        self._exc: BaseException | None = None
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._thread_main, name="DBWritingService", daemon=True)
        self._written = 0
        self._flush_count = 0
        self._tag_cache: Dict[str, int] = {}
        self._stage_tags_in_temp = True
        self._debug = os.environ.get("KE_DBWRITER_DEBUG") == "1"
        if os.environ.get("KE_SKIP_FTS_DURING_TAG") == "1" or self._fts_topk <= 0:
            self._skip_fts = True

    # ------------------------------------------------------------------
    # Public API (DBWriteQueue)
    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def raise_if_failed(self) -> None:
        if self._exc:
            raise self._exc

    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def qsize(self) -> int:
        try:
            return self._queue.qsize()
        except Exception:
            return -1

    def stop(self, *, flush: bool = True, wait_forever: bool = False) -> None:
        self._log.debug("DBWritingService: stop requested (flush=%s, wait_forever=%s)", flush, wait_forever)
        try:
            if flush:
                self._queue.put(DBFlush())
            self._queue.put(DBStop())
        except Exception:
            pass
        self._stop_evt.set()

        if wait_forever:
            while self._thread.is_alive():
                self._thread.join(timeout=1.0)
        else:
            try:
                self._thread.join(timeout=10.0)
            except RuntimeError:
                pass
        self.raise_if_failed()

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
                self._process_queue(conn)
                if self._unsafe_fast and self._stage_tags_in_temp:
                    self._merge_staging_into_persistent(conn)
            finally:
                if self._unsafe_fast:
                    try:
                        self._restore_normal_mode(conn)
                    except Exception as exc:  # pragma: no cover - best effort
                        self._log.warning("DBWritingService: restore_normal_mode failed: %s", exc)
                conn.close()
        except BaseException as exc:  # pragma: no cover - defensive catch
            self._exc = exc
            self._log.exception("DBWritingService crashed: %s", exc)
            self._stop_evt.set()

    def _open_connection(self) -> sqlite3.Connection:
        return get_conn(self._db_path, allow_when_quiesced=True, timeout=120.0)

    def _apply_pragmas(self, conn: sqlite3.Connection) -> None:
        if self._unsafe_fast:
            try:
                self._apply_unsafe_fast_pragmas(conn)
            except sqlite3.OperationalError as exc:
                if "locked" in str(exc).lower():
                    self._log.warning("DBWritingService: EXCLUSIVE lock unavailable; falling back to WAL mode")
                    # フォールバック：通常フローで書き込む
                    self._unsafe_fast = False
                    self._stage_tags_in_temp = False
                    self._apply_wal_pragmas(conn)
                else:
                    raise
        else:
            self._apply_wal_pragmas(conn)

    def _apply_unsafe_fast_pragmas(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-262144")
        try:
            conn.execute("PRAGMA mmap_size=268435456")
        except Exception as exc:  # pragma: no cover - depends on environment
            self._log.warning("DBWritingService: pragma mmap_size failed: %s", exc)
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("COMMIT")
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as exc:  # pragma: no cover - defensive
            self._log.warning("DBWritingService: wal_checkpoint failed: %s", exc)
        for _ in range(5):
            try:
                conn.execute("PRAGMA journal_mode=MEMORY")
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                time.sleep(0.25)
        conn.execute("PRAGMA synchronous=OFF")

    def _apply_wal_pragmas(self, conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA cache_size=-262144")

    def _process_queue(self, conn: sqlite3.Connection) -> None:
        batch: List[DBItem] = []
        while True:
            msg = self._poll_queue()
            if msg is None:
                if batch:
                    self._flush_batch(conn, batch)
                    batch.clear()
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
        conn.execute("BEGIN")
        now = time.time()
        defs_set: set[tuple[str, int]] = set()
        for it in items:
            for name, _score, category in it.tags:
                defs_set.add((name, int(category)))
        if defs_set:
            conn.executemany(
                "INSERT OR IGNORE INTO temp.tmp_tag_defs(name, category) VALUES(?, ?)",
                list(defs_set),
            )
        tag_rows: list[tuple[int, str, float, int]] = []
        for it in items:
            for name, score, category in it.tags:
                tag_rows.append((it.file_id, name, float(score), int(category)))
        if tag_rows:
            conn.executemany(
                "INSERT INTO temp.tmp_file_tags(file_id, tag_name, score, category) VALUES(?, ?, ?, ?)",
                tag_rows,
            )
        metas: list[tuple[int, int | None, int | None, str, float]] = []
        for it in items:
            sig = it.tagger_sig or self._default_tagger_sig
            ts = it.tagged_at or now
            metas.append((it.file_id, it.width, it.height, sig, ts))
        if metas:
            conn.executemany(
                """
                INSERT INTO temp.tmp_files_meta(file_id, width, height, tagger_sig, tagged_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                  width      = COALESCE(excluded.width,  width),
                  height     = COALESCE(excluded.height, height),
                  tagger_sig = excluded.tagger_sig,
                  tagged_at  = excluded.tagged_at
                """,
                metas,
            )
        conn.commit()

    def _flush_standard(self, conn: sqlite3.Connection, items: Sequence[DBItem]) -> None:
        conn.execute("BEGIN IMMEDIATE")
        new_defs: list[dict[str, object]] = []
        for it in items:
            for name, _score, category in it.tags:
                if name not in self._tag_cache:
                    new_defs.append({"name": name, "category": int(category)})
        if new_defs:
            created = upsert_tags(conn, new_defs)
            self._tag_cache.update(created)
        file_ids = [it.file_id for it in items]
        for chunk in self._chunked(file_ids, 900):
            if not chunk:
                continue
            placeholders = ",".join(["?"] * len(chunk))
            conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({placeholders})", chunk)
        tag_rows: list[tuple[int, int, float]] = []
        for it in items:
            for name, score, _category in it.tags:
                tag_id = self._tag_cache.get(name)
                if tag_id is not None:
                    tag_rows.append((it.file_id, int(tag_id), float(score)))
        if tag_rows:
            conn.executemany(
                "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
                tag_rows,
            )
        if not self._skip_fts:
            fts_rows: list[tuple[int, str]] = []
            for it in items:
                top = sorted(it.tags, key=lambda t: t[1], reverse=True)[: self._fts_topk]
                text = " ".join([name for (name, _score, _category) in top])
                if text:
                    fts_rows.append((it.file_id, text))
            if fts_rows:
                fts_replace_rows(conn, fts_rows)
        now = time.time()
        meta_rows: list[tuple[int | None, int | None, str | None, float | None, int]] = []
        for it in items:
            sig = it.tagger_sig or self._default_tagger_sig
            ts = it.tagged_at or now
            meta_rows.append((it.width, it.height, sig, ts, it.file_id))
        bulk_update_files_meta_by_id(conn, meta_rows, coalesce_wh=True)
        conn.commit()
        self._maybe_checkpoint(conn)

    def _maybe_checkpoint(self, conn: sqlite3.Connection) -> None:
        wal_size = self._wal_size_mb()
        if wal_size >= 256:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
            return
        self._flush_count += 1
        if (self._flush_count % 2) == 0:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
        if (self._flush_count % 32) == 0 and self._queue.empty():
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA optimize")
            except Exception:
                pass

    def _wal_size_mb(self) -> int:
        try:
            return os.path.getsize(f"{self._db_path}-wal") // (1024 * 1024)
        except OSError:
            return 0

    def _create_temp_staging(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS tmp_file_tags(
                file_id  INTEGER,
                tag_name TEXT,
                score    REAL,
                category INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS tmp_files_meta(
                file_id    INTEGER PRIMARY KEY,
                width      INTEGER,
                height     INTEGER,
                tagger_sig TEXT,
                tagged_at  REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS tmp_tag_defs(
                name     TEXT PRIMARY KEY,
                category INTEGER
            )
            """
        )

    def _merge_staging_into_persistent(self, conn: sqlite3.Connection) -> None:
        try:
            row = conn.execute("SELECT count(*) FROM temp.tmp_file_tags").fetchone()
        except sqlite3.OperationalError:
            return
        if not row or int(row[0]) == 0:
            return
        total_tags = int(row[0])
        total_files = int(conn.execute("SELECT count(DISTINCT file_id) FROM temp.tmp_file_tags").fetchone()[0])
        total_meta = int(conn.execute("SELECT count(*) FROM temp.tmp_files_meta").fetchone()[0])
        self._log.info("DBWritingService: offline merge start (tmp->disk)")
        self._emit_progress("merge.start", 0, total_tags)
        try:
            conn.execute("DROP INDEX IF EXISTS idx_file_tags_tag_score")
            conn.execute("DROP INDEX IF EXISTS idx_file_tags_tag_id")
        except Exception as exc:
            self._log.warning("DBWritingService: drop index failed: %s", exc)
        conn.execute("BEGIN IMMEDIATE")
        try:
            rows = conn.execute("SELECT name, MAX(category) FROM temp.tmp_tag_defs GROUP BY name").fetchall()
            if rows:
                defs = [{"name": r[0], "category": int(r[1] or 0)} for r in rows]
                upsert_tags(conn, defs)
            conn.execute("DROP TABLE IF EXISTS temp.tmp_file_ids")
            conn.execute("CREATE TABLE temp.tmp_file_ids AS SELECT DISTINCT file_id FROM temp.tmp_file_tags")
            self._emit_progress("merge.delete", 0, total_files)
            done = 0
            for chunk in self._chunked_table(conn, "temp.tmp_file_ids", step=5000):
                placeholders = ",".join(["?"] * len(chunk))
                conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({placeholders})", chunk)
                done += len(chunk)
                self._emit_progress("merge.delete", done, total_files)
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_ft_name ON tmp_file_tags(tag_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_ft_file ON tmp_file_tags(file_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_fm_file ON tmp_files_meta(file_id)")
            self._emit_progress("merge.insert", 0, total_tags)
            done = 0
            for start, end in self._rowid_windows(conn, "temp.tmp_file_tags", 100_000):
                conn.execute(
                    """
                    INSERT INTO file_tags (file_id, tag_id, score)
                    SELECT ft.file_id, t.id, ft.score
                    FROM temp.tmp_file_tags AS ft
                    JOIN tags AS t ON t.name = ft.tag_name
                    WHERE ft.rowid > ? AND ft.rowid <= ?
                    """,
                    (start, end),
                )
                done = min(done + 100_000, total_tags)
                self._emit_progress("merge.insert", done, total_tags)
            self._emit_progress("merge.update", 0, total_meta)
            done = 0
            for start, end in self._rowid_windows(conn, "temp.tmp_files_meta", 20_000):
                conn.execute(
                    """
                    UPDATE files
                    SET width = COALESCE((SELECT m.width FROM temp.tmp_files_meta m WHERE m.file_id = files.id AND m.rowid > ? AND m.rowid <= ?), width),
                        height = COALESCE((SELECT m.height FROM temp.tmp_files_meta m WHERE m.file_id = files.id AND m.rowid > ? AND m.rowid <= ?), height),
                        tagger_sig = (SELECT m.tagger_sig FROM temp.tmp_files_meta m WHERE m.file_id = files.id AND m.rowid > ? AND m.rowid <= ?),
                        last_tagged_at = (SELECT m.tagged_at FROM temp.tmp_files_meta m WHERE m.file_id = files.id AND m.rowid > ? AND m.rowid <= ?)
                    WHERE id IN (SELECT file_id FROM temp.tmp_files_meta WHERE rowid > ? AND rowid <= ?)
                    """,
                    (start, end, start, end, start, end, start, end, start, end),
                )
                done = min(done + 20_000, total_meta)
                self._emit_progress("merge.update", done, total_meta)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            for ddl in (
                "DROP INDEX IF EXISTS temp.idx_tmp_ft_name",
                "DROP INDEX IF EXISTS temp.idx_tmp_ft_file",
                "DROP INDEX IF EXISTS temp.idx_tmp_fm_file",
                "DROP TABLE IF EXISTS temp.tmp_file_ids",
            ):
                try:
                    conn.execute(ddl)
                except Exception:
                    pass
        try:
            self._emit_progress("merge.index", 0, 2)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id ON file_tags(tag_id)")
            self._emit_progress("merge.index", 1, 2)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_tag_score ON file_tags(tag_id, score)")
            self._emit_progress("merge.index", 2, 2)
            conn.commit()
        except Exception as exc:
            self._log.warning("DBWritingService: recreate index failed: %s", exc)
        self._emit_progress("merge.done", 1, 1)
        self._log.info("DBWritingService: offline merge done")

    def _emit_progress(self, kind: str, done: int, total: int) -> None:
        cb = self._progress_cb
        if cb:
            try:
                cb(kind, int(done), int(max(total, 1)))
            except Exception:
                pass

    def _restore_normal_mode(self, conn: sqlite3.Connection) -> None:
        for statement in (
            "END",
            "PRAGMA locking_mode=NORMAL",
            "PRAGMA journal_mode=DELETE",
            "PRAGMA journal_mode=WAL",
            "PRAGMA wal_checkpoint(TRUNCATE)",
            "PRAGMA synchronous=NORMAL",
        ):
            try:
                conn.execute(statement)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------
    def _chunked(self, seq: Sequence[int], size: int) -> List[List[int]]:
        return [list(seq[i : i + size]) for i in range(0, len(seq), size)]

    def _chunked_table(self, conn: sqlite3.Connection, table: str, *, step: int) -> List[List[int]]:
        rows = conn.execute(f"SELECT file_id FROM {table}").fetchall()
        result: List[List[int]] = []
        buf: List[int] = []
        for (fid,) in rows:
            buf.append(int(fid))
            if len(buf) >= step:
                result.append(buf[:])
                buf.clear()
        if buf:
            result.append(buf)
        return result

    def _rowid_windows(self, conn: sqlite3.Connection, table: str, window: int) -> List[tuple[int, int]]:
        max_rowid = conn.execute(f"SELECT max(rowid) FROM {table}").fetchone()[0] or 0
        start = 0
        windows: List[tuple[int, int]] = []
        while start < max_rowid:
            end = min(start + window, max_rowid)
            windows.append((start, end))
            start = end
        return windows


__all__ = ["DBWritingService"]
