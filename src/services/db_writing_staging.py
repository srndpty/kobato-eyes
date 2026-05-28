"""Unsafe-fast temporary staging writer for :mod:`services.db_writing`."""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Sequence

from core.pipeline.contracts import DBItem
from services.db_writing_standard import upsert_tags_uncommitted


def chunked_table(conn: sqlite3.Connection, table: str, *, step: int) -> list[list[int]]:
    """Read ``file_id`` values from ``table`` in chunks."""

    rows = conn.execute(f"SELECT file_id FROM {table}").fetchall()
    result: list[list[int]] = []
    buf: list[int] = []
    for (fid,) in rows:
        buf.append(int(fid))
        if len(buf) >= step:
            result.append(buf[:])
            buf.clear()
    if buf:
        result.append(buf)
    return result


def rowid_windows(conn: sqlite3.Connection, table: str, window: int) -> list[tuple[int, int]]:
    """Return rowid windows for chunked SQL operations."""

    max_rowid = conn.execute(f"SELECT max(rowid) FROM {table}").fetchone()[0] or 0
    start = 0
    windows: list[tuple[int, int]] = []
    while start < max_rowid:
        end = min(start + window, max_rowid)
        windows.append((start, end))
        start = end
    return windows


class StagingBatchWriter:
    """Write batches into temporary staging tables."""

    def create_tables(self, conn: sqlite3.Connection) -> None:
        """Create temporary staging tables."""

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

    def flush(
        self,
        conn: sqlite3.Connection,
        items: Sequence[DBItem],
        *,
        default_tagger_sig: str | None,
    ) -> None:
        """Flush ``items`` into temporary staging tables."""

        conn.execute("BEGIN")
        try:
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
            metas: list[tuple[int, int | None, int | None, str | None, float]] = []
            for it in items:
                sig = it.tagger_sig or default_tagger_sig
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
        except Exception:
            conn.rollback()
            raise


class StagingMerger:
    """Merge temporary staging tables into persistent tables."""

    def __init__(
        self,
        logger: logging.Logger,
        emit_progress: Callable[[str, int, int], None],
        log_best_effort_failure: Callable[[str, BaseException], None],
    ) -> None:
        self._log = logger
        self._emit_progress = emit_progress
        self._log_best_effort_failure = log_best_effort_failure

    def merge(self, conn: sqlite3.Connection) -> None:
        """Merge temporary staging rows into persistent tables."""

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
                upsert_tags_uncommitted(conn, defs)
            conn.execute("DROP TABLE IF EXISTS temp.tmp_file_ids")
            conn.execute("CREATE TABLE temp.tmp_file_ids AS SELECT DISTINCT file_id FROM temp.tmp_file_tags")
            self._emit_progress("merge.delete", 0, total_files)
            done = 0
            for chunk in chunked_table(conn, "temp.tmp_file_ids", step=5000):
                placeholders = ",".join(["?"] * len(chunk))
                conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({placeholders})", chunk)
                done += len(chunk)
                self._emit_progress("merge.delete", done, total_files)
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_ft_name ON tmp_file_tags(tag_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_ft_file ON tmp_file_tags(file_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_fm_file ON tmp_files_meta(file_id)")
            self._emit_progress("merge.insert", 0, total_tags)
            done = 0
            for start, end in rowid_windows(conn, "temp.tmp_file_tags", 100_000):
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
            for start, end in rowid_windows(conn, "temp.tmp_files_meta", 20_000):
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
                except Exception as exc:
                    self._log_best_effort_failure(f"drop temporary merge object: {ddl}", exc)
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


__all__ = ["StagingBatchWriter", "StagingMerger", "chunked_table", "rowid_windows"]
