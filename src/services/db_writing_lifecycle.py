"""SQLite lifecycle helpers for :mod:`services.db_writing`."""

from __future__ import annotations

import logging
import sqlite3
import time


class DBWritingPragmas:
    """Apply SQLite PRAGMAs for standard and unsafe-fast writer modes."""

    def __init__(self, logger: logging.Logger) -> None:
        self._log = logger

    def apply_wal(self, conn: sqlite3.Connection) -> None:
        """Apply standard WAL writer PRAGMAs."""

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA cache_size=-262144")

    def apply_unsafe_fast(self, conn: sqlite3.Connection) -> None:
        """Apply unsafe-fast PRAGMAs before staging writes."""

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

    def restore_normal_mode(self, conn: sqlite3.Connection, log_best_effort_failure) -> None:
        """Restore normal WAL mode after unsafe-fast processing."""

        for statement, level in (
            ("END", logging.DEBUG),
            ("PRAGMA locking_mode=NORMAL", logging.WARNING),
            ("PRAGMA journal_mode=DELETE", logging.WARNING),
            ("PRAGMA journal_mode=WAL", logging.WARNING),
            ("PRAGMA wal_checkpoint(TRUNCATE)", logging.DEBUG),
            ("PRAGMA synchronous=NORMAL", logging.WARNING),
        ):
            try:
                conn.execute(statement)
            except Exception as exc:
                log_best_effort_failure(f"restore normal mode: {statement}", exc, level=level)


__all__ = ["DBWritingPragmas"]
