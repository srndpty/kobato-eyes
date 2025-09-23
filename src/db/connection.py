"""Connection helpers for the kobato-eyes SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path


def get_conn(db_path: str | Path, *, timeout: float = 30.0) -> sqlite3.Connection:
    """Create a SQLite connection with WAL and foreign-key support enabled."""
    path = str(db_path)
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    is_memory_db = path == ":memory:" or (path.startswith("file:") and "mode=memory" in path)
    if not is_memory_db:
        conn.execute("PRAGMA journal_mode = WAL;")

    return conn


__all__ = ["get_conn"]
