"""Regression tests for SQLite connection safety pragmas."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from db.connection import get_conn


def test_file_connection_enables_wal_and_foreign_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "library.db"

    conn = get_conn(db_path)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        wal_autocheckpoint = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
    finally:
        conn.close()

    assert str(journal_mode).lower() == "wal"
    assert int(foreign_keys) == 1
    assert int(busy_timeout) == 30_000
    assert int(wal_autocheckpoint) == 50_000


def test_connection_timeout_is_applied_to_sqlite_busy_timeout(tmp_path: Path) -> None:
    db_path = tmp_path / "custom-timeout.db"

    conn = get_conn(db_path, timeout=2.5)
    try:
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        conn.close()

    assert int(busy_timeout) == 2_500


def test_foreign_key_violation_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "library.db"

    conn = get_conn(db_path)
    try:
        conn.execute("INSERT INTO tags(name, category) VALUES (?, ?)", ("orphan", 0))
        tag_id = int(conn.execute("SELECT id FROM tags WHERE name = ?", ("orphan",)).fetchone()[0])

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO file_tags(file_id, tag_id, score) VALUES (?, ?, ?)", (999_999, tag_id, 1.0))
    finally:
        conn.close()
