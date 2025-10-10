"""Tests for :mod:`db.admin` database reset helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from db.admin import reset_database
from db.connection import get_conn
from db.repository import replace_file_tags, upsert_file, upsert_tags
from db.schema import CURRENT_SCHEMA_VERSION

pytestmark = pytest.mark.not_gui


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
    try:
        value = cursor.fetchone()[0]
    finally:
        cursor.close()
    return int(value)


def test_reset_creates_empty_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "library.db"
    conn = get_conn(db_path)
    try:
        file_id = upsert_file(
            conn,
            path="/images/sample.png",
            size=1024,
            mtime=123.45,
            sha256="deadbeef",
        )
        tags = upsert_tags(
            conn,
            [
                {"name": "tag:one", "category": 0},
                {"name": "tag:two", "category": 1},
            ],
        )
        replace_file_tags(
            conn,
            file_id,
            [(tags["tag:one"], 0.9), (tags["tag:two"], 0.6)],
        )
        conn.commit()
    finally:
        conn.close()

    result = reset_database(db_path, backup=False)
    assert Path(result["db"]) == db_path

    fresh = get_conn(db_path)
    try:
        for table in ["files", "tags", "file_tags", "signatures"]:
            assert _table_count(fresh, table) == 0
        version = fresh.execute("PRAGMA user_version").fetchone()[0]
    finally:
        fresh.close()

    assert int(version) == CURRENT_SCHEMA_VERSION
