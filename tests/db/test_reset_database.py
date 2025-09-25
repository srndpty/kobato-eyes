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

    result = reset_database(db_path, backup=False, purge_hnsw=False)
    assert Path(result["db"]) == db_path

    fresh = get_conn(db_path)
    try:
        for table in ["files", "tags", "file_tags", "signatures", "embeddings"]:
            assert _table_count(fresh, table) == 0
        version = fresh.execute("PRAGMA user_version").fetchone()[0]
    finally:
        fresh.close()

    assert int(version) == CURRENT_SCHEMA_VERSION


def test_reset_with_backup_and_hnsw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "kobato-eyes.db"
    conn = get_conn(db_path)
    try:
        conn.execute("INSERT INTO tags (name) VALUES ('existing')")
        conn.commit()
    finally:
        conn.close()

    wal_path = db_path.with_name(f"{db_path.name}-wal")
    wal_path.write_bytes(b"wal")
    shm_path = db_path.with_name(f"{db_path.name}-shm")
    shm_path.write_bytes(b"shm")

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    hnsw_path = index_dir / "hnsw_cosine.bin"
    hnsw_path.write_bytes(b"index")

    monkeypatch.setattr("db.admin.get_index_dir", lambda: index_dir)

    result = reset_database(db_path, backup=True, purge_hnsw=True)

    backups = result["backup_paths"]
    assert isinstance(backups, list)
    assert len(backups) == 3
    for backup in backups:
        assert isinstance(backup, Path)
        assert backup.exists()

    expected_prefixes = {f"{p.name}.bak-" for p in (db_path, wal_path, shm_path)}
    matched_prefixes = {
        prefix
        for path in backups
        for prefix in expected_prefixes
        if path.name.startswith(prefix)
    }
    assert matched_prefixes == expected_prefixes

    assert result["hnsw_deleted"] is True
    assert not hnsw_path.exists()

    fresh = get_conn(db_path)
    try:
        count = _table_count(fresh, "tags")
    finally:
        fresh.close()

    assert count == 0
