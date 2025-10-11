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


def test_reset_database_with_backups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "library.db"
    wal_path = db_path.with_name("library.db-wal")
    shm_path = db_path.with_name("library.db-shm")

    for path in (db_path, wal_path, shm_path):
        path.write_bytes(b"dummy")

    resolved_db_path = db_path.resolve()
    backup_suffix = "fixed-suffix"
    monkeypatch.setattr("db.admin._format_backup_suffix", lambda: backup_suffix)

    bootstrap_calls: list[Path] = []

    def _bootstrap(path: Path) -> None:
        bootstrap_calls.append(path)

    monkeypatch.setattr("db.connection.bootstrap_if_needed", _bootstrap)

    result = reset_database(db_path)

    expected_backups = [
        db_path.with_name(f"{db_path.name}.{backup_suffix}"),
        wal_path.with_name(f"{wal_path.name}.{backup_suffix}"),
        shm_path.with_name(f"{shm_path.name}.{backup_suffix}"),
    ]

    assert result["db"] == resolved_db_path
    assert result["backup_paths"] == expected_backups

    for backup_path in expected_backups:
        assert backup_path.exists()

    for original in (db_path, wal_path, shm_path):
        assert not original.exists()

    assert bootstrap_calls == [resolved_db_path]


def test_reset_database_backup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "library.db"
    wal_path = db_path.with_name("library.db-wal")
    shm_path = db_path.with_name("library.db-shm")

    for path in (db_path, wal_path, shm_path):
        path.write_bytes(b"dummy")

    error = OSError("no space left on device")

    def _copy_backup(_path: Path, _suffix: str) -> Path:
        raise error

    bootstrap_calls: list[Path] = []

    def _bootstrap(path: Path) -> None:
        bootstrap_calls.append(path)

    monkeypatch.setattr("db.admin._copy_backup", _copy_backup)
    monkeypatch.setattr("db.connection.bootstrap_if_needed", _bootstrap)

    with pytest.raises(OSError) as excinfo:
        reset_database(db_path)

    assert excinfo.value is error
    assert bootstrap_calls == []

    for path in (db_path, wal_path, shm_path):
        assert path.exists()
