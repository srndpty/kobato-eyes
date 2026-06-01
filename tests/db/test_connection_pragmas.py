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


def test_ensure_indexes_logs_warning_and_reraises_on_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_ensure_indexes が OperationalError を WARNING ログに残してから re-raise する。"""
    import logging

    import db.connection as conn_module

    # sqlite3.Connection は C 拡張型なのでパッチできないため、ラッパーを使う
    fail_once = {"done": False}
    real_conn = sqlite3.connect(":memory:")
    from db.schema import apply_schema

    apply_schema(real_conn)

    class _FailingConn:
        """CREATE INDEX SQL の最初の実行だけ OperationalError を raise するラッパー。"""

        def execute(self, sql: str, *args, **kwargs):
            if not fail_once["done"] and "CREATE" in sql and "INDEX" in sql:
                fail_once["done"] = True
                raise sqlite3.OperationalError("disk I/O error (injected)")
            return real_conn.execute(sql, *args, **kwargs)

        def commit(self):
            return real_conn.commit()

        def close(self):
            return real_conn.close()

    with caplog.at_level(logging.WARNING, logger="db.connection"):
        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            conn_module._ensure_indexes(_FailingConn())  # type: ignore[arg-type]

    assert any("FAILED" in r.message for r in caplog.records)
    real_conn.close()


def test_ensure_indexes_skips_heavy_when_env_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """KE_SKIP_HEAVY_INDEXES=1 のとき idx_file_tags_tag_score がスキップされる。"""
    import logging

    import db.connection as conn_module

    monkeypatch.setenv("KE_SKIP_HEAVY_INDEXES", "1")

    conn = sqlite3.connect(":memory:")
    from db.schema import apply_schema

    apply_schema(conn)

    with caplog.at_level(logging.WARNING, logger="db.connection"):
        conn_module._ensure_indexes(conn)

    skip_msgs = [r.message for r in caplog.records if "idx_file_tags_tag_score" in r.message]
    assert any("skipped" in m for m in skip_msgs)
    conn.close()
