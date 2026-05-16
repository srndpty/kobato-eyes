"""Tests for FTS persistence helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import pytest

import db.fts as fts_module
from db.fts import fts_delete_rows, fts_replace_rows, update_fts_bulk


class _RecordingConnection:
    def __init__(self, *, fail_delete_command: bool = False, fail_delete_fallback: bool = False) -> None:
        self.fail_delete_command = fail_delete_command
        self.fail_delete_fallback = fail_delete_fallback
        self.executed: list[tuple[str, list[object]]] = []
        self.executemany_calls: list[tuple[str, list[tuple[int, str]]]] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, sql: str, params: Sequence[object] | None = None):
        values = list(params or [])
        self.executed.append((sql, values))
        if "INSERT INTO fts_files(fts_files, rowid)" in sql and self.fail_delete_command:
            raise sqlite3.DatabaseError("delete command failed")
        if "DELETE FROM fts_files" in sql and self.fail_delete_fallback:
            raise sqlite3.DatabaseError("delete fallback failed")
        return None

    def executemany(self, sql: str, rows) -> None:
        self.executemany_calls.append((sql, list(rows)))

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


def test_fts_delete_rows_uses_contentless_delete_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fts_module, "fts_is_contentless", lambda conn: True)
    conn = _RecordingConnection()

    fts_delete_rows(conn, [1, "2"])  # type: ignore[arg-type]

    assert conn.executed == [("INSERT INTO fts_files(fts_files, rowid) VALUES ('delete', ?),('delete', ?)", [1, 2])]


def test_fts_delete_rows_falls_back_to_delete_for_contentless_table(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fts_module, "fts_is_contentless", lambda conn: True)
    conn = _RecordingConnection(fail_delete_command=True)

    fts_delete_rows(conn, [1, 2])  # type: ignore[arg-type]

    assert conn.executed == [
        ("INSERT INTO fts_files(fts_files, rowid) VALUES ('delete', ?),('delete', ?)", [1, 2]),
        ("DELETE FROM fts_files WHERE rowid IN (?,?)", [1, 2]),
    ]


def test_fts_replace_rows_falls_back_and_still_inserts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fts_module, "fts_is_contentless", lambda conn: True)
    conn = _RecordingConnection(fail_delete_command=True, fail_delete_fallback=True)

    fts_replace_rows(conn, [(1, "alpha"), (2, "beta")])  # type: ignore[arg-type]

    assert conn.executed == [
        ("INSERT INTO fts_files(fts_files, rowid) VALUES ('delete', ?),('delete', ?)", [1, 2]),
        ("DELETE FROM fts_files WHERE rowid IN (?,?)", [1, 2]),
        ("INSERT INTO fts_files(rowid, text) VALUES (?, ?),(?, ?)", [1, "alpha", 2, "beta"]),
    ]


def test_update_fts_bulk_deletes_all_ids_and_inserts_non_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fts_module, "fts_is_contentless", lambda conn: False)
    conn = _RecordingConnection()

    update_fts_bulk(conn, [(1, "alpha"), (2, None), (3, "")])  # type: ignore[arg-type]

    assert conn.executed == [("DELETE FROM fts_files WHERE rowid IN (?,?,?)", [1, 2, 3])]
    assert conn.executemany_calls == [("INSERT INTO fts_files (rowid, text) VALUES (?, ?)", [(1, "alpha")])]
    assert conn.commits == 1
