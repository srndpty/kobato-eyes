"""Tests for maintenance utilities."""
from __future__ import annotations

import importlib.util
import logging
import sqlite3
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "core" / "pipeline" / "maintenance.py"

spec = importlib.util.spec_from_file_location("core.pipeline.maintenance", MODULE_PATH)
maintenance = importlib.util.module_from_spec(spec)
sys.modules.setdefault("core.pipeline.maintenance", maintenance)
assert spec.loader is not None
spec.loader.exec_module(maintenance)

wait_for_unlock = maintenance.wait_for_unlock
_settle_after_quiesce = maintenance._settle_after_quiesce


class DummyConnection:
    """Utility connection for tests."""

    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query):
        return self._cursor.execute(query)

    def cursor(self):
        return self._cursor

    def commit(self):
        return None


class DummyCursor:
    """Utility cursor for tests."""

    def __init__(self):
        self.calls = []

    def execute(self, query):
        self.calls.append(query)
        if "wal_checkpoint" in query.lower():
            raise sqlite3.OperationalError("wal failure")
        if "optimize" in query.lower():
            raise sqlite3.OperationalError("opt failure")
        return None


def test_wait_for_unlock_retries_on_locked(monkeypatch):
    calls = []
    sleep_calls = []

    def fake_connect(db_path, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise sqlite3.OperationalError("database is locked")
        return DummyConnection(DummyCursor())

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(maintenance.sqlite3, "connect", fake_connect)
    monkeypatch.setattr(maintenance.time, "sleep", fake_sleep)

    assert wait_for_unlock("dummy.db", timeout=1.0) is True
    assert len(calls) == 3
    assert sleep_calls == [0.25, 0.25]


def test_wait_for_unlock_returns_true_on_non_locked_error(monkeypatch):
    calls = []
    sleep_calls = []

    def fake_connect(db_path, timeout):
        calls.append(timeout)
        raise sqlite3.OperationalError("disk I/O error")

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(maintenance.sqlite3, "connect", fake_connect)
    monkeypatch.setattr(maintenance.time, "sleep", fake_sleep)

    assert wait_for_unlock("dummy.db", timeout=1.0) is True
    assert len(calls) == 1
    assert sleep_calls == []


def test_wait_for_unlock_returns_true_on_success(monkeypatch):
    calls = []
    sleep_calls = []

    def fake_connect(db_path, timeout):
        calls.append(timeout)
        return DummyConnection(DummyCursor())

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(maintenance.sqlite3, "connect", fake_connect)
    monkeypatch.setattr(maintenance.time, "sleep", fake_sleep)

    assert wait_for_unlock("dummy.db", timeout=1.0) is True
    assert len(calls) == 1
    assert sleep_calls == []


def test_settle_after_quiesce_logs_warnings(monkeypatch, caplog):
    sleep_calls = []

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    def fake_wait_for_unlock(db_path, timeout=15.0):
        return True

    cursor = DummyCursor()

    def fake_connect(db_path, timeout):
        return DummyConnection(cursor)

    monkeypatch.setattr(maintenance.time, "sleep", fake_sleep)
    monkeypatch.setattr(maintenance, "wait_for_unlock", fake_wait_for_unlock)
    monkeypatch.setattr(maintenance.sqlite3, "connect", fake_connect)

    caplog.set_level(logging.WARNING)

    _settle_after_quiesce("dummy.db")

    warnings = {record.message for record in caplog.records}
    assert any("wal_checkpoint failed" in message for message in warnings)
    assert any("optimize failed" in message for message in warnings)
    assert sleep_calls == [0.2]
