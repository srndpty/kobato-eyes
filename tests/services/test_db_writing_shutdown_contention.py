"""DB writer shutdown tests under real SQLite lock contention."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from core.pipeline.contracts import DBItem
from db.connection import get_conn
from db.repository import upsert_file
from db.schema import apply_schema
from services.db_writing import DBWritingService

pytestmark = pytest.mark.db_stress


class _ObservedDBWritingService(DBWritingService):
    """DB writer that signals when SQLite executes ``BEGIN IMMEDIATE``."""

    def __init__(
        self,
        db_path: str,
        transaction_attempted: threading.Event,
        *,
        flush_chunk: int,
        fts_topk: int,
    ) -> None:
        super().__init__(db_path, flush_chunk=flush_chunk, fts_topk=fts_topk)
        self._transaction_attempted = transaction_attempted

    def _open_connection(self) -> sqlite3.Connection:
        conn = super()._open_connection()

        def observe_statement(statement: str) -> None:
            normalized = statement.strip().rstrip(";").upper()
            if normalized == "BEGIN IMMEDIATE":
                self._transaction_attempted.set()

        conn.set_trace_callback(observe_statement)
        return conn


def _prepare_db(db_path: Path, file_count: int) -> list[int]:
    """Create a real database and return identifiers for queued test files."""

    conn = get_conn(db_path)
    try:
        apply_schema(conn)
        return [upsert_file(conn, path=f"C:/images/contention-{index}.png") for index in range(file_count)]
    finally:
        conn.close()


def test_stop_flush_waits_for_lock_then_persists_all_rows(tmp_path: Path) -> None:
    """A locked flush completes after release without dropping queued writes."""

    db_path = tmp_path / "shutdown-contention.db"
    file_ids = _prepare_db(db_path, file_count=3)
    transaction_attempted = threading.Event()
    service = _ObservedDBWritingService(str(db_path), transaction_attempted, flush_chunk=16, fts_topk=0)
    service.start()

    lock_conn = sqlite3.connect(db_path, timeout=1.0)
    lock_conn.execute("BEGIN IMMEDIATE")
    transaction_attempted.clear()
    for file_id in file_ids:
        service.put(DBItem(file_id, [("artist:kobato", 0.9, 1)], 64, 48, "sig:v1", 1234.5))

    stop_started = threading.Event()
    stop_finished = threading.Event()
    stop_errors: list[Exception] = []

    def stop_service() -> None:
        stop_started.set()
        try:
            service.stop(flush=True, wait_forever=True)
        except Exception as exc:  # pragma: no cover - assertion re-raises worker failures
            stop_errors.append(exc)
        finally:
            stop_finished.set()

    # The DB writer is also a daemon. Keeping this coordinator daemonized ensures
    # a writer deadlock fails the assertion instead of holding pytest open.
    stop_thread = threading.Thread(target=stop_service, name="DBWriterStopTest", daemon=True)
    stop_thread_started = False
    stop_started_in_time = False
    transaction_attempted_in_time = False
    stop_was_blocked = False
    stop_finished_in_time = False
    try:
        stop_thread.start()
        stop_thread_started = True
        stop_started_in_time = stop_started.wait(timeout=2.0)
        transaction_attempted_in_time = transaction_attempted.wait(timeout=2.0)
        stop_was_blocked = not stop_finished.is_set()
    finally:
        lock_conn.rollback()
        lock_conn.close()
        if stop_thread_started:
            stop_finished_in_time = stop_finished.wait(timeout=10.0)
            stop_thread.join(timeout=1.0)

    assert stop_started_in_time
    assert transaction_attempted_in_time
    assert stop_was_blocked
    assert stop_finished_in_time
    assert not stop_thread.is_alive()
    if stop_errors:
        raise stop_errors[0]
    assert not service.is_running

    conn = get_conn(db_path, timeout=2.0)
    try:
        placeholders = ", ".join("?" for _ in file_ids)
        stored = conn.execute(
            f"""
            SELECT f.id,
                   f.width,
                   f.height,
                   f.tagger_sig,
                   f.last_tagged_at,
                   t.name AS tag_name,
                   t.category,
                   ft.score
            FROM files AS f
            JOIN file_tags AS ft ON ft.file_id = f.id
            JOIN tags AS t ON t.id = ft.tag_id
            WHERE f.id IN ({placeholders})
            ORDER BY f.id
            """,
            tuple(file_ids),
        ).fetchall()
    finally:
        conn.close()

    assert [int(row["id"]) for row in stored] == file_ids
    assert all(row["width"] == 64 and row["height"] == 48 for row in stored)
    assert all(row["tagger_sig"] == "sig:v1" for row in stored)
    assert all(row["last_tagged_at"] == pytest.approx(1234.5) for row in stored)
    assert all(row["tag_name"] == "artist:kobato" for row in stored)
    assert all(row["category"] == 1 for row in stored)
    assert all(row["score"] == pytest.approx(0.9) for row in stored)
