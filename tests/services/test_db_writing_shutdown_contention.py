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
    service = DBWritingService(str(db_path), flush_chunk=16, fts_topk=0)
    service.start()

    lock_conn = sqlite3.connect(db_path, timeout=1.0)
    lock_conn.execute("BEGIN IMMEDIATE")
    for file_id in file_ids:
        service.put(DBItem(file_id, [("artist:kobato", 0.9, 1)], 64, 48, "sig:v1", 1234.5))

    stop_started = threading.Event()
    stop_finished = threading.Event()
    stop_errors: list[BaseException] = []

    def stop_service() -> None:
        stop_started.set()
        try:
            service.stop(flush=True, wait_forever=True)
        except BaseException as exc:  # pragma: no cover - assertion reports worker failures
            stop_errors.append(exc)
        finally:
            stop_finished.set()

    stop_thread = threading.Thread(target=stop_service, name="DBWriterStopTest")
    stop_thread.start()
    try:
        assert stop_started.wait(timeout=2.0)
        assert not stop_finished.wait(timeout=0.5)
    finally:
        lock_conn.rollback()
        lock_conn.close()

    assert stop_finished.wait(timeout=10.0)
    stop_thread.join(timeout=1.0)
    assert not stop_thread.is_alive()
    assert not service._thread.is_alive()
    assert stop_errors == []

    conn = get_conn(db_path, timeout=2.0)
    try:
        stored = conn.execute(
            """
            SELECT f.id, f.width, f.height, f.tagger_sig, COUNT(ft.tag_id) AS tag_count
            FROM files AS f
            LEFT JOIN file_tags AS ft ON ft.file_id = f.id
            WHERE f.id IN (?, ?, ?)
            GROUP BY f.id
            ORDER BY f.id
            """,
            file_ids,
        ).fetchall()
    finally:
        conn.close()

    assert [int(row["id"]) for row in stored] == file_ids
    assert all(row["width"] == 64 and row["height"] == 48 for row in stored)
    assert all(row["tagger_sig"] == "sig:v1" for row in stored)
    assert all(row["tag_count"] == 1 for row in stored)
