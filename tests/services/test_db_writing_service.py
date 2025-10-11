"""Unit tests for :mod:`services.db_writing`."""

from __future__ import annotations

import importlib.util
import queue
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

from db.schema import apply_schema

_THIS_FILE = Path(__file__).resolve()
for _candidate in _THIS_FILE.parents:
    if (_candidate / "pyproject.toml").exists():
        PROJECT_ROOT = _candidate
        break
else:  # pragma: no cover - defensive fallback for unexpected layouts
    raise RuntimeError("Project root not found")

SRC_ROOT = PROJECT_ROOT / "src"

if "core.pipeline.contracts" not in sys.modules:
    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [str(SRC_ROOT / "core")]
    sys.modules.setdefault("core", core_pkg)

    pipeline_pkg = types.ModuleType("core.pipeline")
    pipeline_pkg.__path__ = [str(SRC_ROOT / "core" / "pipeline")]
    core_pkg.pipeline = pipeline_pkg
    sys.modules.setdefault("core.pipeline", pipeline_pkg)

    spec = importlib.util.spec_from_file_location(
        "core.pipeline.contracts",
        SRC_ROOT / "core" / "pipeline" / "contracts.py",
    )
    if spec and spec.loader:
        contracts_mod = importlib.util.module_from_spec(spec)
        sys.modules["core.pipeline.contracts"] = contracts_mod
        spec.loader.exec_module(contracts_mod)
        pipeline_pkg.contracts = contracts_mod

from core.pipeline.contracts import DBFlush, DBItem, DBStop
from db.connection import get_conn
from db.repository import upsert_file
import services.db_writing as db_writing_module
from services.db_writing import DBWritingService


@dataclass(frozen=True)
class _TestDBItem:
    file_id: int
    tags: Sequence[tuple[str, float, int]]
    width: int | None
    height: int | None
    tagger_sig: str | None
    tagged_at: float | None


def _prepare_db(db_path: str, path: str) -> int:
    conn = get_conn(db_path)
    try:
        apply_schema(conn)
        file_id = upsert_file(conn, path=path)
    finally:
        conn.close()
    return file_id


def _make_item(file_id: int) -> _TestDBItem:
    return _TestDBItem(
        file_id=file_id,
        tags=[("artist:kobato", 0.9, 1), ("rating:safe", 0.8, 0)],
        width=64,
        height=48,
        tagger_sig="sig:v1",
        tagged_at=1234.5,
    )


def test_flush_batch_standard_inserts_tags_and_fts(tmp_path: Path) -> None:
    db_path = tmp_path / "standard.db"
    file_id = _prepare_db(str(db_path), "C:/images/standard.png")

    service = DBWritingService(str(db_path), flush_chunk=2, fts_topk=16)
    conn = service._open_connection()
    try:
        service._apply_pragmas(conn)
        service._flush_batch(conn, [_make_item(file_id)])

        rows = conn.execute(
            """
            SELECT t.name, ft.score
            FROM file_tags AS ft
            JOIN tags AS t ON t.id = ft.tag_id
            WHERE ft.file_id = ?
            ORDER BY t.name
            """,
            (file_id,),
        ).fetchall()
        assert len(rows) == 2
        scores = {row["name"]: row["score"] for row in rows}
        assert scores["artist:kobato"] == pytest.approx(0.9)
        assert scores["rating:safe"] == pytest.approx(0.8)

        fts_hit = conn.execute(
            "SELECT rowid FROM fts_files WHERE fts_files MATCH ?",
            ('"artist:kobato"',),
        ).fetchone()
        assert fts_hit is not None
        assert int(fts_hit["rowid"]) == file_id

        meta = conn.execute(
            "SELECT width, height, tagger_sig, last_tagged_at FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        assert meta["width"] == 64
        assert meta["height"] == 48
        assert meta["tagger_sig"] == "sig:v1"
        assert meta["last_tagged_at"] == pytest.approx(1234.5)
    finally:
        conn.close()


def test_start_spawns_worker_and_processes_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    removed_items: list[object] = []

    class _RecordingQueue(queue.Queue):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)

        def get(self, block: bool = True, timeout: float | None = None) -> object:
            item = super().get(block=block, timeout=timeout)
            removed_items.append(item)
            return item

    monkeypatch.setattr(db_writing_module.queue, "Queue", _RecordingQueue)

    start_evt = threading.Event()
    flush_evt = threading.Event()
    close_evt = threading.Event()
    join_timeouts: list[float | None] = []
    flush_calls: list[tuple[DBItem, ...]] = []

    class _DummyConn:
        def close(self) -> None:
            close_evt.set()

    def _fake_open_connection(self: DBWritingService) -> _DummyConn:
        start_evt.set()
        return _DummyConn()

    def _fake_apply_pragmas(self: DBWritingService, conn: _DummyConn) -> None:  # noqa: ARG001
        return None

    def _fake_flush_batch(
        self: DBWritingService,
        conn: _DummyConn,  # noqa: ARG001
        items: Sequence[DBItem],
    ) -> None:
        flush_calls.append(tuple(items))
        flush_evt.set()

    monkeypatch.setattr(DBWritingService, "_open_connection", _fake_open_connection)
    monkeypatch.setattr(DBWritingService, "_apply_pragmas", _fake_apply_pragmas)
    monkeypatch.setattr(DBWritingService, "_flush_batch", _fake_flush_batch)

    service = DBWritingService("ignored.db", flush_chunk=10)

    orig_put = service._queue.put

    def _recording_put(item: object, block: bool = True, timeout: float | None = None) -> None:
        orig_put(item, block=block, timeout=timeout)
        if isinstance(item, DBFlush):
            time.sleep(0.05)

    monkeypatch.setattr(service._queue, "put", _recording_put)

    orig_join = service._thread.join

    def _join_wrapper(timeout: float | None = None) -> None:
        join_timeouts.append(timeout)
        orig_join(timeout=timeout)

    monkeypatch.setattr(service._thread, "join", _join_wrapper)

    service.start()
    assert start_evt.wait(1.0)

    item = DBItem(1, [("tag", 0.5, 0)], None, None, None, None)
    service.put(item)

    for _ in range(100):
        if service.qsize() == 0:
            break
        time.sleep(0.01)
    else:
        service.stop(wait_forever=True)
        pytest.fail("DBWritingService worker did not consume the queued DBItem in time")

    service.stop(wait_forever=True)

    assert flush_evt.wait(1.0)
    assert flush_calls == [(item,)]
    assert close_evt.wait(1.0)
    assert service._stop_evt.is_set()
    assert join_timeouts and all(timeout == 1.0 for timeout in join_timeouts)
    assert [type(msg) for msg in removed_items[:3]] == [DBItem, DBFlush, DBStop]


def test_stop_wait_forever_false_uses_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    service = DBWritingService("ignored.db")

    join_timeouts: list[float | None] = []

    class _DummyThread:
        def join(self, timeout: float | None = None) -> None:
            join_timeouts.append(timeout)

        def is_alive(self) -> bool:
            return False

    service._thread = _DummyThread()  # type: ignore[assignment]

    service.stop(wait_forever=False)

    assert join_timeouts == [10.0]
    assert service._stop_evt.is_set()


def test_flush_batch_skips_fts_when_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("KE_SKIP_FTS_DURING_TAG", "1")
    db_path = tmp_path / "skipfts.db"
    file_id = _prepare_db(str(db_path), "C:/images/skipfts.png")

    service = DBWritingService(str(db_path), flush_chunk=2)
    conn = service._open_connection()
    try:
        service._apply_pragmas(conn)
        service._flush_batch(conn, [_make_item(file_id)])

        count = conn.execute("SELECT COUNT(*) FROM fts_files").fetchone()[0]
        assert count == 0
    finally:
        conn.close()


def test_flush_batch_unsafe_fast_merges_into_persistent(tmp_path: Path) -> None:
    db_path = tmp_path / "unsafe.db"
    file_id = _prepare_db(str(db_path), "C:/images/unsafe.png")

    events: list[tuple[str, int, int]] = []

    def progress(kind: str, done: int, total: int) -> None:
        events.append((kind, done, total))

    service = DBWritingService(
        str(db_path),
        flush_chunk=1,
        unsafe_fast=True,
        progress_cb=progress,
    )
    conn = service._open_connection()
    try:
        service._apply_pragmas(conn)
        service._create_temp_staging(conn)
        service._flush_batch(conn, [_make_item(file_id)])
        service._merge_staging_into_persistent(conn)

        names = conn.execute(
            """
            SELECT t.name
            FROM file_tags AS ft
            JOIN tags AS t ON t.id = ft.tag_id
            WHERE ft.file_id = ?
            ORDER BY t.name
            """,
            (file_id,),
        ).fetchall()
        assert [row["name"] for row in names] == ["artist:kobato", "rating:safe"]

        meta = conn.execute(
            "SELECT width, height, tagger_sig, last_tagged_at FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        assert meta["width"] == 64
        assert meta["height"] == 48
        assert meta["tagger_sig"] == "sig:v1"
        assert meta["last_tagged_at"] == pytest.approx(1234.5)

        kinds = [kind for kind, _done, _total in events]
        assert "merge.start" in kinds
        assert "merge.done" in kinds
    finally:
        conn.close()


def test_flush_batch_triggers_checkpoint_when_wal_large(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "checkpoint.db"
    file_id = _prepare_db(str(db_path), "C:/images/checkpoint.png")

    service = DBWritingService(str(db_path), flush_chunk=1)
    monkeypatch.setattr(service, "_wal_size_mb", lambda: 300)

    conn = service._open_connection()
    try:
        service._apply_pragmas(conn)
        executed: list[str] = []
        conn.set_trace_callback(executed.append)
        service._flush_batch(conn, [_make_item(file_id)])
        conn.set_trace_callback(None)

        pragma_statements = [sql for sql in executed if "wal_checkpoint" in sql.lower()]
        assert any("PRAGMA wal_checkpoint(PASSIVE)" in sql for sql in pragma_statements)
    finally:
        conn.close()
