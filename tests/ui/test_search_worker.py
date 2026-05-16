"""Tests for the background search worker without starting a thread."""

from __future__ import annotations

from pathlib import Path

import ui.search_worker as search_worker
from db.fts import fts_replace_rows
from db.schema import apply_schema
from ui.search_worker import SearchWorker


def test_search_worker_emits_chunks_until_max_rows(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_search_files(conn, where_sql, params, **kwargs):
        calls.append({"where": where_sql, "params": params, **kwargs})
        offset = int(kwargs["offset"])
        limit = int(kwargs["limit"])
        rows = [{"id": idx} for idx in range(offset, min(offset + limit, 5))]
        return rows

    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    worker = SearchWorker(
        tmp_path / "db.sqlite",
        "",
        [],
        tags_for_relevance=["1girl"],
        thresholds={0: 0.35},
        order="relevance",
        chunk=2,
        offset=-10,
        max_rows=3,
    )
    chunks: list[list[dict[str, int]]] = []
    finished: list[tuple[bool, bool]] = []
    worker.chunkReady.connect(chunks.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert chunks == [[{"id": 0}, {"id": 1}], [{"id": 2}]]
    assert finished == [(True, False)]
    assert calls[0]["where"] == "1=1"
    assert calls[0]["params"] == ()
    assert calls[0]["tags_for_relevance"] == ("1girl",)
    assert calls[0]["thresholds"] == {0: 0.35}
    assert calls[0]["limit"] == 2
    assert calls[1]["limit"] == 1


def test_search_worker_reports_operational_error(monkeypatch, tmp_path: Path) -> None:
    def fake_search_files(*args, **kwargs):
        raise search_worker.sqlite3.OperationalError("broken query")

    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    worker = SearchWorker(tmp_path / "db.sqlite", "bad sql", [], chunk=0)
    errors: list[str] = []
    finished: list[tuple[bool, bool]] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert errors == ["broken query"]
    assert finished == [(False, False)]


def test_search_worker_reports_unexpected_error_without_leaving_busy_state(monkeypatch, tmp_path: Path) -> None:
    def fake_search_files(*args, **kwargs):
        raise ValueError("unexpected search failure")

    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    errors: list[str] = []
    finished: list[tuple[bool, bool]] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert errors == ["unexpected search failure"]
    assert finished == [(False, False)]
    assert worker._connection is None


def test_search_worker_reports_connection_error(monkeypatch, tmp_path: Path) -> None:
    def fail_connect(*args, **kwargs):
        raise search_worker.sqlite3.OperationalError("cannot open database")

    monkeypatch.setattr(search_worker.sqlite3, "connect", fail_connect)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    errors: list[str] = []
    finished: list[tuple[bool, bool]] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert errors == ["cannot open database"]
    assert finished == [(False, False)]


def test_search_worker_cancel_before_run_does_not_connect(monkeypatch, tmp_path: Path) -> None:
    def fail_connect(*args, **kwargs):
        raise AssertionError("connect should not be called after pre-run cancellation")

    monkeypatch.setattr(search_worker.sqlite3, "connect", fail_connect)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    finished: list[tuple[bool, bool]] = []
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.cancel()
    worker.run()

    assert finished == [(False, True)]


def test_search_worker_cancelled_connection_error_reports_cancelled(monkeypatch, tmp_path: Path) -> None:
    def fail_connect(*args, **kwargs):
        worker.cancel()
        raise search_worker.sqlite3.OperationalError("cannot open database")

    monkeypatch.setattr(search_worker.sqlite3, "connect", fail_connect)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    errors: list[str] = []
    finished: list[tuple[bool, bool]] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert errors == []
    assert finished == [(False, True)]


def test_search_worker_cancel_after_chunk_reports_cancelled(monkeypatch, tmp_path: Path) -> None:
    def fake_search_files(conn, where_sql, params, **kwargs):
        offset = int(kwargs["offset"])
        if offset:
            return []
        return [{"id": 1}]

    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [], chunk=10, chunk_delay=60.0)
    chunks: list[list[dict[str, int]]] = []
    finished: list[tuple[bool, bool]] = []

    def on_chunk(rows):
        chunks.append(rows)
        worker.cancel()

    worker.chunkReady.connect(on_chunk)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert chunks == [[{"id": 1}]]
    assert finished == [(False, True)]


def test_search_worker_deleted_during_chunk_emit_reports_cancelled(monkeypatch, tmp_path: Path) -> None:
    def fake_search_files(*args, **kwargs):
        return [{"id": 1}]

    def fail_emit_chunk(rows):
        assert rows == [{"id": 1}]
        raise RuntimeError("wrapped C/C++ object of type SearchWorker has been deleted")

    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    monkeypatch.setattr(worker, "_emit_chunk", fail_emit_chunk)
    finished: list[tuple[bool, bool]] = []
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert finished == [(False, True)]
    assert worker._progress_handler() == 1


def test_search_worker_deleted_during_chunk_emit_quits_thread_when_finished_cannot_emit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_search_files(*args, **kwargs):
        return [{"id": 1}]

    def fail_emit_chunk(rows):
        assert rows == [{"id": 1}]
        raise RuntimeError("wrapped C/C++ object of type SearchWorker has been deleted")

    class _FakeThread:
        def __init__(self) -> None:
            self.quit_calls = 0

        def quit(self) -> None:
            self.quit_calls += 1

    class _FakeApp:
        def __init__(self, main_thread: _FakeThread) -> None:
            self._main_thread = main_thread

        def thread(self) -> _FakeThread:
            return self._main_thread

    class _FakeQCoreApplication:
        @staticmethod
        def instance() -> _FakeApp:
            return fake_app

    class _FakeQThread:
        @staticmethod
        def currentThread() -> _FakeThread:
            return worker_thread

    main_thread = _FakeThread()
    worker_thread = _FakeThread()
    fake_app = _FakeApp(main_thread)

    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    monkeypatch.setattr(search_worker, "QCoreApplication", _FakeQCoreApplication)
    monkeypatch.setattr(search_worker, "QThread", _FakeQThread)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    monkeypatch.setattr(worker, "_emit_chunk", fail_emit_chunk)
    monkeypatch.setattr(worker, "_emit_finished", lambda ok, cancelled: False)

    worker.run()

    assert worker_thread.quit_calls == 1
    assert main_thread.quit_calls == 0
    assert worker._progress_handler() == 1


def test_search_worker_cancelled_operational_error_reports_cancelled(monkeypatch, tmp_path: Path) -> None:
    def fake_search_files(*args, **kwargs):
        worker.cancel()
        raise search_worker.sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    errors: list[str] = []
    finished: list[tuple[bool, bool]] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert errors == []
    assert finished == [(False, True)]


def test_search_worker_cancel_interrupts_open_connection(monkeypatch, tmp_path: Path) -> None:
    class _FakeConnection:
        def __init__(self) -> None:
            self.row_factory = None
            self.interrupt_calls = 0
            self.close_calls = 0
            self.progress_handler = None

        def set_progress_handler(self, callback, instructions: int) -> None:
            self.progress_handler = (callback, instructions)

        def interrupt(self) -> None:
            self.interrupt_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    fake_conn = _FakeConnection()

    def fake_connect(*args, **kwargs):
        return fake_conn

    def fake_search_files(*args, **kwargs):
        worker.cancel()
        raise search_worker.sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(search_worker.sqlite3, "connect", fake_connect)
    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    errors: list[str] = []
    finished: list[tuple[bool, bool]] = []
    worker.error.connect(errors.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert errors == []
    assert finished == [(False, True)]
    assert fake_conn.interrupt_calls == 1
    assert fake_conn.close_calls == 1
    assert fake_conn.progress_handler is not None


def test_search_worker_max_rows_zero_finishes_without_query(monkeypatch, tmp_path: Path) -> None:
    def fail_search_files(*args, **kwargs):
        raise AssertionError("search should not run when max_rows is zero")

    monkeypatch.setattr(search_worker, "_search_files", fail_search_files)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [], max_rows=0)
    chunks: list[list[object]] = []
    finished: list[tuple[bool, bool]] = []
    worker.chunkReady.connect(chunks.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert chunks == []
    assert finished == [(True, False)]


def test_search_worker_deleted_during_finished_emit_quits_thread(monkeypatch, tmp_path: Path) -> None:
    class _FakeThread:
        def __init__(self) -> None:
            self.quit_calls = 0

        def quit(self) -> None:
            self.quit_calls += 1

    class _FakeApp:
        def __init__(self, main_thread: _FakeThread) -> None:
            self._main_thread = main_thread

        def thread(self) -> _FakeThread:
            return self._main_thread

    class _FakeQCoreApplication:
        @staticmethod
        def instance() -> _FakeApp:
            return fake_app

    class _FakeQThread:
        @staticmethod
        def currentThread() -> _FakeThread:
            return worker_thread

    def fake_search_files(*args, **kwargs):
        return []

    def fail_emit_finished(ok: bool, cancelled: bool) -> bool:
        assert (ok, cancelled) == (True, False)
        return False

    main_thread = _FakeThread()
    worker_thread = _FakeThread()
    fake_app = _FakeApp(main_thread)

    monkeypatch.setattr(search_worker, "_search_files", fake_search_files)
    monkeypatch.setattr(search_worker, "QCoreApplication", _FakeQCoreApplication)
    monkeypatch.setattr(search_worker, "QThread", _FakeQThread)
    worker = SearchWorker(tmp_path / "db.sqlite", "1=1", [])
    monkeypatch.setattr(worker, "_emit_finished", fail_emit_finished)

    worker.run()

    assert worker_thread.quit_calls == 1
    assert main_thread.quit_calls == 0


def test_search_worker_cancel_sets_progress_handler() -> None:
    worker = SearchWorker(":memory:", "1=1", [])

    assert worker._progress_handler() == 0
    worker.cancel()
    assert worker._progress_handler() == 1


def test_search_worker_reads_rebuilt_fts_with_offset_and_max_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "search-rebuilt.db"
    conn = search_worker.sqlite3.connect(db_path)
    conn.row_factory = search_worker.sqlite3.Row
    try:
        apply_schema(conn)
        file_ids: list[int] = []
        for idx in range(4):
            cursor = conn.execute(
                "INSERT INTO files(path, size, mtime, is_present) VALUES (?, ?, ?, 1)",
                (f"C:/images/{idx}.png", 100 + idx, float(idx)),
            )
            file_ids.append(int(cursor.lastrowid))
        fts_replace_rows(conn, [(file_id, "tag_alpha") for file_id in file_ids])
        conn.commit()
    finally:
        conn.close()

    worker = SearchWorker(
        db_path,
        "f.id IN (SELECT rowid FROM fts_files WHERE fts_files MATCH ?)",
        ["tag_alpha"],
        chunk=1,
        offset=1,
        max_rows=2,
    )
    chunks: list[list[dict[str, object]]] = []
    finished: list[tuple[bool, bool]] = []
    worker.chunkReady.connect(chunks.append)
    worker.finished.connect(lambda ok, cancelled: finished.append((ok, cancelled)))

    worker.run()

    assert [[row["path"] for row in chunk] for chunk in chunks] == [
        ["C:/images/2.png"],
        ["C:/images/1.png"],
    ]
    assert finished == [(True, False)]
