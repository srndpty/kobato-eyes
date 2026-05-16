"""Tests for the background search worker without starting a thread."""

from __future__ import annotations

from pathlib import Path

import ui.search_worker as search_worker
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


def test_search_worker_cancel_sets_progress_handler() -> None:
    worker = SearchWorker(":memory:", "1=1", [])

    assert worker._progress_handler() == 0
    worker.cancel()
    assert worker._progress_handler() == 1
