"""Tests for watcher path scheduling helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.config import PipelineSettings
from core.pipeline import watcher
from core.pipeline.watcher import ProcessingPipeline, _FileProcessJob, resolve_watch_paths
from core.tag_job import TagJobConfig


def test_resolve_watch_paths_filters_duplicates_and_extensions(tmp_path: Path) -> None:
    first = tmp_path / "a.PNG"
    duplicate = tmp_path / "a.PNG"
    ignored = tmp_path / "note.txt"

    resolved, scheduled = resolve_watch_paths([first, duplicate, ignored], allow_exts={".png"})

    assert resolved == [first.expanduser().resolve()]
    assert scheduled == {first.expanduser().resolve()}


def test_resolve_watch_paths_skips_already_scheduled(tmp_path: Path) -> None:
    first = tmp_path / "a.png"
    existing = first.expanduser().resolve()

    resolved, scheduled = resolve_watch_paths([first], allow_exts={".png"}, already_scheduled={existing})

    assert resolved == []
    assert scheduled == set()


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeIndexer:
    def __init__(self, *, result: list[int] | None = None, fail: bool = False) -> None:
        self.result = result or []
        self.fail = fail
        self.calls: list[list[Path]] = []

    def index_paths(self, paths):  # type: ignore[no-untyped-def]
        self.calls.append(list(paths))
        if self.fail:
            raise RuntimeError("index failed")
        return self.result


class _FakeSignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback):  # type: ignore[no-untyped-def]
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in list(self._callbacks):
            callback()


class _FakeSignals:
    def __init__(self) -> None:
        self.finished = _FakeSignal()


class _CapturingJobManager:
    def __init__(self) -> None:
        self.jobs = []
        self.signals: list[_FakeSignals] = []
        self.shutdown_calls = 0

    def submit(self, job, priority=None):  # type: ignore[no-untyped-def]
        self.jobs.append(job)
        signals = _FakeSignals()
        self.signals.append(signals)
        return signals

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_file_process_job_prepare_and_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = _FakeConnection()
    monkeypatch.setattr(watcher, "get_conn", lambda _db_path: conn)
    job = _FileProcessJob(
        [tmp_path / "missing.png"],
        db_path=tmp_path / "kobato.db",
        tagger=object(),
        tag_config=TagJobConfig(),
        indexer=None,
    )

    job.prepare()
    assert job._conn is conn

    job.cleanup()
    assert conn.closed is True
    assert job._conn is None


def test_file_process_job_skips_when_not_prepared(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not an image")
    job = _FileProcessJob(
        [image_path],
        db_path=tmp_path / "kobato.db",
        tagger=object(),
        tag_config=TagJobConfig(),
        indexer=None,
    )

    assert job.process_item(image_path, image_path) is None


def test_file_process_job_skips_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    conn = _FakeConnection()
    monkeypatch.setattr(watcher, "get_conn", lambda _db_path: conn)
    job = _FileProcessJob(
        [tmp_path / "missing.png"],
        db_path=tmp_path / "kobato.db",
        tagger=object(),
        tag_config=TagJobConfig(),
        indexer=None,
    )
    job.prepare()

    assert job.process_item(tmp_path / "missing.png", tmp_path / "missing.png") is None


def test_file_process_job_handles_tagger_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not an image")
    monkeypatch.setattr(watcher, "get_conn", lambda _db_path: _FakeConnection())

    def fail_tag_job(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("tag failed")

    monkeypatch.setattr(watcher, "run_tag_job", fail_tag_job)
    job = _FileProcessJob(
        [image_path],
        db_path=tmp_path / "kobato.db",
        tagger=object(),
        tag_config=TagJobConfig(),
        indexer=None,
    )
    job.prepare()

    assert job.process_item(image_path, image_path) is None


def test_file_process_job_uses_indexer_when_tag_job_has_no_file_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not an image")
    monkeypatch.setattr(watcher, "get_conn", lambda _db_path: _FakeConnection())
    monkeypatch.setattr(watcher, "run_tag_job", lambda *_args, **_kwargs: None)
    indexer = _FakeIndexer(result=[42])
    job = _FileProcessJob(
        [image_path],
        db_path=tmp_path / "kobato.db",
        tagger=object(),
        tag_config=TagJobConfig(),
        indexer=indexer,
    )
    job.prepare()

    assert job.process_item(image_path, image_path) == (image_path, 42)
    assert indexer.calls == [[image_path]]


def test_file_process_job_keeps_tag_file_id_when_indexer_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not an image")
    monkeypatch.setattr(watcher, "get_conn", lambda _db_path: _FakeConnection())
    monkeypatch.setattr(watcher, "run_tag_job", lambda *_args, **_kwargs: SimpleNamespace(file_id=7))
    job = _FileProcessJob(
        [image_path],
        db_path=tmp_path / "kobato.db",
        tagger=object(),
        tag_config=TagJobConfig(),
        indexer=_FakeIndexer(fail=True),
    )
    job.prepare()

    assert job.process_item(image_path, image_path) == (image_path, 7)


def test_processing_pipeline_update_settings_and_finished_callback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(watcher, "bootstrap_if_needed", lambda _db_path: None)
    monkeypatch.setattr(watcher, "ensure_dirs", lambda: None)
    manager = _CapturingJobManager()
    pipeline = ProcessingPipeline(
        db_path=tmp_path / "kobato.db",
        tagger=object(),
        job_manager=manager,
        settings=PipelineSettings.from_mapping({"allow_exts": [".png"]}),
    )
    updated = PipelineSettings.from_mapping(
        {
            "allow_exts": [".jpg"],
            "tagger": {"thresholds": {"general": 0.6}},
        }
    )

    pipeline.update_settings(updated)
    pipeline.enqueue_index([tmp_path / "image.png", tmp_path / "image.jpg"])

    assert len(manager.jobs) == 1
    assert manager.jobs[0].items == [(tmp_path / "image.jpg").expanduser().resolve()]
    assert pipeline._tag_config.thresholds is not None
    assert manager.jobs[0]._tag_config is pipeline._tag_config

    manager.signals[0].finished.emit()
    assert pipeline._scheduled == set()
