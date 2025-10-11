"""Integration tests for :mod:`services.db_writing`."""

from __future__ import annotations

import sqlite3
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Sequence

import pytest

from db.schema import ensure_schema

if "pydantic" not in sys.modules:  # pragma: no cover - test shim
    fake = types.ModuleType("pydantic")

    class _BaseModel:
        model_config = {}

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - defensive
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def model_validate(cls, data):
            if isinstance(data, dict):
                return cls(**data)
            return cls()

        def model_dump(self):
            return dict(self.__dict__)

    def _field(*args, default=None, default_factory=None, **kwargs):  # noqa: D401 - mimic signature
        return default if default_factory is None else default_factory()

    def _validator(*_fields, **_kwargs):
        def _decorator(func):
            return func

        return _decorator

    fake.BaseModel = _BaseModel
    fake.ConfigDict = dict
    fake.Field = _field
    fake.field_validator = _validator
    fake.model_validator = lambda *args, **kwargs: _validator(*args, **kwargs)
    sys.modules["pydantic"] = fake

if "yaml" not in sys.modules:  # pragma: no cover - test shim
    yaml_stub = types.ModuleType("yaml")

    def _identity(value):
        return value

    yaml_stub.safe_load = _identity
    yaml_stub.safe_dump = _identity
    sys.modules["yaml"] = yaml_stub

if "core" not in sys.modules:
    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["core"] = core_pkg

if "core.pipeline" not in sys.modules:
    pipeline_pkg = types.ModuleType("core.pipeline")
    pipeline_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules["core.pipeline"] = pipeline_pkg

contracts_module = types.ModuleType("core.pipeline.contracts")


@dataclass(frozen=True)
class DBItem:
    file_id: int
    tags: Sequence[tuple[str, float, int]]
    width: Optional[int]
    height: Optional[int]
    tagger_sig: Optional[str]
    tagged_at: Optional[float]


@dataclass(frozen=True)
class DBFlush:
    pass


@dataclass(frozen=True)
class DBStop:
    pass


class DBWriteQueue(Protocol):  # pragma: no cover - structural contract only
    def start(self) -> None: ...

    def raise_if_failed(self) -> None: ...

    def put(self, item: object, block: bool = True, timeout: float | None = None) -> None: ...

    def qsize(self) -> int: ...

    def stop(self, *, flush: bool = True, wait_forever: bool = False) -> None: ...


contracts_module.DBItem = DBItem
contracts_module.DBFlush = DBFlush
contracts_module.DBStop = DBStop
contracts_module.DBWriteQueue = DBWriteQueue
contracts_module.__all__ = ["DBItem", "DBFlush", "DBStop", "DBWriteQueue"]

sys.modules["core.pipeline.contracts"] = contracts_module

from services.db_writing import DBWritingService


def _prepare_database(
    db_path: str,
    *,
    files: Sequence[tuple[int, str, tuple[int | None, int | None, str | None, float | None]]],
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        ensure_schema(conn)
        for file_id, path, meta in files:
            width, height, tagger_sig, last_tagged_at = meta
            conn.execute(
                "INSERT INTO files(id, path, width, height, tagger_sig, last_tagged_at) VALUES(?, ?, ?, ?, ?, ?)",
                (file_id, path, width, height, tagger_sig, last_tagged_at),
            )
        conn.commit()
    finally:
        conn.close()


def _patch_get_conn(monkeypatch: pytest.MonkeyPatch, db_path: str) -> None:
    def _connect(_: str, *, timeout: float = 30.0, allow_when_quiesced: bool = False) -> sqlite3.Connection:
        del allow_when_quiesced
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    monkeypatch.setattr("services.db_writing.get_conn", _connect)


def _collect_rows(
    conn: sqlite3.Connection,
    query: str,
    params: Iterable[object] | None = None,
) -> List[tuple]:
    cursor = conn.execute(query, tuple(params or ()))
    try:
        return [tuple(row) for row in cursor.fetchall()]
    finally:
        cursor.close()


def _wait_for_writes(service: DBWritingService, expected: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        service.raise_if_failed()
        if getattr(service, "_written", 0) >= expected:
            return
        time.sleep(0.01)
    service.raise_if_failed()
    pytest.fail(
        "DBWritingService did not flush "
        f"{expected} items in time (written={getattr(service, '_written', 0)}, "
        f"pending={service.qsize()})"
    )


def test_db_writing_standard_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "standard.db"
    _prepare_database(
        str(db_path),
        files=[
            (1, "a.jpg", (None, None, None, None)),
            (2, "b.jpg", (100, 200, "legacy", 1.0)),
        ],
    )
    _patch_get_conn(monkeypatch, str(db_path))
    service = DBWritingService(
        str(db_path),
        flush_chunk=2,
        fts_topk=8,
        default_tagger_sig="default",
    )
    service.start()
    try:
        service.put(
            DBItem(
                file_id=1,
                tags=[("tag_alpha", 0.9, 0), ("tag_beta", 0.5, 1)],
                width=640,
                height=480,
                tagger_sig="explicit",
                tagged_at=123.0,
            )
        )
        service.put(
            DBItem(
                file_id=2,
                tags=[("tag_gamma", 0.8, 0)],
                width=None,
                height=None,
                tagger_sig=None,
                tagged_at=456.0,
            )
        )
        _wait_for_writes(service, 2)
    finally:
        service.stop(wait_forever=True)

    conn = sqlite3.connect(db_path)
    try:
        tag_rows = _collect_rows(
            conn,
            """
            SELECT ft.file_id, t.name, ft.score
            FROM file_tags AS ft
            JOIN tags AS t ON t.id = ft.tag_id
            ORDER BY ft.file_id, t.name
            """,
        )
        assert tag_rows == [
            (1, "tag_alpha", pytest.approx(0.9)),
            (1, "tag_beta", pytest.approx(0.5)),
            (2, "tag_gamma", pytest.approx(0.8)),
        ]

        assert _collect_rows(
            conn,
            "SELECT rowid FROM fts_files WHERE fts_files MATCH ? ORDER BY rowid",
            ("tag_alpha",),
        ) == [(1,)]
        assert _collect_rows(
            conn,
            "SELECT rowid FROM fts_files WHERE fts_files MATCH ? ORDER BY rowid",
            ("tag_beta",),
        ) == [(1,)]
        assert _collect_rows(
            conn,
            "SELECT rowid FROM fts_files WHERE fts_files MATCH ? ORDER BY rowid",
            ("tag_gamma",),
        ) == [(2,)]

        meta_rows = _collect_rows(
            conn,
            "SELECT id, width, height, tagger_sig, last_tagged_at FROM files ORDER BY id",
        )
        assert meta_rows == [
            (1, 640, 480, "explicit", pytest.approx(123.0)),
            (2, 100, 200, "default", pytest.approx(456.0)),
        ]
    finally:
        conn.close()


@pytest.mark.parametrize("env_flag, skip_flag", [(True, False), (False, True)])
def test_db_writing_skip_fts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_flag: bool,
    skip_flag: bool,
) -> None:
    db_path = tmp_path / ("skip-env.db" if env_flag else "skip-flag.db")
    _prepare_database(str(db_path), files=[(10, "x.jpg", (None, None, None, None))])
    _patch_get_conn(monkeypatch, str(db_path))
    if env_flag:
        monkeypatch.setenv("KE_SKIP_FTS_DURING_TAG", "1")
    else:
        monkeypatch.delenv("KE_SKIP_FTS_DURING_TAG", raising=False)

    service = DBWritingService(
        str(db_path),
        flush_chunk=1,
        fts_topk=8,
        skip_fts=skip_flag,
        default_tagger_sig="sig",
    )
    service.start()
    try:
        service.put(
            DBItem(
                file_id=10,
                tags=[("tag_only", 0.77, 0)],
                width=320,
                height=240,
                tagger_sig=None,
                tagged_at=789.0,
            )
        )
        _wait_for_writes(service, 1)
    finally:
        service.stop(wait_forever=True)

    conn = sqlite3.connect(db_path)
    try:
        tag_rows = _collect_rows(
            conn,
            """
            SELECT ft.file_id, t.name
            FROM file_tags AS ft
            JOIN tags AS t ON t.id = ft.tag_id
            ORDER BY ft.file_id
            """,
        )
        assert tag_rows == [(10, "tag_only")]

        fts_count = conn.execute("SELECT COUNT(*) FROM fts_files").fetchone()[0]
        assert fts_count == 0
    finally:
        conn.close()


def test_db_writing_unsafe_fast_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "unsafe.db"
    _prepare_database(
        str(db_path),
        files=[
            (21, "m.jpg", (None, None, None, None)),
            (22, "n.jpg", (None, None, None, None)),
        ],
    )
    _patch_get_conn(monkeypatch, str(db_path))
    progress_log: list[tuple[str, int, int]] = []

    def _progress(kind: str, done: int, total: int) -> None:
        progress_log.append((kind, done, total))

    service = DBWritingService(
        str(db_path),
        flush_chunk=1,
        fts_topk=8,
        unsafe_fast=True,
        default_tagger_sig="fast",
        progress_cb=_progress,
    )
    service.start()
    try:
        service.put(
            DBItem(
                file_id=21,
                tags=[("tag_one", 0.6, 0)],
                width=512,
                height=512,
                tagger_sig=None,
                tagged_at=101.0,
            )
        )
        service.put(
            DBItem(
                file_id=22,
                tags=[("tag_two", 0.4, 0)],
                width=256,
                height=256,
                tagger_sig=None,
                tagged_at=202.0,
            )
        )
        _wait_for_writes(service, 2)
    finally:
        service.stop(wait_forever=True)

    conn = sqlite3.connect(db_path)
    try:
        meta_rows = _collect_rows(
            conn,
            "SELECT id, width, height, tagger_sig FROM files ORDER BY id",
        )
        assert meta_rows == [
            (21, 512, 512, "fast"),
            (22, 256, 256, "fast"),
        ]

        temp_names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'tmp_%'")
        }
        assert "tmp_file_ids" not in temp_names
    finally:
        conn.close()

    kinds = {kind for kind, _done, _total in progress_log}
    assert "merge.start" in kinds
    assert "merge.done" in kinds
    assert any(kind.startswith("merge.") for kind, _done, _total in progress_log)
