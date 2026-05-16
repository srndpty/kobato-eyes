"""Unit tests for the retagging helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from core.config import PipelineSettings
from core.pipeline.retag import RetagResult, _RetagScanStage, retag_all, retag_query
from core.pipeline.signature import current_tagger_sig
from core.pipeline.types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter
from db.schema import ensure_schema


def _prepare_files(
    db_path: Path,
    rows: list[tuple[str, str | None, float | None]],
) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    conn.executemany(
        "INSERT INTO files (path, tagger_sig, last_tagged_at) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _fetch_tagging_state(db_path: Path) -> dict[str, tuple[str | None, float | None]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT path, tagger_sig, last_tagged_at FROM files ORDER BY path",
        ).fetchall()
        return {str(row["path"]): (row["tagger_sig"], row["last_tagged_at"]) for row in rows}
    finally:
        conn.close()


def _make_context(db_path: Path, *, is_cancelled=None) -> PipelineContext:
    settings = PipelineSettings()
    return PipelineContext(
        db_path=db_path,
        settings=settings,
        thresholds={},
        max_tags_map={},
        tagger_sig=current_tagger_sig(settings),
        is_cancelled=is_cancelled,
    )


def _insert_indexed_file(db_path: Path, path: Path, *, sha256: str = "sha") -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ensure_schema(conn)
        stat = path.stat()
        cursor = conn.execute(
            """
            INSERT INTO files (path, size, mtime, sha256, indexed_at, tagger_sig, last_tagged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (str(path), stat.st_size, stat.st_mtime, sha256, 10.0, "old-sig", 20.0),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def test_retag_query_resets_selected_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_query.db"
    _prepare_files(
        db_path,
        [
            ("file_a", "sig-alpha", 123.0),
            ("file_b", "sig-alpha", 456.0),
            ("file_c", "sig-beta", 789.0),
        ],
    )

    result = retag_query(db_path, "path IN (?, ?)", ("file_a", "file_b"))

    assert isinstance(result, RetagResult)
    assert result.affected == 2
    assert sorted(result.file_ids) == [1, 2]
    state = _fetch_tagging_state(db_path)
    assert state["file_a"] == (None, None)
    assert state["file_b"] == (None, None)
    assert state["file_c"] == ("sig-beta", 789.0)


def test_retag_all_without_force_only_matches_signature(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_all_selective.db"
    settings = PipelineSettings()
    signature = current_tagger_sig(settings)
    _prepare_files(
        db_path,
        [
            ("matched", signature, 111.0),
            ("other", "different", 222.0),
        ],
    )

    result = retag_all(db_path, force=False, settings=settings)

    assert isinstance(result, RetagResult)
    assert result.affected == 1
    state = _fetch_tagging_state(db_path)
    assert state["matched"] == (None, None)
    assert state["other"] == ("different", 222.0)


def test_retag_all_force_updates_all_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_all_force.db"
    settings = PipelineSettings()
    signature = current_tagger_sig(settings)
    _prepare_files(
        db_path,
        [
            ("match_one", signature, 10.0),
            ("match_two", "different", 20.0),
            ("match_three", "another", 30.0),
        ],
    )

    result = retag_all(db_path, force=True, settings=settings)

    assert isinstance(result, RetagResult)
    assert result.affected == 3
    state = _fetch_tagging_state(db_path)
    assert state == {
        "match_one": (None, None),
        "match_three": (None, None),
        "match_two": (None, None),
    }


def test_retag_query_handles_empty_selection(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_empty.db"
    _prepare_files(db_path, [("file_a", "sig-alpha", 123.0)])

    result = retag_query(db_path, "path = ?", ("missing",))

    assert result == RetagResult(affected=0, file_ids=[])
    assert _fetch_tagging_state(db_path)["file_a"] == ("sig-alpha", 123.0)


def test_retag_query_updates_large_selection_in_chunks(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_many.db"
    rows = [(f"file_{index:04d}", "sig-alpha", 123.0) for index in range(905)]
    _prepare_files(db_path, rows)

    result = retag_query(db_path, "tagger_sig = ?", ("sig-alpha",))

    assert result.affected == 905
    assert len(result.file_ids) == 905
    assert set(_fetch_tagging_state(db_path).values()) == {(None, None)}


def test_retag_scan_stage_ignores_missing_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_missing_id.db"
    _prepare_files(db_path, [("file_a", "sig-alpha", 123.0)])
    progress: list[IndexProgress] = []

    result = _RetagScanStage([999]).run(_make_context(db_path), ProgressEmitter(progress.append))

    assert result.records == []
    assert result.scanned == 0
    assert result.new_or_changed == 0
    assert progress[-1].phase is IndexPhase.SCAN
    assert progress[-1].done == 1
    assert progress[-1].total == 1


def test_retag_scan_stage_skips_rows_when_stat_fails(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_stat_failure.db"
    missing_path = tmp_path / "missing.png"
    _prepare_files(db_path, [(str(missing_path), "sig-alpha", 123.0)])

    result = _RetagScanStage([1]).run(_make_context(db_path), ProgressEmitter(None))

    assert result.records == []
    assert result.scanned == 0
    assert result.new_or_changed == 0


def test_retag_scan_stage_detects_hash_change(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_changed.db"
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"old")
    file_id = _insert_indexed_file(db_path, image_path, sha256="old-sha")
    image_path.write_bytes(b"changed")

    result = _RetagScanStage([file_id]).run(_make_context(db_path), ProgressEmitter(None))

    assert result.scanned == 1
    assert result.new_or_changed == 1
    assert result.records[0].file_id == file_id
    assert result.records[0].changed is True


def test_retag_scan_stage_honors_cancellation(tmp_path: Path) -> None:
    db_path = tmp_path / "retag_cancel.db"
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    first_id = _insert_indexed_file(db_path, first_path)
    second_id = _insert_indexed_file(db_path, second_path)
    calls = 0

    def is_cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    result = _RetagScanStage([first_id, second_id]).run(
        _make_context(db_path, is_cancelled=is_cancelled),
        ProgressEmitter(None),
    )

    assert [record.file_id for record in result.records] == [first_id]
    assert result.scanned == 1
