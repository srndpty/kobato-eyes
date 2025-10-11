"""Unit tests for the retagging helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from core.config import PipelineSettings
from core.pipeline.retag import retag_all, retag_query
from core.pipeline.signature import current_tagger_sig
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
        return {
            str(row["path"]): (row["tagger_sig"], row["last_tagged_at"])
            for row in rows
        }
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

    affected = retag_query(db_path, "path IN (?, ?)", ("file_a", "file_b"))

    assert affected == 2
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

    affected = retag_all(db_path, force=False, settings=settings)

    assert affected == 1
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

    affected = retag_all(db_path, force=True, settings=settings)

    assert affected == 3
    state = _fetch_tagging_state(db_path)
    assert state == {
        "match_one": (None, None),
        "match_three": (None, None),
        "match_two": (None, None),
    }

