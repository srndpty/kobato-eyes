"""Tests for the SQLite repository layer."""

from __future__ import annotations

import sqlite3
from typing import Iterator

import pytest

import db.repository as repository_mod
from db.connection import get_conn
from db.files import bulk_update_files_meta_by_id as files_bulk_update
from db.fts import fts_replace_rows as fts_replace
from db.repository import (
    iter_files_for_dup,
    mark_files_absent,
    replace_file_tags,
    update_fts,
    upsert_file,
    upsert_signatures,
    upsert_tags,
)
from db.schema import apply_schema
from db.tags import upsert_tags as tags_upsert


@pytest.fixture()
def memory_conn() -> Iterator[sqlite3.Connection]:
    """Provide an in-memory database initialized with the application schema."""
    conn = get_conn(":memory:")
    apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_repository_roundtrip(memory_conn: sqlite3.Connection) -> None:
    file_id = upsert_file(
        memory_conn,
        path="C:/images/sample.png",
        size=1_024,
        mtime=123456.0,
        sha256="abc123",
    )
    # Updating the same file should keep the identifier stable.
    updated_id = upsert_file(
        memory_conn,
        path="C:/images/sample.png",
        size=2_048,
        mtime=654321.0,
        sha256="def456",
    )
    assert updated_id == file_id

    row = memory_conn.execute("SELECT size, sha256 FROM files WHERE id = ?", (file_id,)).fetchone()
    assert row is not None
    assert row["size"] == 2_048
    assert row["sha256"] == "def456"

    tags = upsert_tags(
        memory_conn,
        [
            {"name": "character:kobato", "category": 1},
            {"name": "rating:safe", "category": 0},
        ],
    )
    assert sorted(tags) == ["character:kobato", "rating:safe"]

    # Updating the category for an existing tag should persist the change.
    refreshed = upsert_tags(memory_conn, [{"name": "rating:safe", "category": 2}])
    assert tags["rating:safe"] == refreshed["rating:safe"]
    tag_row = memory_conn.execute(
        "SELECT category FROM tags WHERE id = ?",
        (refreshed["rating:safe"],),
    ).fetchone()
    assert tag_row is not None and tag_row["category"] == 2

    replace_file_tags(
        memory_conn,
        file_id,
        [
            (tags["character:kobato"], 0.9),
            (tags["rating:safe"], 1.0),
        ],
    )
    count_row = memory_conn.execute("SELECT COUNT(*) AS cnt FROM file_tags WHERE file_id = ?", (file_id,)).fetchone()
    assert count_row is not None and count_row["cnt"] == 2

    replace_file_tags(memory_conn, file_id, [(tags["rating:safe"], 0.5)])
    stored_score = memory_conn.execute(
        "SELECT score FROM file_tags WHERE file_id = ? AND tag_id = ?",
        (file_id, tags["rating:safe"]),
    ).fetchone()
    assert stored_score is not None
    assert stored_score["score"] == pytest.approx(0.5)

    update_fts(memory_conn, file_id, "kobato safe test")
    match = memory_conn.execute(
        "SELECT rowid AS file_id FROM fts_files WHERE fts_files MATCH ?",
        ("safe",),
    ).fetchone()
    assert match is not None and match["file_id"] == file_id

    update_fts(memory_conn, file_id, None)

    update_fts(memory_conn, file_id, "kobato duplicate")

    upsert_signatures(
        memory_conn,
        file_id=file_id,
        phash_u64=123,
        dhash_u64=456,
    )
    upsert_signatures(
        memory_conn,
        file_id=file_id,
        phash_u64=789,
        dhash_u64=654,
    )
    sig_row = memory_conn.execute(
        "SELECT phash_u64, dhash_u64 FROM signatures WHERE file_id = ?",
        (file_id,),
    ).fetchone()
    assert sig_row is not None
    assert sig_row["phash_u64"] == 789
    assert sig_row["dhash_u64"] == 654

    # Ensure FTS content is still queryable after all operations.
    final_match = memory_conn.execute(
        "SELECT rowid FROM fts_files WHERE fts_files MATCH ?",
        ("duplicate",),
    ).fetchone()
    assert final_match is not None and final_match["rowid"] == file_id


def test_iter_files_for_dup_and_mark_absent(memory_conn: sqlite3.Connection) -> None:
    present_id = upsert_file(
        memory_conn,
        path="C:/images/present.png",
        size=512,
        width=32,
        height=32,
    )
    upsert_signatures(
        memory_conn,
        file_id=present_id,
        phash_u64=0x1234,
        dhash_u64=0x5678,
    )

    missing_id = upsert_file(
        memory_conn,
        path="C:/images/missing.png",
        size=1024,
        width=64,
        height=64,
        is_present=0,
    )
    upsert_signatures(
        memory_conn,
        file_id=missing_id,
        phash_u64=0x1235,
        dhash_u64=0x5679,
    )

    rows = list(iter_files_for_dup(memory_conn, None))
    assert len(rows) == 1
    row = rows[0]
    assert row["file_id"] == present_id

    filtered_rows = list(iter_files_for_dup(memory_conn, "C:/images/present%"))
    assert len(filtered_rows) == 1
    assert filtered_rows[0]["file_id"] == present_id

    empty_rows = list(iter_files_for_dup(memory_conn, "Z:%"))
    assert empty_rows == []

    updated = mark_files_absent(memory_conn, [present_id])
    assert updated == 1
    status = memory_conn.execute("SELECT is_present FROM files WHERE id = ?", (present_id,)).fetchone()
    assert status is not None and status["is_present"] == 0


def test_repository_module_reexports() -> None:
    assert repository_mod.upsert_tags is tags_upsert
    assert repository_mod.bulk_update_files_meta_by_id is files_bulk_update
    assert repository_mod.fts_replace_rows is fts_replace
