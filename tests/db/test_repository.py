"""Tests for the SQLite repository layer."""

from __future__ import annotations

import sqlite3
from typing import Iterator

import pytest

from db.connection import get_conn
from db.repository import (
    replace_file_tags,
    update_fts,
    upsert_embedding,
    upsert_file,
    upsert_signatures,
    upsert_tags,
)
from db.schema import apply_schema


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

    row = memory_conn.execute(
        "SELECT size, sha256 FROM files WHERE id = ?", (file_id,)
    ).fetchone()
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
    count_row = memory_conn.execute(
        "SELECT COUNT(*) AS cnt FROM file_tags WHERE file_id = ?", (file_id,)
    ).fetchone()
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
        "SELECT file_id FROM fts_files WHERE fts_files MATCH ?",
        ("safe",),
    ).fetchone()
    assert match is not None and match["file_id"] == file_id

    update_fts(memory_conn, file_id, None)
    match_after_delete = memory_conn.execute(
        "SELECT file_id FROM fts_files WHERE rowid = ?",
        (file_id,),
    ).fetchone()
    assert match_after_delete is None

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

    vector = bytes(range(16))
    upsert_embedding(
        memory_conn,
        file_id=file_id,
        model="clip-vit",
        dim=16,
        vector=vector,
    )
    upsert_embedding(
        memory_conn,
        file_id=file_id,
        model="clip-vit",
        dim=16,
        vector=vector[::-1],
    )
    embed_row = memory_conn.execute(
        "SELECT dim, vector FROM embeddings WHERE file_id = ? AND model = ?",
        (file_id, "clip-vit"),
    ).fetchone()
    assert embed_row is not None
    assert embed_row["dim"] == 16
    assert bytes(embed_row["vector"]) == vector[::-1]

    # Ensure FTS content is still queryable after all operations.
    final_match = memory_conn.execute(
        "SELECT rowid FROM fts_files WHERE fts_files MATCH ?",
        ("duplicate",),
    ).fetchone()
    assert final_match is not None and final_match["rowid"] == file_id
