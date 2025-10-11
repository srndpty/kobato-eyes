"""Tests for the SQLite repository layer."""

from __future__ import annotations

import sqlite3
import sys
from types import ModuleType, SimpleNamespace
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
    write_tagging_batch,
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


@pytest.fixture()
def tagging_dataset(
    memory_conn: sqlite3.Connection,
) -> tuple[sqlite3.Connection, dict[str, int], dict[str, int]]:
    """Seed files, tags, and FTS rows for write_tagging_batch tests."""

    first_file = upsert_file(
        memory_conn,
        path="C:/images/first.png",
        size=111,
        mtime=11.0,
        sha256="oldhash1",
        width=640,
        height=480,
        tagger_sig="sig-old-1",
        last_tagged_at=111.1,
    )
    second_file = upsert_file(
        memory_conn,
        path="C:/images/second.png",
        size=222,
        mtime=22.0,
        sha256="oldhash2",
        width=1024,
        height=768,
        tagger_sig="sig-old-2",
        last_tagged_at=222.2,
    )

    tags = upsert_tags(
        memory_conn,
        [
            {"name": "general:old", "category": 0},
            {"name": "character:kobato", "category": 1},
            {"name": "rating:safe", "category": 0},
        ],
    )

    replace_file_tags(memory_conn, first_file, [(tags["general:old"], 0.25)])
    replace_file_tags(memory_conn, second_file, [(tags["general:old"], 0.5)])
    update_fts(memory_conn, first_file, "old text one")
    update_fts(memory_conn, second_file, "old text two")

    return memory_conn, {"first": first_file, "second": second_file}, tags


def test_build_where_uses_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure thresholds are passed through when loading succeeds."""

    conn = sqlite3.connect(":memory:")

    def fake_load(connection: sqlite3.Connection) -> dict[int, float]:
        assert connection is conn
        return {1: 0.42}

    def fake_translate(
        query: str,
        *,
        file_alias: str,
        thresholds: dict[int, float],
    ) -> SimpleNamespace:
        assert query == "score>0"
        assert file_alias == "x"
        assert thresholds == {1: 0.42}
        return SimpleNamespace(where="x.score > ?", params=[0.5])

    monkeypatch.setattr(repository_mod, "_load_tag_thresholds", fake_load)
    stub_query_module = ModuleType("core.query")
    stub_query_module.translate_query = fake_translate
    stub_core_package = ModuleType("core")
    stub_core_package.__path__ = []  # mark as package for submodule imports
    stub_core_package.query = stub_query_module
    monkeypatch.setitem(sys.modules, "core", stub_core_package)
    monkeypatch.setitem(sys.modules, "core.query", stub_query_module)

    try:
        where, params = repository_mod.build_where_and_params_for_query(
            conn,
            "score>0",
            alias_file="x",
        )
    finally:
        conn.close()

    assert where == "x.score > ?"
    assert params == [0.5]


def test_build_where_falls_back_on_threshold_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When loading thresholds fails, fall back to empty values."""

    conn = sqlite3.connect(":memory:")

    def fake_load(connection: sqlite3.Connection) -> dict[int, float]:
        raise RuntimeError("thresholds unavailable")

    def fake_translate(
        query: str,
        *,
        file_alias: str,
        thresholds: dict[int, float],
    ) -> SimpleNamespace:
        assert thresholds == {}
        assert file_alias == "f"
        return SimpleNamespace(where="  ", params=("value",))

    monkeypatch.setattr(repository_mod, "_load_tag_thresholds", fake_load)
    stub_query_module = ModuleType("core.query")
    stub_query_module.translate_query = fake_translate
    stub_core_package = ModuleType("core")
    stub_core_package.__path__ = []  # mark as package for submodule imports
    stub_core_package.query = stub_query_module
    monkeypatch.setitem(sys.modules, "core", stub_core_package)
    monkeypatch.setitem(sys.modules, "core.query", stub_query_module)

    try:
        where, params = repository_mod.build_where_and_params_for_query(conn, "")
    finally:
        conn.close()

    assert where == "1=1"
    assert isinstance(params, list)
    assert params == ["value"]


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


def test_write_tagging_batch_updates_related_tables(
    tagging_dataset: tuple[sqlite3.Connection, dict[str, int], dict[str, int]],
) -> None:
    conn, files, tags = tagging_dataset

    count = write_tagging_batch(
        conn,
        [
            {
                "file_id": files["first"],
                "file_meta": {
                    "size": 333,
                    "mtime": 3333.3,
                    "sha256": "newhash1",
                    "width": 800,
                    "height": 600,
                    "tagger_sig": "sig-new-1",
                    "last_tagged_at": 9999.9,
                },
                "tags": [
                    (tags["character:kobato"], 0.95),
                    (tags["rating:safe"], 0.75),
                ],
                "fts_text": "kobato safe newtext",
            },
            {
                "file_id": files["second"],
                "file_meta": {
                    "size": 444,
                    "mtime": 4444.4,
                    "sha256": "newhash2",
                },
                "tags": [],
                "fts_text": None,
            },
        ],
    )

    assert count == 2

    first_row = conn.execute(
        "SELECT size, mtime, sha256, width, height, tagger_sig, last_tagged_at, is_present"
        " FROM files WHERE id = ?",
        (files["first"],),
    ).fetchone()
    assert first_row is not None
    assert first_row["size"] == 333
    assert first_row["mtime"] == pytest.approx(3333.3)
    assert first_row["sha256"] == "newhash1"
    assert first_row["width"] == 800
    assert first_row["height"] == 600
    assert first_row["tagger_sig"] == "sig-new-1"
    assert first_row["last_tagged_at"] == pytest.approx(9999.9)
    assert first_row["is_present"] == 1

    second_row = conn.execute(
        "SELECT size, mtime, sha256, width, height, tagger_sig, last_tagged_at, is_present"
        " FROM files WHERE id = ?",
        (files["second"],),
    ).fetchone()
    assert second_row is not None
    assert second_row["size"] == 444
    assert second_row["mtime"] == pytest.approx(4444.4)
    assert second_row["sha256"] == "newhash2"
    assert second_row["width"] == 1024
    assert second_row["height"] == 768
    assert second_row["tagger_sig"] == "sig-old-2"
    assert second_row["last_tagged_at"] == pytest.approx(222.2)
    assert second_row["is_present"] == 1

    first_tag_rows = conn.execute(
        "SELECT tag_id, score FROM file_tags WHERE file_id = ?",
        (files["first"],),
    ).fetchall()
    assert len(first_tag_rows) == 2
    first_scores = {row["tag_id"]: row["score"] for row in first_tag_rows}
    assert tags["general:old"] not in first_scores
    assert first_scores[tags["character:kobato"]] == pytest.approx(0.95)
    assert first_scores[tags["rating:safe"]] == pytest.approx(0.75)

    second_tag_rows = conn.execute(
        "SELECT tag_id, score FROM file_tags WHERE file_id = ?",
        (files["second"],),
    ).fetchall()
    assert second_tag_rows == []

    first_match = conn.execute(
        "SELECT rowid FROM fts_files WHERE fts_files MATCH ?",
        ("kobato",),
    ).fetchone()
    assert first_match is not None
    assert first_match["rowid"] == files["first"]
