"""Tests for repository search helpers."""

from __future__ import annotations

import pytest

from core.query import translate_query
from db.connection import get_conn
from db.repository import search_files
from db.schema import apply_schema


@pytest.fixture()
def conn():
    connection = get_conn(":memory:")
    apply_schema(connection)
    yield connection
    connection.close()


def _insert_tag(connection, name: str, score: float, file_id: int) -> None:
    row = connection.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()
    if row is None:
        cursor = connection.execute("INSERT INTO tags (name, category) VALUES (?, 0)", (name,))
        tag_id = cursor.lastrowid
    else:
        tag_id = row["id"]
    connection.execute(
        "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
        (file_id, tag_id, score),
    )


def _insert_file(connection, *, path: str, size: int, mtime: float, sha: str) -> int:
    cursor = connection.execute(
        "INSERT INTO files (path, size, mtime, sha256) VALUES (?, ?, ?, ?)",
        (path, size, mtime, sha),
    )
    return cursor.lastrowid


def _set_threshold(connection, category: str, threshold: float) -> None:
    connection.execute(
        "INSERT INTO tagger_thresholds (category, threshold) VALUES (?, ?) "
        "ON CONFLICT(category) DO UPDATE SET threshold = excluded.threshold",
        (category, threshold),
    )


def test_search_files_returns_expected_record(conn) -> None:
    file_a = _insert_file(conn, path="A.png", size=123, mtime=200.0, sha="a")
    file_b = _insert_file(conn, path="B.png", size=456, mtime=100.0, sha="b")

    tags = [
        ("1girl", 0.95),
        ("solo", 0.9),
        ("smile", 0.8),
        ("long_hair", 0.7),
        ("dress", 0.6),
        ("outdoors", 0.5),
    ]
    for idx, (name, score) in enumerate(tags):
        _insert_tag(conn, name, score, file_a)

    _insert_tag(conn, "landscape", 0.4, file_b)

    where_sql = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id " "WHERE ft.file_id = f.id AND t.name = ?)"
    )
    results = search_files(conn, where_sql, ["1girl"])

    assert len(results) == 1
    record = results[0]
    assert record["id"] == file_a
    assert record["path"] == "A.png"
    assert record["width"] is None and record["height"] is None
    assert record["size"] == 123
    assert record["mtime"] == pytest.approx(200.0)
    assert "tags" in record
    assert len(record["tags"]) == len(tags)
    tag_names = [name for name, _ in record["tags"]]
    assert tag_names == [name for name, _ in tags]
    assert record["top_tags"] == record["tags"]


def test_search_files_order_limit_offset(conn) -> None:
    file_c = _insert_file(conn, path="C.png", size=100, mtime=50.0, sha="c")
    file_d = _insert_file(conn, path="D.png", size=101, mtime=60.0, sha="d")
    for fid in (file_c, file_d):
        _insert_tag(conn, "test", 0.5, fid)

    where_sql = "1=1"
    first = search_files(conn, where_sql, [], order_by="f.mtime ASC", limit=1)
    assert first[0]["id"] == file_c

    second = search_files(conn, where_sql, [], order_by="f.mtime ASC", limit=1, offset=1)
    assert second[0]["id"] == file_d


def test_translate_query_integrates_with_search(conn) -> None:
    file_id = _insert_file(conn, path="E.png", size=111, mtime=10.0, sha="e")
    _insert_tag(conn, "1girl", 0.88, file_id)

    fragment = translate_query("1girl", file_alias="f")
    results = search_files(conn, fragment.where, fragment.params)

    assert [row["id"] for row in results] == [file_id]


def test_search_files_filters_tags_below_threshold(conn) -> None:
    file_id = _insert_file(conn, path="F.png", size=222, mtime=20.0, sha="f")
    _set_threshold(conn, "general", 0.8)

    tags = [
        ("1girl", 0.95),
        ("smile", 0.81),
        ("long_hair", 0.79),
        ("outdoors", 0.4),
    ]
    for name, score in tags:
        _insert_tag(conn, name, score, file_id)

    where_sql = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.file_id = f.id AND t.name = ?)"
    )
    results = search_files(conn, where_sql, ["1girl"])

    assert len(results) == 1
    record = results[0]
    assert [name for name, _ in record["tags"]] == ["1girl", "smile"]
    assert record["tags"] == record["top_tags"]
