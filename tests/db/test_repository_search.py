"""Tests for repository search helpers."""

from __future__ import annotations

import pytest

from core.query import translate_query
from db.connection import get_conn
from core.search_parser import parse_search
from db.repository import search_files, search_files_by_query
from db.schema import apply_schema


@pytest.fixture()
def conn():
    connection = get_conn(":memory:")
    apply_schema(connection)
    yield connection
    connection.close()


def _insert_tag(
    connection, name: str, score: float, file_id: int, *, category: int = 0
) -> None:
    cursor = connection.execute(
        "INSERT INTO tags (name, category) VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET category = excluded.category "
        "RETURNING id",
        (name, category),
    )
    tag_id = cursor.fetchone()[0]
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
    assert record["relevance"] == pytest.approx(0.0)
    assert record["mtime"] == pytest.approx(200.0)
    assert "tags" in record
    assert len(record["tags"]) == len(tags)
    tag_names = [name for name, _, _ in record["tags"]]
    assert tag_names == [name for name, _ in tags]
    for _, score, category in record["tags"]:
        assert isinstance(category, int) or category is None
        assert isinstance(score, float)
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


def test_search_files_includes_tag_categories(conn) -> None:
    file_id = _insert_file(conn, path="F.png", size=222, mtime=20.0, sha="f")

    _insert_tag(conn, "1girl", 0.95, file_id, category=0)
    _insert_tag(conn, "original_character", 0.85, file_id, category=1)

    where_sql = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.file_id = f.id AND t.name = ?)"
    )
    results = search_files(conn, where_sql, ["1girl"])

    assert len(results) == 1
    record = results[0]
    assert record["tags"] == record["top_tags"]
    assert record["tags"][0][0] == "1girl"
    assert record["tags"][0][1] == pytest.approx(0.95)
    assert record["tags"][0][2] == 0
    assert record["tags"][1][0] == "original_character"
    assert record["tags"][1][1] == pytest.approx(0.85)
    assert record["tags"][1][2] == 1


def test_search_files_by_query_supports_negative_tags(conn) -> None:
    file_a = _insert_file(conn, path="match.png", size=100, mtime=10.0, sha="a")
    file_b = _insert_file(conn, path="other.png", size=200, mtime=20.0, sha="b")

    _insert_tag(conn, "foo", 0.9, file_a)
    _insert_tag(conn, "half-closed_eyes", 0.8, file_a)

    _insert_tag(conn, "foo", 0.9, file_b)
    _insert_tag(conn, "bar", 0.7, file_b)

    terms = parse_search("foo -bar half-closed_eyes")
    results = search_files_by_query(conn, terms)

    assert [row["id"] for row in results] == [file_a]


def test_search_files_by_query_treats_spaced_minus_as_free(conn) -> None:
    file_a = _insert_file(conn, path="hair.png", size=128, mtime=30.0, sha="c")
    _insert_tag(conn, "big-hair", 0.95, file_a)

    terms = parse_search("- big-hair")
    results = search_files_by_query(conn, terms)

    assert [row["id"] for row in results] == [file_a]


def test_search_files_by_query_supports_not_keyword(conn) -> None:
    file_a = _insert_file(conn, path="vocaloid.png", size=512, mtime=50.0, sha="v")
    file_b = _insert_file(conn, path="duo.png", size=256, mtime=60.0, sha="d")

    _insert_tag(conn, "megurine_luka", 0.95, file_a)
    _insert_tag(conn, "hatsune_miku", 0.9, file_b)
    _insert_tag(conn, "megurine_luka", 0.9, file_b)

    terms = parse_search("megurine_luka NOT hatsune_miku")
    results = search_files_by_query(conn, terms)

    assert [row["id"] for row in results] == [file_a]
