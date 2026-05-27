"""Tests for repository search helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.query import translate_query
from db.connection import get_conn
from db.repository import search_files
from db.schema import apply_schema
from tagger.base import TagCategory


@pytest.fixture()
def conn():
    connection = get_conn(":memory:")
    apply_schema(connection)
    yield connection
    connection.close()


def _insert_tag(connection, name: str, score: float, file_id: int, *, category: int = 0) -> None:
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
    for _idx, (name, score) in enumerate(tags):
        _insert_tag(conn, name, score, file_a)

    _insert_tag(conn, "landscape", 0.4, file_b)

    where_sql = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id WHERE ft.file_id = f.id AND t.name = ?)"
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


@pytest.mark.parametrize(
    ("order_by", "expected_paths"),
    [
        ("f.mtime DESC", ["B.png", "A.png"]),
        ("f.mtime ASC", ["A.png", "B.png"]),
        ("f.path ASC", ["A.png", "B.png"]),
        ("f.path DESC", ["B.png", "A.png"]),
        ("f.id ASC", ["A.png", "B.png"]),
        ("f.id DESC", ["B.png", "A.png"]),
    ],
)
def test_search_files_accepts_only_known_order_by_clauses(conn, order_by: str, expected_paths: list[str]) -> None:
    _insert_file(conn, path="A.png", size=100, mtime=10.0, sha="a-order")
    _insert_file(conn, path="B.png", size=100, mtime=20.0, sha="b-order")

    results = search_files(conn, "1=1", [], order_by=order_by)

    assert [row["path"] for row in results] == expected_paths


def test_search_files_rejects_unsupported_order_by(conn) -> None:
    _insert_file(conn, path="unsafe.png", size=100, mtime=50.0, sha="unsafe")

    with pytest.raises(ValueError, match="Unsupported search order_by"):
        search_files(conn, "1=1", [], order_by="f.mtime ASC; DROP TABLE files")


def test_search_files_relevance_order_requires_tag_terms(conn) -> None:
    file_old = _insert_file(conn, path="old.png", size=100, mtime=10.0, sha="old")
    file_new = _insert_file(conn, path="new.png", size=100, mtime=20.0, sha="new")
    _insert_tag(conn, "1girl", 0.99, file_old)
    _insert_tag(conn, "1girl", 0.10, file_new)

    without_terms = search_files(conn, "1=1", [], order="relevance", tags_for_relevance=[])
    with_terms = search_files(conn, "1=1", [], order="relevance", tags_for_relevance=["1girl"])

    assert [row["id"] for row in without_terms] == [file_new, file_old]
    assert [row["id"] for row in with_terms] == [file_old, file_new]
    assert with_terms[0]["relevance"] == pytest.approx(0.99)


def test_translate_query_treats_leading_dash_as_negation_not_literal_tag(conn) -> None:
    safe_id = _insert_file(conn, path="safe.png", size=100, mtime=20.0, sha="safe")
    other_id = _insert_file(conn, path="other.png", size=100, mtime=10.0, sha="other")
    _insert_tag(conn, "rating:safe", 0.99, safe_id)
    _insert_tag(conn, "-rating:safe", 0.99, other_id)

    fragment = translate_query("-rating:safe", file_alias="f")
    results = search_files(conn, fragment.where, fragment.params)

    assert [row["id"] for row in results] == [other_id]


def test_translate_query_treats_whitespace_as_term_separator_not_literal_tag(conn) -> None:
    literal_id = _insert_file(conn, path="literal-space.png", size=100, mtime=10.0, sha="literal-space")
    _insert_tag(conn, "white hair", 0.99, literal_id)

    fragment = translate_query("white hair", file_alias="f")
    results = search_files(conn, fragment.where, fragment.params)

    assert results == []


def test_translate_query_integrates_with_search(conn) -> None:
    file_id = _insert_file(conn, path="E.png", size=111, mtime=10.0, sha="e")
    _insert_tag(conn, "1girl", 0.88, file_id)

    fragment = translate_query("1girl", file_alias="f")
    results = search_files(conn, fragment.where, fragment.params)

    assert [row["id"] for row in results] == [file_id]


def test_search_files_includes_tag_categories(conn) -> None:
    file_id = _insert_file(conn, path="F.png", size=222, mtime=20.0, sha="f")

    _insert_tag(conn, "1girl", 0.95, file_id, category=0)
    _insert_tag(conn, "original_character", 0.85, file_id, category=TagCategory.CHARACTER.value)

    where_sql = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id WHERE ft.file_id = f.id AND t.name = ?)"
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
    assert record["tags"][1][2] == TagCategory.CHARACTER.value


def test_search_files_relevance_uses_character_threshold(conn) -> None:
    file_low = _insert_file(conn, path="character-low.png", size=100, mtime=30.0, sha="character-low")
    file_high = _insert_file(conn, path="character-high.png", size=100, mtime=20.0, sha="character-high")
    _insert_tag(conn, "alice", 0.70, file_low, category=TagCategory.CHARACTER.value)
    _insert_tag(conn, "alice", 0.85, file_high, category=TagCategory.CHARACTER.value)

    results = search_files(
        conn,
        "1=1",
        [],
        order="relevance",
        tags_for_relevance=["alice"],
        thresholds={TagCategory.CHARACTER.value: 0.8},
    )

    assert [row["id"] for row in results] == [file_high, file_low]
    assert results[0]["relevance"] == pytest.approx(0.85)
    assert results[1]["relevance"] == pytest.approx(0.0)


def test_search_files_excludes_missing_records(conn) -> None:
    file_id = _insert_file(conn, path="G.png", size=300, mtime=30.0, sha="g")
    _insert_tag(conn, "1girl", 0.95, file_id)
    conn.execute("UPDATE files SET is_present = 0, deleted_at = CURRENT_TIMESTAMP WHERE id = ?", (file_id,))

    results = search_files(conn, "1=1", [])

    assert results == []
