"""Tests for relevance-based ordering in ``search_files``."""

from __future__ import annotations

import pytest

from db.connection import get_conn
from db.repository import search_files
from db.schema import apply_schema


@pytest.fixture()
def conn():
    connection = get_conn(":memory:")
    apply_schema(connection)
    yield connection
    connection.close()


def _insert_file(connection, *, path: str, size: int, mtime: float, sha: str) -> int:
    cursor = connection.execute(
        "INSERT INTO files (path, size, mtime, sha256) VALUES (?, ?, ?, ?)",
        (path, size, mtime, sha),
    )
    return int(cursor.lastrowid)


def _insert_tag(
    connection,
    name: str,
    score: float,
    file_id: int,
    *,
    category: int = 0,
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


def test_search_files_orders_by_relevance(conn) -> None:
    high = _insert_file(conn, path="high.png", size=1, mtime=10.0, sha="h")
    low = _insert_file(conn, path="low.png", size=1, mtime=20.0, sha="l")

    _insert_tag(conn, "haruhi", 0.9, high)
    _insert_tag(conn, "haruhi", 0.2, low)

    rows = search_files(
        conn,
        "1=1",
        [],
        tags_for_relevance=["haruhi"],
        thresholds={0: 0.0},
        order="relevance",
    )

    assert [row["id"] for row in rows] == [high, low]
    assert rows[0]["relevance"] == pytest.approx(0.9)
    assert rows[1]["relevance"] == pytest.approx(0.2)


def test_search_files_aggregates_scores_for_multiple_tags(conn) -> None:
    file_a = _insert_file(conn, path="a.png", size=1, mtime=5.0, sha="a")
    file_b = _insert_file(conn, path="b.png", size=1, mtime=6.0, sha="b")

    _insert_tag(conn, "haruhi", 0.8, file_a)
    _insert_tag(conn, "miku", 0.1, file_a)
    _insert_tag(conn, "haruhi", 0.5, file_b)
    _insert_tag(conn, "miku", 0.5, file_b)

    rows = search_files(
        conn,
        "1=1",
        [],
        tags_for_relevance=["haruhi", "miku"],
        thresholds={0: 0.0},
        order="relevance",
    )

    assert [row["id"] for row in rows] == [file_b, file_a]
    assert rows[0]["relevance"] == pytest.approx(1.0)
    assert rows[1]["relevance"] == pytest.approx(0.9)
