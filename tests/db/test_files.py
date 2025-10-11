"""Tests for helpers operating on the ``files`` table."""

from __future__ import annotations

import sqlite3
from typing import Iterator

import pytest

from db.connection import get_conn
from db.files import bulk_upsert_files_meta
from db.schema import apply_schema


@pytest.fixture()
def memory_conn() -> Iterator[sqlite3.Connection]:
    """Provide an in-memory SQLite database initialised with the application schema."""

    conn = get_conn(":memory:")
    apply_schema(conn)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@pytest.mark.parametrize("coalesce_wh", [True, False])
def test_bulk_upsert_inserts_new_rows(memory_conn: sqlite3.Connection, coalesce_wh: bool) -> None:
    """Rows are inserted when the target identifiers do not yet have metadata."""

    rows = [
        (1, 100, 200, "sig-0", 0.5),
        (2, 101, 201, "sig-1", 1.5),
    ]

    bulk_upsert_files_meta(memory_conn, rows, coalesce_wh=coalesce_wh, chunk=len(rows))

    stored = memory_conn.execute(
        "SELECT id, width, height, tagger_sig, last_tagged_at FROM files ORDER BY id",
    ).fetchall()
    fetched = [
        (
            row["id"],
            row["width"],
            row["height"],
            row["tagger_sig"],
            row["last_tagged_at"],
        )
        for row in stored
    ]
    assert fetched == rows


@pytest.mark.parametrize(
    ("coalesce_wh", "expected_width", "expected_height"),
    [
        (True, 320, 240),
        (False, None, None),
    ],
)
def test_bulk_upsert_coalesce_width_height(
    memory_conn: sqlite3.Connection,
    coalesce_wh: bool,
    expected_width: int | None,
    expected_height: int | None,
) -> None:
    """Width and height are either retained or overwritten based on ``coalesce_wh``."""

    file_id = 10
    memory_conn.execute(
        "INSERT INTO files(id, path, width, height, tagger_sig, last_tagged_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (file_id, "C:/images/existing.png", 320, 240, "sig-old", 21.0),
    )

    bulk_upsert_files_meta(
        memory_conn,
        [(file_id, None, None, "sig-new", 42.0)],
        coalesce_wh=coalesce_wh,
    )

    row = memory_conn.execute(
        "SELECT width, height, tagger_sig, last_tagged_at FROM files WHERE id = ?",
        (file_id,),
    ).fetchone()
    assert row is not None
    assert row["width"] == expected_width
    assert row["height"] == expected_height
    assert row["tagger_sig"] == "sig-new"
    assert row["last_tagged_at"] == pytest.approx(42.0)


def test_bulk_upsert_honours_chunk_size(memory_conn: sqlite3.Connection) -> None:
    """All rows are applied even when the operation is split into multiple chunks."""

    rows: list[tuple[int, int, int, str, float]] = []
    for index in range(7):
        file_id = index + 1
        rows.append((file_id, index * 10, index * 20, f"sig-{index}", float(index)))

    bulk_upsert_files_meta(memory_conn, rows, chunk=2)

    stored = memory_conn.execute(
        "SELECT width, height, tagger_sig, last_tagged_at FROM files ORDER BY id",
    ).fetchall()
    assert len(stored) == len(rows)
    for index, row in enumerate(stored):
        assert row["width"] == index * 10
        assert row["height"] == index * 20
        assert row["tagger_sig"] == f"sig-{index}"
        assert row["last_tagged_at"] == pytest.approx(float(index))
