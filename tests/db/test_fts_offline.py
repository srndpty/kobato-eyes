"""Tests for rebuilding the offline FTS index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from db.fts_offline import _truncate_fts, rebuild_fts_offline
from db.schema import ensure_schema


def _prepare_database(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)

    file_ids: dict[str, int] = {}
    for path, is_present in (
        ("present_a", 1),
        ("present_b", 1),
        ("missing_file", 0),
        ("blank_tag", 1),
        ("no_tags", 1),
    ):
        cur = conn.execute(
            "INSERT INTO files (path, is_present) VALUES (?, ?)",
            (path, is_present),
        )
        file_ids[path] = int(cur.lastrowid)

    tag_ids: dict[str, int] = {}
    for name in (
        "tag_alpha",
        "tag_beta",
        "tag_gamma",
        "tag_zeta",
        "tag_hidden",
        "   ",
    ):
        cur = conn.execute(
            "INSERT INTO tags (name) VALUES (?)",
            (name,),
        )
        tag_ids[name] = int(cur.lastrowid)

    conn.executemany(
        "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
        [
            (file_ids["present_a"], tag_ids["tag_alpha"], 0.9),
            (file_ids["present_a"], tag_ids["tag_beta"], 0.8),
            (file_ids["present_a"], tag_ids["tag_gamma"], 0.7),
            (file_ids["present_b"], tag_ids["tag_beta"], 0.6),
            (file_ids["present_b"], tag_ids["tag_zeta"], 0.4),
            (file_ids["missing_file"], tag_ids["tag_hidden"], 1.0),
            (file_ids["blank_tag"], tag_ids["   "], 0.5),
        ],
    )
    conn.commit()
    conn.close()


def _match_paths(conn: sqlite3.Connection, expression: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT f.path
        FROM fts_files AS fts
        JOIN files AS f ON f.id = fts.rowid
        WHERE fts_files MATCH ?
        ORDER BY f.path
        """,
        (expression,),
    ).fetchall()
    return {str(row["path"]) for row in rows}


def _fts_indexed_paths(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT f.path
        FROM fts_files AS fts
        JOIN files AS f ON f.id = fts.rowid
        ORDER BY f.path
        """,
    ).fetchall()
    return [str(row["path"]) for row in rows]


def test_rebuild_fts_offline_populates_index(tmp_path: Path) -> None:
    db_path = tmp_path / "fts_basic.db"
    _prepare_database(db_path)

    inserted = rebuild_fts_offline(db_path)
    assert inserted == 2

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        assert conn.execute("SELECT COUNT(*) FROM fts_files").fetchone()[0] == inserted
        assert _fts_indexed_paths(conn) == ["present_a", "present_b"]
        assert _match_paths(conn, '"tag_alpha tag_beta tag_gamma"') == {"present_a"}
        assert _match_paths(conn, '"tag_beta tag_zeta"') == {"present_b"}
        assert _match_paths(conn, '"tag_beta tag_alpha"') == set()
        assert _match_paths(conn, "tag_hidden") == set()
        assert _match_paths(conn, "tag_alpha") == {"present_a"}
    conn.close()


@pytest.mark.parametrize(
    ("topk", "batch", "expected_paths", "expected", "absent"),
    [
        (
            1,
            1,
            {"present_a", "present_b"},
            {"tag_alpha": {"present_a"}, "tag_beta": {"present_b"}},
            {"tag_gamma", "tag_zeta"},
        ),
        (
            2,
            2,
            {"present_a", "present_b"},
            {
                '"tag_alpha tag_beta"': {"present_a"},
                '"tag_beta tag_zeta"': {"present_b"},
                "tag_beta": {"present_a", "present_b"},
                "tag_zeta": {"present_b"},
            },
            {"tag_gamma"},
        ),
    ],
)
def test_rebuild_fts_offline_topk_batch_variants(
    tmp_path: Path,
    topk: int,
    batch: int,
    expected_paths: set[str],
    expected: dict[str, set[str]],
    absent: set[str],
) -> None:
    db_path = tmp_path / f"fts_topk_{topk}_{batch}.db"
    _prepare_database(db_path)

    inserted = rebuild_fts_offline(db_path, topk=topk, batch=batch)
    assert inserted == len(expected_paths)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    with conn:
        assert conn.execute("SELECT COUNT(*) FROM fts_files").fetchone()[0] == len(expected_paths)
        assert set(_fts_indexed_paths(conn)) == expected_paths
        for expression, paths in expected.items():
            assert _match_paths(conn, expression) == paths
        for expression in absent:
            assert _match_paths(conn, expression) == set()
    conn.close()


def test_truncate_fts_handles_missing_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "no_fts.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE example(id INTEGER PRIMARY KEY)")
    with conn:
        conn.execute("INSERT INTO example(id) VALUES (1)")

    # No FTS5 tables are defined; truncation should simply return.
    _truncate_fts(conn)

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    assert [str(row["name"]) for row in tables] == ["example"]
    conn.close()
