"""Tests for helpers defined in :mod:`db.common`."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db.common import chunk, fts_is_contentless, load_tag_thresholds, normalise_category
from db.connection import get_conn
from db.schema import apply_schema


@pytest.fixture()
def memory_conn() -> sqlite3.Connection:
    """Provide an in-memory SQLite database initialised with the application schema."""

    conn = get_conn(":memory:")
    apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0, 0, id="int-zero"),
        pytest.param(1.9, 1, id="float"),
        pytest.param(True, 1, id="bool"),
        pytest.param("general", 0, id="category-name"),
        pytest.param(" Artist ", 4, id="category-name-with-spaces"),
        pytest.param(" 3 ", 3, id="numeric-string"),
        pytest.param("   ", None, id="blank-string"),
        pytest.param("unknown", None, id="invalid-string"),
        pytest.param(None, None, id="none"),
    ],
)
def test_normalise_category_returns_expected(value: object, expected: int | None) -> None:
    """``normalise_category`` should convert diverse inputs to expected codes."""

    assert normalise_category(value) == expected


def test_load_tag_thresholds_returns_defaults_when_table_empty(memory_conn: sqlite3.Connection) -> None:
    """When the tagger thresholds table is empty the defaults should be returned."""

    memory_conn.execute("DELETE FROM tagger_thresholds")
    memory_conn.commit()

    thresholds = load_tag_thresholds(memory_conn)

    assert thresholds == {0: 0.35, 1: 0.25, 3: 0.25}


def test_load_tag_thresholds_overrides_defaults_with_database_values(
    memory_conn: sqlite3.Connection,
) -> None:
    """Values stored in the database should override or extend the defaults."""

    memory_conn.execute("DELETE FROM tagger_thresholds")
    memory_conn.executemany(
        "INSERT INTO tagger_thresholds(category, threshold) VALUES(?, ?)",
        [
            ("general", 0.5),
            ("character", 0.4),
            ("5", 0.1),
        ],
    )
    memory_conn.commit()

    thresholds = load_tag_thresholds(memory_conn)

    assert thresholds[0] == pytest.approx(0.5)
    assert thresholds[1] == pytest.approx(0.4)
    assert thresholds[3] == pytest.approx(0.25)
    assert thresholds[5] == pytest.approx(0.1)


def test_fts_is_contentless_detects_contentless_table(memory_conn: sqlite3.Connection) -> None:
    """The helper should detect FTS tables created with ``content=''``."""

    memory_conn.execute("DROP TABLE IF EXISTS fts_files")
    memory_conn.execute(
        "CREATE VIRTUAL TABLE fts_files USING fts5(text, content='', tokenize='unicode61')"
    )

    assert fts_is_contentless(memory_conn) is True


def test_fts_is_contentless_detects_standard_table(memory_conn: sqlite3.Connection) -> None:
    """The helper should return ``False`` for standard content-backed FTS tables."""

    memory_conn.execute("DROP TABLE IF EXISTS fts_files")
    memory_conn.execute("CREATE VIRTUAL TABLE fts_files USING fts5(text)")

    assert fts_is_contentless(memory_conn) is False


def test_chunk_splits_sequence_into_requested_sizes() -> None:
    """``chunk`` should split sequences into sub-sequences of the requested size."""

    chunks = list(chunk([1, 2, 3, 4, 5], 2))

    assert chunks == [[1, 2], [3, 4], [5]]
