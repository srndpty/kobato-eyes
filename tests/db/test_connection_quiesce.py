"""Tests for database quiesce state management."""

from __future__ import annotations

import pytest

from db.connection import get_conn, quiesced


def test_quiesced_releases_after_exception(tmp_path) -> None:
    """Normal connections should be allowed again after a failing quiesced block."""

    db_path = tmp_path / "library.db"

    with pytest.raises(RuntimeError, match="boom"):
        with quiesced():
            with pytest.raises(RuntimeError, match="quiesced"):
                get_conn(db_path)
            raise RuntimeError("boom")

    conn = get_conn(db_path)
    try:
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row is not None
        assert int(row[0]) == 1
    finally:
        conn.close()
