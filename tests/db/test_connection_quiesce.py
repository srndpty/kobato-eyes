"""Tests for database quiesce state management."""

from __future__ import annotations

import pytest

from db.connection import begin_quiesce, end_quiesce, get_conn, is_quiesced, quiesced


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


def test_nested_quiesce_blocks_until_outer_scope_exits(tmp_path) -> None:
    """Leaving an inner quiesce scope must not reopen normal connections."""

    db_path = tmp_path / "nested.db"

    with quiesced():
        assert is_quiesced()
        with quiesced():
            with pytest.raises(RuntimeError, match="quiesced"):
                get_conn(db_path)

        assert is_quiesced()
        with pytest.raises(RuntimeError, match="quiesced"):
            get_conn(db_path)

    assert not is_quiesced()
    conn = get_conn(db_path)
    conn.close()


def test_extra_end_quiesce_does_not_underflow() -> None:
    """A defensive extra cleanup call should not make the counter negative."""

    end_quiesce()
    assert not is_quiesced()

    begin_quiesce()
    assert is_quiesced()
    end_quiesce()
    end_quiesce()
    assert not is_quiesced()
