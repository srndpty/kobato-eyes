"""Tests for database quiesce state management."""

from __future__ import annotations

import threading

import pytest

from db.connection import begin_quiesce, end_quiesce, get_conn, is_quiesced, quiesced


@pytest.fixture(autouse=True)
def _reset_quiesce_state() -> None:
    """Keep the process-global quiesce counter from leaking between tests."""

    while is_quiesced():
        end_quiesce()
    yield
    while is_quiesced():
        end_quiesce()


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


def test_quiesce_releases_after_inner_scope_exception(tmp_path) -> None:
    """A nested exception must not strand the global quiesce counter."""

    db_path = tmp_path / "inner-exception.db"

    with pytest.raises(ValueError, match="inner"):
        with quiesced():
            with quiesced():
                raise ValueError("inner")

    assert not is_quiesced()
    conn = get_conn(db_path)
    conn.close()


@pytest.mark.db_stress
def test_quiesce_blocks_regular_connection_from_other_thread(tmp_path) -> None:
    """Normal connections should be rejected even when opened from another thread."""

    db_path = tmp_path / "threaded.db"
    errors: list[str] = []

    def _connect() -> None:
        try:
            get_conn(db_path)
        except RuntimeError as exc:
            errors.append(str(exc))

    with quiesced():
        thread = threading.Thread(target=_connect, daemon=True)
        thread.start()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert errors == ["DB is quiesced (UNSAFE fast mode active)"]


@pytest.mark.db_stress
def test_connection_allowed_when_quiesced_for_writer_path(tmp_path) -> None:
    """Dedicated writer paths can explicitly opt in during quiesce."""

    db_path = tmp_path / "writer.db"

    with quiesced():
        conn = get_conn(db_path, allow_when_quiesced=True)
        try:
            row = conn.execute("PRAGMA foreign_keys").fetchone()
        finally:
            conn.close()

    assert row is not None
    assert int(row[0]) == 1
