"""Tests mapping tag queries to SQL fragments for UI consumption."""

from __future__ import annotations

from core.query import translate_query


def test_translate_query_single_token() -> None:
    fragment = translate_query("1girl")
    assert "ft.file_id = f.id" in fragment.where
    assert "t.name" in fragment.where
    assert fragment.params == ["1girl"]


def test_translate_query_empty_returns_trivial_fragment() -> None:
    fragment = translate_query("")
    assert fragment.where == "1=1"
    assert fragment.params == []
