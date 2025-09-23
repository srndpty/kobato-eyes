"""Tests for simplified query to SQL translation."""

from __future__ import annotations

import pytest

from core.query import QueryFragment, translate_query


def _assert_query(sql: str, params: list[object], query: str) -> None:
    fragment = translate_query(query)
    assert fragment.where == sql
    assert fragment.params == params


def test_single_tag_translates_to_exists() -> None:
    expected = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.file_id = files.id AND t.name = ?)"
    )
    _assert_query(expected, ["kobato"], "kobato")


def test_or_expression() -> None:
    left = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.file_id = files.id AND t.name = ?)"
    )
    right = left
    expected = f"({left}) OR ({right})"
    fragment = translate_query("kobato OR azusa")
    assert fragment.where == expected
    assert fragment.params == ["kobato", "azusa"]


def test_not_expression_with_implicit_and() -> None:
    tag_clause = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.file_id = files.id AND t.name = ?)"
    )
    expected = f"({tag_clause}) AND (NOT ({tag_clause}))"
    fragment = translate_query("rating:safe NOT spoiler")
    assert fragment.where == expected
    assert fragment.params == ["rating:safe", "spoiler"]


def test_category_and_score_filters() -> None:
    category_clause = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.file_id = files.id AND t.category = ?)"
    )
    score_clause = "EXISTS (SELECT 1 FROM file_tags ft WHERE ft.file_id = files.id AND ft.score > ? )"
    expected = f"({category_clause}) AND ({score_clause})"
    fragment = translate_query("category:character score>0.75")
    assert fragment.where == expected
    assert fragment.params == [1, 0.75]


def test_parentheses_override_precedence() -> None:
    clause = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        "WHERE ft.file_id = files.id AND t.name = ?)"
    )
    expected = f"({clause}) OR (({clause}) AND ({clause}))"
    fragment = translate_query("tag1 OR (tag2 tag3)")
    assert fragment.where == expected
    assert fragment.params == ["tag1", "tag2", "tag3"]


def test_invalid_category_raises() -> None:
    with pytest.raises(ValueError):
        translate_query("category:unknown")


def test_invalid_trailing_token_raises() -> None:
    with pytest.raises(ValueError):
        translate_query("kobato AND")


def test_empty_query_returns_trivial_fragment() -> None:
    fragment = translate_query("")
    assert fragment == QueryFragment(where="1=1", params=[])
