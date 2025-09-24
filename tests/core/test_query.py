"""Tests for simplified query to SQL translation."""

from __future__ import annotations

import pytest

from core.query import QueryFragment, translate_query


ALIAS = "f"


def expected_tag_exists(alias: str = ALIAS) -> str:
    return (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        f"WHERE ft.file_id = {alias}.id AND t.name = ?)"
    )


def _assert_query(sql: str, params: list[object], query: str) -> None:
    fragment = translate_query(query, file_alias=ALIAS)
    assert fragment.where == sql
    assert fragment.params == params


def test_single_tag_translates_to_exists() -> None:
    expected = expected_tag_exists()
    _assert_query(expected, ["kobato"], "kobato")


def test_or_expression() -> None:
    left = expected_tag_exists()
    right = left
    expected = f"({left}) OR ({right})"
    fragment = translate_query("kobato OR azusa", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["kobato", "azusa"]


def test_not_expression_with_implicit_and() -> None:
    tag_clause = expected_tag_exists()
    expected = f"({tag_clause}) AND (NOT ({tag_clause}))"
    fragment = translate_query("rating:safe NOT spoiler", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["rating:safe", "spoiler"]


def test_category_and_score_filters() -> None:
    category_clause = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        f"WHERE ft.file_id = {ALIAS}.id AND t.category = ?)"
    )
    score_clause = (
        "EXISTS (SELECT 1 FROM file_tags ft "
        f"WHERE ft.file_id = {ALIAS}.id AND ft.score > ? )"
    )
    expected = f"({category_clause}) AND ({score_clause})"
    fragment = translate_query("category:character score>0.75", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == [1, 0.75]


def test_parentheses_override_precedence() -> None:
    clause = expected_tag_exists()
    expected = f"({clause}) OR (({clause}) AND ({clause}))"
    fragment = translate_query("tag1 OR (tag2 tag3)", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["tag1", "tag2", "tag3"]


def test_invalid_category_raises() -> None:
    with pytest.raises(ValueError):
        translate_query("category:unknown", file_alias=ALIAS)


def test_invalid_trailing_token_raises() -> None:
    with pytest.raises(ValueError):
        translate_query("kobato AND", file_alias=ALIAS)


def test_empty_query_returns_trivial_fragment() -> None:
    fragment = translate_query("", file_alias=ALIAS)
    assert fragment == QueryFragment(where="1=1", params=[])
