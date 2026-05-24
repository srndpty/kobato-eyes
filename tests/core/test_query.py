"""Tests for simplified query to SQL translation."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.query import QueryFragment, translate_query
from tagger.base import TagCategory

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


def test_vertical_bar_expression_matches_or() -> None:
    left = expected_tag_exists()
    right = left
    expected = f"({left}) OR ({right})"
    fragment = translate_query("kobato | azusa", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["kobato", "azusa"]


def test_not_expression_with_implicit_and() -> None:
    tag_clause = expected_tag_exists()
    expected = f"({tag_clause}) AND (NOT ({tag_clause}))"
    fragment = translate_query("rating:safe NOT spoiler", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["rating:safe", "spoiler"]


def test_hyphen_prefix_negates_tag() -> None:
    tag_clause = expected_tag_exists()
    expected = f"({tag_clause}) AND (NOT ({tag_clause}))"
    fragment = translate_query("rating:safe -^^^", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["rating:safe", "^^^"]


def test_hyphen_prefix_negates_group() -> None:
    tag_clause = expected_tag_exists()
    expected = f"NOT (({tag_clause}) OR ({tag_clause}))"
    fragment = translate_query("-(kobato OR azusa)", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["kobato", "azusa"]


def test_hyphen_prefix_negates_single_token_group() -> None:
    tag_clause = expected_tag_exists()
    expected = f"NOT ({tag_clause})"
    fragment = translate_query("-(kobato)", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["kobato"]


def test_category_and_score_filters() -> None:
    category_clause = (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        f"WHERE ft.file_id = {ALIAS}.id AND t.category = ?)"
    )
    score_clause = f"EXISTS (SELECT 1 FROM file_tags ft WHERE ft.file_id = {ALIAS}.id AND ft.score > ? )"
    expected = f"({category_clause}) AND ({score_clause})"
    fragment = translate_query("category:character score>0.75", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == [TagCategory.CHARACTER.value, 0.75]


def test_parentheses_override_precedence() -> None:
    clause = expected_tag_exists()
    expected = f"({clause}) OR (({clause}) AND ({clause}))"
    fragment = translate_query("tag1 OR (tag2 tag3)", file_alias=ALIAS)
    assert fragment.where == expected
    assert fragment.params == ["tag1", "tag2", "tag3"]


def test_invalid_category_raises() -> None:
    with pytest.raises(ValueError):
        translate_query("category:unknown", file_alias=ALIAS)


def test_tag_with_parentheses_is_single_token() -> None:
    clause = expected_tag_exists()
    fragment = translate_query("mallow_(pokemon)", file_alias=ALIAS)
    assert fragment.where == clause
    assert fragment.params == ["mallow_(pokemon)"]


@pytest.mark.parametrize(
    "tag",
    [
        'don\'t_say_"lazy"',
        "!?",
        "tag",
        "cute_&_girly_(idolmaster)",
        "=_=",
        "^^^",
        "<|>_<|>",
        "@_@",
        "(:3",
        ";)",
        ">:)",
        "(leading_paren_tag",
        "st._gloriana's_military_uniform",
        r"double_\m/",
        "kaguya-sama_wa_kokurasetai_~tensai-tachi_no_renai_zunousen~",
        "otome_game_no_hametsu_flag_shika_nai_akuyaku_reijou_ni_tensei_shite_shimatta",
    ],
)
def test_special_character_tags_are_single_tokens(tag: str) -> None:
    clause = expected_tag_exists()
    fragment = translate_query(tag, file_alias=ALIAS)
    assert fragment.where == clause
    assert fragment.params == [tag]


def test_tag_with_colons_is_treated_as_tag() -> None:
    clause = expected_tag_exists()
    fragment = translate_query("artist:name:with:colon", file_alias=ALIAS)
    assert fragment.where == clause
    assert fragment.params == ["artist:name:with:colon"]


def test_invalid_trailing_token_raises() -> None:
    with pytest.raises(ValueError):
        translate_query("kobato AND", file_alias=ALIAS)


def test_empty_query_returns_trivial_fragment() -> None:
    fragment = translate_query("", file_alias=ALIAS)
    assert fragment == QueryFragment(where="1=1", params=[])


_DANBOORU_TAGS = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="_:.",
    ),
    min_size=1,
    max_size=24,
).filter(
    lambda value: not value.startswith(("-", "(", ")"))
    and not value.lower().startswith("category:")
    and value.upper() not in {"AND", "OR", "NOT"}
)


@given(_DANBOORU_TAGS, _DANBOORU_TAGS, st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_danbooru_query_combinations_do_not_raise(tag_a: str, tag_b: str, score: float) -> None:
    query = f"{tag_a} OR (-{tag_b} category:general score>={score:.3f})"

    fragment = translate_query(query, file_alias=ALIAS)

    assert " OR " in fragment.where
    assert tag_a in fragment.params
    assert tag_b in fragment.params
