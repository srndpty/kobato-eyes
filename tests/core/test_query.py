import pytest

from core.query import QueryFragment, parse_search, translate_query

ALIAS = "f"


def expected_tag_exists(alias: str = ALIAS) -> str:
    return (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        f"WHERE ft.file_id = {alias}.id AND t.name = ?)"
    )


def expected_tag_exists_with_category(alias: str = ALIAS) -> str:
    return (
        "EXISTS (SELECT 1 FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
        f"WHERE ft.file_id = {alias}.id AND t.category = ? AND t.name = ?)"
    )


def test_single_tag_translates_to_exists() -> None:
    fragment = translate_query("kobato", file_alias=ALIAS)
    assert fragment.where == expected_tag_exists()
    assert fragment.params == ["kobato"]


def test_negative_prefix_translates_to_not_exists() -> None:
    fragment = translate_query("-hatsune_miku", file_alias=ALIAS)
    assert fragment.where == f"NOT {expected_tag_exists()}"
    assert fragment.params == ["hatsune_miku"]


def test_not_keyword_translates_to_exclusion() -> None:
    fragment = translate_query("megurine_luka NOT hatsune_miku", file_alias=ALIAS)
    expected = f"{expected_tag_exists()} AND NOT {expected_tag_exists()}"
    assert fragment.where == expected
    assert fragment.params == ["megurine_luka", "hatsune_miku"]


def test_category_prefix_maps_to_numeric_category() -> None:
    fragment = translate_query("character:megurine_luka", file_alias=ALIAS)
    assert fragment.where == expected_tag_exists_with_category()
    assert fragment.params == [1, "megurine_luka"]


def test_unknown_category_treated_as_plain_tag() -> None:
    fragment = translate_query("pool:summer", file_alias=ALIAS)
    assert fragment.where == expected_tag_exists()
    assert fragment.params == ["pool:summer"]


@pytest.mark.parametrize(
    "token",
    [
        "don't_say_\"lazy\"",
        "!?",
        "cute_&_girly_(idolmaster)",
        "=_=",
        "^^^",
        "<|>_<|>",
        "@_@",
        ";)",
        ">:)",
        "st._gloriana's_military_uniform",
        "double_\\m/",
        "kaguya-sama_wa_kokurasetai_~tensai-tachi_no_renai_zunousen~",
        "otome_game_no_hametsu_flag_shika_nai_akuyaku_reijou_ni_tensei_shite_shimatta",
    ],
)
def test_special_character_tokens_do_not_break_translation(token: str) -> None:
    fragment = translate_query(token, file_alias=ALIAS)
    assert fragment.where == expected_tag_exists()
    assert fragment.params == [token]


def test_mixed_positive_and_negative_tokens() -> None:
    fragment = translate_query("megurine_luka -hatsune_miku", file_alias=ALIAS)
    expected = f"{expected_tag_exists()} AND NOT {expected_tag_exists()}"
    assert fragment.where == expected
    assert fragment.params == ["megurine_luka", "hatsune_miku"]


def test_parse_search_uses_unicode_whitespace() -> None:
    parsed = parse_search("miku\u3000luka\t rin")
    assert [spec.name for spec in parsed["include"]] == ["miku", "luka", "rin"]


def test_empty_query_returns_trivial_fragment() -> None:
    fragment = translate_query("", file_alias=ALIAS)
    assert fragment == QueryFragment(where="1=1", params=[])
