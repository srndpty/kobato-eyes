"""Tests for the lightweight search parser."""

from __future__ import annotations

from core.search_parser import identify_token_at, parse_search, strip_negative_prefix


def _names(specs) -> list[str]:
    return [spec.name for spec in specs]


def test_parse_search_splits_include_and_exclude() -> None:
    result = parse_search("foo -bar half-closed_eyes")

    include = result["include"]
    exclude = result["exclude"]

    assert _names(include) == ["foo", "half-closed_eyes"]
    assert _names(exclude) == ["bar"]
    assert result["free"] == []


def test_parse_search_supports_not_keyword() -> None:
    result = parse_search("megurine_luka NOT hatsune_miku")

    include = result["include"]
    exclude = result["exclude"]

    assert _names(include) == ["megurine_luka"]
    assert _names(exclude) == ["hatsune_miku"]


def test_parse_search_handles_quoted_negative() -> None:
    result = parse_search('-"big-hair"')

    assert result["include"] == []
    assert _names(result["exclude"]) == ["big-hair"]


def test_parse_search_treats_spaced_minus_as_free() -> None:
    result = parse_search("- big-hair")

    assert result["include"] == []
    assert result["exclude"] == []
    assert result["free"] == ["-", "big-hair"]


def test_parse_search_extracts_category_prefix() -> None:
    result = parse_search("artist:john")

    include = result["include"]
    assert len(include) == 1
    spec = include[0]
    assert spec.raw == "artist:john"
    assert spec.category == "artist"
    assert spec.name == "john"


def test_parse_search_leaves_unknown_prefix_as_raw() -> None:
    result = parse_search("prefix:value")

    include = result["include"]
    assert len(include) == 1
    spec = include[0]
    assert spec.raw == "prefix:value"
    assert spec.category is None
    assert spec.name == "prefix:value"


def test_identify_token_at_handles_various_positions() -> None:
    text = "foo -bar baz"
    start, end, token = identify_token_at(text, 6)
    assert (start, end, token) == (4, 8, "-bar")

    start, end, token = identify_token_at(text, 3)
    assert (start, end, token) == (3, 3, "")

    start, end, token = identify_token_at("-hatsune", 5)
    assert (start, end, token) == (0, 8, "-hatsune")


def test_strip_negative_prefix_variants() -> None:
    assert strip_negative_prefix("-hatsune_miku") == (True, "hatsune_miku", False)
    assert strip_negative_prefix("hatsune_miku") == (False, "hatsune_miku", False)
    assert strip_negative_prefix('-"big-hair"') == (True, "big-hair", True)
    assert strip_negative_prefix("-\"") == (True, "", True)
    assert strip_negative_prefix("\"quoted\"") == (False, "quoted", True)
