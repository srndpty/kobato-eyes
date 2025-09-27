"""Tests for the lightweight search parser."""

from __future__ import annotations

from core.search_parser import parse_search
from tagger.base import TagCategory


def test_parse_search_splits_include_and_exclude() -> None:
    result = parse_search("foo -bar half-closed_eyes")

    include = result["include"]
    exclude = result["exclude"]
    free = result["free"]

    assert [spec.name for spec in include] == ["foo", "half-closed_eyes"]
    assert [spec.name for spec in exclude] == ["bar"]
    assert free == []


def test_parse_search_handles_quoted_negative() -> None:
    result = parse_search('-"big-hair"')

    include = result["include"]
    exclude = result["exclude"]

    assert include == []
    assert [spec.name for spec in exclude] == ["big-hair"]


def test_parse_search_ignores_spaced_minus() -> None:
    result = parse_search("- big-hair")

    assert result["include"] == []
    assert result["exclude"] == []
    assert result["free"] == ["-", "big-hair"]


def test_parse_search_extracts_category_prefix() -> None:
    result = parse_search("artist:john")

    include = result["include"]
    assert len(include) == 1
    assert include[0].name == "artist:john"
    assert include[0].category == TagCategory.ARTIST


def test_parse_search_retains_unknown_prefix() -> None:
    result = parse_search("prefix:value")

    include = result["include"]
    assert len(include) == 1
    assert include[0].name == "prefix:value"
    assert include[0].category is None
