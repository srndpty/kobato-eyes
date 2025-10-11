"""Tests for helpers in :mod:`core.pipeline.utils`."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.pipeline.utils import (
    _digest_identifier,
    _format_sig_mapping,
    _normalise_sig_source,
    _serialise_max_tags,
    _serialise_thresholds,
)
from tagger.base import TagCategory


def test_serialise_thresholds_and_max_tags() -> None:
    thresholds = {
        TagCategory.GENERAL: 0.5,
        TagCategory.ARTIST: 0.75,
    }
    max_tags = {
        TagCategory.RATING: 10,
        TagCategory.META: 3,
    }

    assert _serialise_thresholds(thresholds) == {
        "general": 0.5,
        "artist": 0.75,
    }
    assert _serialise_thresholds({}) == {}

    assert _serialise_max_tags(max_tags) == {
        "rating": 10,
        "meta": 3,
    }
    assert _serialise_max_tags(None) == {}


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        ({}, "none"),
        ({"beta": 1, "alpha": 2}, "alpha=2,beta=1"),
        ({"value": 1.23456789}, "value=1.234568"),
        ({"trailing": 1.230000}, "trailing=1.23"),
    ],
)
def test_format_sig_mapping(mapping: dict[str, float | int], expected: str) -> None:
    assert _format_sig_mapping(mapping) == expected


def test_normalise_sig_source_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    path_value = tmp_path / "as_path" / "file.bin"
    expected_path = str(path_value.resolve())
    assert _normalise_sig_source(path_value) == expected_path
    assert _digest_identifier(path_value) == hashlib.sha256(
        expected_path.encode("utf-8")
    ).hexdigest()

    relative_value = Path("relative") / "file.dat"
    expected_relative = str((tmp_path / relative_value).resolve())
    assert _normalise_sig_source(str(relative_value)) == expected_relative
    assert _digest_identifier(str(relative_value)) == hashlib.sha256(
        expected_relative.encode("utf-8")
    ).hexdigest()

    numeric_value = 123
    expected_numeric = str((tmp_path / str(numeric_value)).resolve())
    assert _normalise_sig_source(numeric_value) == expected_numeric
    assert _digest_identifier(numeric_value) == hashlib.sha256(
        expected_numeric.encode("utf-8")
    ).hexdigest()

    assert _normalise_sig_source(None) is None
    assert _digest_identifier(None) == "none"
