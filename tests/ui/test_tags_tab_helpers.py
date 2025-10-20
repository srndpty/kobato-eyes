"""Unit tests for helper functions in ``ui.tags_tab``."""

from __future__ import annotations

import pytest

from tagger.base import TagCategory
from ui.tags_tab import _filter_tags_by_threshold


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        pytest.param(
            [{"name": "cute", "score": "0.15"}],
            [("cute", 0.15, None)],
            id="dict-input",
        ),
        pytest.param(
            [("tuple", 0.2)],
            [("tuple", 0.2, None)],
            id="tuple-two",
        ),
        pytest.param(
            [("triple", 0.3, TagCategory.GENERAL)],
            [("triple", 0.3, TagCategory.GENERAL)],
            id="tuple-three",
        ),
        pytest.param(
            [("too", 0.2, "many", "values")],
            [],
            id="invalid-length",
        ),
        pytest.param(
            [("low", 0.05), ("edge", 0.1)],
            [("edge", 0.1, None)],
            id="threshold-boundary",
        ),
        pytest.param(
            [(123, 0.25)],
            [("123", 0.25, None)],
            id="name-coercion",
        ),
        pytest.param(
            [("str-cat", 0.2, "character")],
            [("str-cat", 0.2, TagCategory.CHARACTER)],
            id="string-category",
        ),
    ],
)
def test_filter_tags_by_threshold(rows, expected) -> None:
    assert _filter_tags_by_threshold(rows) == expected
