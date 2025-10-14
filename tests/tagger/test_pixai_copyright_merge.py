"""Tests for PixAI copyright merging utilities."""

from __future__ import annotations

from tagger.base import TagCategory, TagPrediction
from tagger.pixai_onnx import merge_character_ips


def test_merge_adds_copyrights_for_character_tags() -> None:
    preds = [TagPrediction(name="hero", score=0.9, category=TagCategory.CHARACTER)]
    merged = merge_character_ips(preds, {"hero": ["series_a", "series_b"]})
    names = [tag.name for tag in merged]
    assert names == ["hero", "series_a", "series_b"]
    assert all(tag.category == TagCategory.COPYRIGHT for tag in merged[1:])


def test_merge_ignores_non_character_tags() -> None:
    preds = [
        TagPrediction(name="scenery", score=0.8, category=TagCategory.GENERAL),
        TagPrediction(name="alice", score=0.7, category=TagCategory.CHARACTER),
    ]
    merged = merge_character_ips(preds, {"alice": ["wonderland"]})
    assert [tag.name for tag in merged] == ["scenery", "alice", "wonderland"]
    assert merged[-1].category == TagCategory.COPYRIGHT


def test_merge_skips_duplicates_and_empty_names() -> None:
    preds = [
        TagPrediction(name="", score=0.5, category=TagCategory.CHARACTER),
        TagPrediction(name="hero", score=0.9, category=TagCategory.CHARACTER),
        TagPrediction(name="hero", score=0.8, category=TagCategory.CHARACTER),
        TagPrediction(name="series_x", score=0.4, category=TagCategory.COPYRIGHT),
    ]
    merged = merge_character_ips(
        preds,
        {"hero": ["series_x", "series_y", ""]},
    )
    names = [tag.name for tag in merged]
    assert names == ["hero", "series_x", "series_y"]
