"""Tests for the tagging job pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

import pytest
from PIL import Image

from core.tag_job import TagJobConfig, run_tag_job
from db.connection import get_conn
from db.schema import apply_schema
from tagger.base import ITagger, TagCategory, TagPrediction, TagResult


class DummyTagger(ITagger):
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict | None, dict | None]] = []

    def infer_batch(
        self,
        images: Sequence[Image.Image],
        *,
        thresholds=None,
        max_tags=None,
    ) -> list[TagResult]:
        self.calls.append((len(images), dict(thresholds or {}), dict(max_tags or {})))
        predictions = [
            TagPrediction(
                name="character:kobato", score=0.9, category=TagCategory.CHARACTER
            ),
            TagPrediction(name="rating:safe", score=0.95, category=TagCategory.GENERAL),
        ]
        return [TagResult(tags=predictions)]


def _make_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), color=(0, 255, 0))
    image.save(path, format="PNG")


@pytest.fixture()
def memory_conn(tmp_path: Path) -> Iterable[sqlite3.Connection]:
    conn = get_conn(":memory:")
    apply_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def test_run_tag_job_persists_predictions(
    memory_conn: sqlite3.Connection, tmp_path: Path
) -> None:
    source = tmp_path / "image.png"
    _make_image(source)

    tagger = DummyTagger()
    config = TagJobConfig(
        thresholds={TagCategory.GENERAL: 0.5},
        max_tags={TagCategory.GENERAL: 10},
    )

    output = run_tag_job(tagger, source, memory_conn, config=config)
    assert output is not None
    assert output.file_id > 0
    assert tagger.calls
    _, recorded_thresholds, recorded_max = tagger.calls[-1]
    assert recorded_thresholds == {TagCategory.GENERAL: 0.5}
    assert recorded_max == {TagCategory.GENERAL: 10}

    file_row = memory_conn.execute(
        "SELECT size, sha256 FROM files WHERE id = ?",
        (output.file_id,),
    ).fetchone()
    assert file_row is not None and file_row["sha256"]

    tag_rows = memory_conn.execute(
        "SELECT name, category FROM tags ORDER BY name",
    ).fetchall()
    assert {(row["name"], row["category"]) for row in tag_rows} == {
        ("character:kobato", TagCategory.CHARACTER),
        ("rating:safe", TagCategory.GENERAL),
    }

    tag_scores = memory_conn.execute(
        "SELECT score FROM file_tags ORDER BY score DESC",
    ).fetchall()
    assert len(tag_scores) == 2
    assert tag_scores[0]["score"] >= tag_scores[1]["score"]

    fts_row = memory_conn.execute(
        "SELECT text FROM fts_files WHERE rowid = ?",
        (output.file_id,),
    ).fetchone()
    assert fts_row is not None
    assert "character:kobato" in fts_row["text"]


def test_run_tag_job_returns_none_for_missing(memory_conn: sqlite3.Connection) -> None:
    tagger = DummyTagger()
    result = run_tag_job(tagger, Path("/nonexistent.png"), memory_conn)
    assert result is None
    assert not tagger.calls
