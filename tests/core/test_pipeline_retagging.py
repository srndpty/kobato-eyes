"""Regression tests for pipeline re-tagging behaviour."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from core.pipeline import current_tagger_sig, run_index_once
from core.settings import PipelineSettings, TaggerSettings
from db.connection import get_conn
from tagger.base import ITagger, TagCategory, TagPrediction, TagResult

pytestmark = pytest.mark.integration


class _StubTagger(ITagger):
    def __init__(self, label: str) -> None:
        self._label = label

    def infer_batch(self, images, *, thresholds=None, max_tags=None):  # type: ignore[override]
        predictions = [TagPrediction(name=self._label, score=0.9, category=TagCategory.GENERAL)]
        return [TagResult(tags=predictions) for _ in images]


def _create_test_image(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (16, 16), color=(128, 128, 128))
    image.save(target)


def _fetch_tags(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT t.name FROM file_tags ft JOIN tags t ON t.id = ft.tag_id").fetchall()
    return {str(row[0]) for row in rows}


def test_retagging_on_signature_change(tmp_path: Path) -> None:
    db_path = tmp_path / "library.db"
    image_path = tmp_path / "assets" / "sample.png"
    _create_test_image(image_path)

    settings_dummy = PipelineSettings(
        roots=[str(tmp_path / "assets")],
        tagger=TaggerSettings(name="dummy"),
    )

    first_stats = run_index_once(
        db_path,
        settings=settings_dummy,
        tagger_override=_StubTagger("initial_tag"),
    )
    assert int(first_stats.get("tagged", 0)) == 1

    conn = get_conn(db_path)
    try:
        row = conn.execute("SELECT tagger_sig, last_tagged_at FROM files LIMIT 1").fetchone()
        assert row is not None
        initial_sig = str(row["tagger_sig"])
        initial_timestamp = float(row["last_tagged_at"] or 0.0)
        assert initial_sig
        assert initial_timestamp > 0
        assert _fetch_tags(conn) == {"initial_tag"}
    finally:
        conn.close()

    model_path = tmp_path / "models" / "wd14.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"onnx")

    settings_wd14 = PipelineSettings(
        roots=[str(tmp_path / "assets")],
        tagger=TaggerSettings(name="wd14-onnx", model_path=str(model_path)),
    )

    second_stats = run_index_once(
        db_path,
        settings=settings_wd14,
        tagger_override=_StubTagger("retagged_tag"),
    )
    assert int(second_stats.get("tagged", 0)) == 1
    assert int(second_stats.get("retagged", 0)) == 1

    conn = get_conn(db_path)
    try:
        row = conn.execute("SELECT tagger_sig, last_tagged_at FROM files LIMIT 1").fetchone()
        assert row is not None
        new_sig = str(row["tagger_sig"])
        new_timestamp = float(row["last_tagged_at"] or 0.0)
        assert new_sig != initial_sig
        assert new_timestamp > initial_timestamp
        expected_sig = current_tagger_sig(settings_wd14)
        assert new_sig == expected_sig
        assert _fetch_tags(conn) == {"retagged_tag"}
    finally:
        conn.close()
