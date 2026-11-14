"""Regression tests for pipeline re-tagging behaviour."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.config import PipelineSettings, TaggerSettings
from core.pipeline import IndexPipeline, current_tagger_sig, run_index_once
from core.pipeline.contracts import DBItem
from core.pipeline.stages.scan_stage import ScanStageResult
from core.pipeline.stages.tag_stage import TagStageResult
from core.pipeline.stages.write_stage import WriteStageResult
from core.pipeline.types import _FileRecord
from db.connection import get_conn
from tagger.base import ITagger, TagCategory, TagPrediction, TagResult

pytestmark = pytest.mark.integration


class _StubTagger(ITagger):
    def __init__(self, label: str) -> None:
        self._label = label

    def prepare_batch_from_rgb_np(self, images):  # type: ignore[override]
        return np.stack([np.asarray(image, dtype=np.float32) for image in images], axis=0)

    def infer_batch_prepared(self, batch, *, thresholds=None, max_tags=None):  # type: ignore[override]
        predictions = [TagPrediction(name=self._label, score=0.9, category=TagCategory.GENERAL)]
        return [TagResult(tags=predictions) for _ in range(len(batch))]

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


def _record(file_id: int, path: Path, *, is_new: bool, changed: bool, tag_exists: bool) -> _FileRecord:
    return _FileRecord(
        file_id=file_id,
        path=path,
        size=1,
        mtime=1.0,
        sha=f"sha-{file_id}",
        is_new=is_new,
        changed=changed,
        tag_exists=tag_exists,
        needs_tagging=True,
    )


def test_retagging_on_signature_change(tmp_path: Path) -> None:
    db_path = tmp_path / "library.db"
    image_path = tmp_path / "assets" / "sample.png"
    _create_test_image(image_path)

    settings_dummy = PipelineSettings(
        roots=[str(tmp_path / "assets")],
        excluded=[],
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
        excluded=[],
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


def test_retagged_count_uses_processed_existing_file_ids(tmp_path: Path) -> None:
    settings = PipelineSettings(roots=[str(tmp_path)], excluded=[], tagger=TaggerSettings(name="dummy"))
    new_record = _record(1, tmp_path / "new.png", is_new=True, changed=False, tag_exists=False)
    stale_record = _record(2, tmp_path / "stale.png", is_new=False, changed=False, tag_exists=True)

    class ScanStage:
        def run(self, ctx, emitter):
            return ScanStageResult(records=[new_record, stale_record], scanned=2, new_or_changed=1)

    class TagStage:
        def run(self, ctx, emitter, records):
            item = DBItem(1, [("new_tag", 0.9, int(TagCategory.GENERAL))], None, None, ctx.tagger_sig, 1.0)
            return TagStageResult(records=records, db_items=[item], tagged_count=1)

    class WriteStage:
        def run(self, ctx, emitter, tag_result):
            return WriteStageResult(written=len(tag_result.db_items), fts_processed=len(tag_result.db_items))

    pipeline = IndexPipeline(tmp_path / "library.db", settings=settings, tagger_override=_StubTagger("unused"))
    pipeline.set_stage_override("scan", ScanStage())
    pipeline.set_stage_override("tag", TagStage())
    pipeline.set_stage_override("write", WriteStage())

    stats = pipeline.run()

    assert stats["tagged"] == 1
    assert stats["retagged"] == 0


def test_v1_categories_saved_and_searchable(tmp_path: Path) -> None:
    """All six categories reach DB/FTS and category:style finds the indexed image."""
    from core.query import translate_query

    class SixCategoryTagger(_StubTagger):
        preserve_source_rgb = True
        preprocess_version = "integration-v1"

        def infer_batch_prepared(self, batch, *, thresholds=None, max_tags=None):
            categories = [
                TagCategory.GENERAL,
                TagCategory.CHARACTER,
                TagCategory.COPYRIGHT,
                TagCategory.STYLE,
                TagCategory.META,
                TagCategory.RATING,
            ]
            return [TagResult([TagPrediction(f"{c.name.lower()}_tag", 0.9, c) for c in categories]) for _ in batch]

    image = tmp_path / "images" / "sample.png"
    _create_test_image(image)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"v1-fixture")
    settings = PipelineSettings(
        roots=[str(image.parent)], excluded=[], tagger=TaggerSettings(name="pixai-v1-onnx", model_path=str(model))
    )
    db = tmp_path / "library.db"
    stats = run_index_once(db, settings=settings, tagger_override=SixCategoryTagger("unused"))
    assert int(stats.get("tagged", 0)) == 1
    conn = get_conn(db)
    try:
        assert len(_fetch_tags(conn)) == 6
        fragment = translate_query("category:style style_tag")
        assert conn.execute(f"SELECT f.id FROM files f WHERE {fragment.where}", fragment.params).fetchone()
        assert conn.execute("SELECT rowid FROM fts_files WHERE fts_files MATCH 'style_tag'").fetchone()
        before = conn.execute("SELECT tagger_sig FROM files").fetchone()[0]
    finally:
        conn.close()
    model.write_bytes(b"v1-updated")
    stats = run_index_once(db, settings=settings, tagger_override=SixCategoryTagger("unused"))
    assert int(stats.get("retagged", 0)) == 1
    conn = get_conn(db)
    try:
        assert conn.execute("SELECT tagger_sig FROM files").fetchone()[0] != before
    finally:
        conn.close()


def test_v1_invalid_explicit_labels_fail_even_when_nothing_needs_retagging(tmp_path: Path) -> None:
    """A tagged DB must not let a bad explicit labels_json finish as a successful no-op index."""
    image = tmp_path / "images" / "sample.png"
    _create_test_image(image)
    model = tmp_path / "model.onnx"
    model.write_bytes(b"v1-fixture")
    settings = PipelineSettings(
        roots=[str(image.parent)], excluded=[], tagger=TaggerSettings(name="pixai-v1-onnx", model_path=str(model))
    )
    db = tmp_path / "library.db"
    tagger = _StubTagger("tag")
    assert int(run_index_once(db, settings=settings, tagger_override=tagger).get("tagged", 0)) == 1
    assert int(run_index_once(db, settings=settings, tagger_override=tagger).get("tagged", 0)) == 0

    settings.tagger.labels_json = str(tmp_path / "custom.json")
    with pytest.raises(ValueError, match="labels not found"):
        run_index_once(db, settings=settings, tagger_override=tagger)
