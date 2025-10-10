"""Tests for the processing pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

import pytest
from PIL import Image

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 core required", exc_type=ImportError)
from PyQt6.QtCore import QCoreApplication

from core.jobs import JobManager
from core.pipeline import ProcessingPipeline
from core.settings import PipelineSettings
from db.connection import get_conn
from db.schema import apply_schema
from tagger.base import ITagger, TagCategory, TagPrediction, TagResult

pytestmark = [pytest.mark.gui, pytest.mark.integration]


@pytest.fixture(scope="module")
def qapp() -> Iterable[QCoreApplication]:
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield app


class DummyTagger(ITagger):
    def infer_batch(
        self,
        images: Sequence[Image.Image],
        *,
        thresholds=None,
        max_tags=None,
    ) -> list[TagResult]:
        predictions = [TagPrediction(name="color:red", score=0.9, category=TagCategory.GENERAL)]
        return [TagResult(tags=predictions) for _ in images]


def _create_test_image(path: Path) -> None:
    Image.new("RGB", (16, 16), color=(200, 20, 20)).save(path, format="PNG")


def _wait_for_completion(manager: JobManager, app: QCoreApplication, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while manager.has_pending_jobs():
        app.processEvents()
        time.sleep(0.01)
        if time.time() >= deadline:
            raise TimeoutError("Jobs did not finish in time")


def test_pipeline_processes_paths(tmp_path: Path, qapp: QCoreApplication) -> None:
    db_path = tmp_path / "kobato.db"
    conn = get_conn(db_path)
    apply_schema(conn)
    conn.close()

    image_path = tmp_path / "sample.png"
    _create_test_image(image_path)

    tagger = DummyTagger()

    manager = JobManager(max_workers=1)
    pipeline = ProcessingPipeline(
        db_path=db_path,
        tagger=tagger,
        job_manager=manager,
        settings=PipelineSettings(),
    )

    pipeline.enqueue_path(image_path)
    _wait_for_completion(manager, qapp)

    conn = get_conn(db_path)
    tags = conn.execute("SELECT name FROM tags").fetchall()
    conn.close()

    assert any(row["name"] == "color:red" for row in tags)

    pipeline.shutdown()
