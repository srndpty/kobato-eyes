"""Tests for the processing pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 core required", exc_type=ImportError)
from PyQt6.QtCore import QCoreApplication

from core.jobs import JobManager
from core.pipeline import PipelineSettings, ProcessingPipeline
from core.settings import EmbedModel
from db.connection import get_conn
from db.schema import apply_schema
from dup.indexer import EmbedderProtocol
from index.hnsw import HNSWIndex
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


class DummyEmbedder(EmbedderProtocol):  # type: ignore[misc]
    @property
    def embedding_dim(self) -> int:
        return 4

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        vectors = []
        for image in images:
            arr = np.asarray(image.resize((2, 2))).astype(np.float32).flatten()
            if arr.size < self.embedding_dim:
                arr = np.pad(arr, (0, self.embedding_dim - arr.size))
            else:
                arr = arr[: self.embedding_dim]
            norm = np.linalg.norm(arr)
            if norm:
                arr /= norm
            vectors.append(arr.astype(np.float32))
        return np.vstack(vectors)


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
    embedder = DummyEmbedder()
    hnsw = HNSWIndex(space="cosine")
    hnsw.build(dim=4, max_elements=10)

    manager = JobManager(max_workers=1)
    pipeline = ProcessingPipeline(
        db_path=db_path,
        tagger=tagger,
        embedder=embedder,
        hnsw_index=hnsw,
        job_manager=manager,
        settings=PipelineSettings(embed_model=EmbedModel(name="dummy")),
    )

    pipeline.enqueue_path(image_path)
    _wait_for_completion(manager, qapp)

    conn = get_conn(db_path)
    tags = conn.execute("SELECT name FROM tags").fetchall()
    embeddings = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()

    assert any(row["name"] == "color:red" for row in tags)
    assert embeddings == 1
    assert hnsw.current_count == 1

    pipeline.shutdown()
