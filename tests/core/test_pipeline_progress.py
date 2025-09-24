"""Tests for progress reporting in the indexing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from PIL import Image

from core.pipeline import IndexPhase, IndexProgress, run_index_once
from core.settings import EmbedModel, PipelineSettings, TaggerSettings
from db.connection import get_conn
from db.schema import apply_schema
from tagger.dummy import DummyTagger

pytestmark = pytest.mark.not_gui


class TinyEmbedder:
    """Deterministic embedder for progress tests."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        vectors = []
        for image in images:
            arr = np.asarray(image.resize((2, 2))).astype(np.float32).flatten()
            if arr.size < self._dim:
                arr = np.pad(arr, (0, self._dim - arr.size))
            else:
                arr = arr[: self._dim]
            norm = np.linalg.norm(arr)
            if norm:
                arr /= norm
            vectors.append(arr.astype(np.float32))
        return np.vstack(vectors)


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "progress.db"
    conn = get_conn(db_path)
    apply_schema(conn)
    conn.close()
    return db_path


def _make_image(path: Path, colour: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (128, 128), color=colour)
    image.save(path, format="PNG")


def test_run_index_once_reports_progress(tmp_path: Path, temp_db: Path) -> None:
    images = [tmp_path / f"sample_{idx}.png" for idx in range(3)]
    for idx, image_path in enumerate(images):
        _make_image(image_path, (50 * idx, 20 * (idx + 1), 80))

    settings = PipelineSettings(
        roots=[str(tmp_path)],
        excluded=[],
        allow_exts=[".png"],
        batch_size=2,
        tagger=TaggerSettings(name="dummy"),
        embed_model=EmbedModel(name="dummy", device="cpu", dim=4),
        index_dir=str(tmp_path / "index"),
    )

    events: list[IndexProgress] = []

    def _collect(progress: IndexProgress) -> None:
        events.append(progress)

    stats = run_index_once(
        temp_db,
        settings=settings,
        tagger_override=DummyTagger(),
        embedder_override=TinyEmbedder(),
        progress_cb=_collect,
    )

    assert events, "Expected progress events to be emitted"
    phase_order = [event.phase for event in events]
    assert phase_order[-1] is IndexPhase.DONE

    for phase in {event.phase for event in events}:
        counts = [event.done for event in events if event.phase == phase]
        assert counts == sorted(counts), f"Non-monotonic progress for {phase}"

    scan_events = [event for event in events if event.phase is IndexPhase.SCAN]
    assert scan_events[0].total == -1
    assert any(event.total == stats["scanned"] for event in scan_events)

    assert any(event.phase is IndexPhase.TAG for event in events)
    assert any(event.phase is IndexPhase.EMBED for event in events)
    assert any(event.phase is IndexPhase.FTS for event in events)
    assert any(event.phase is IndexPhase.HNSW for event in events)
