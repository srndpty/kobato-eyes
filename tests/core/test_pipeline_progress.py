"""Tests for progress reporting in the indexing pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.config import PipelineSettings, TaggerSettings
from core.pipeline import IndexPhase, IndexProgress, run_index_once
from db.connection import get_conn
from db.schema import apply_schema
from tagger.dummy import DummyTagger

pytestmark = pytest.mark.not_gui


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
        index_dir=str(tmp_path / "index"),
    )

    events: list[IndexProgress] = []

    def _collect(progress: IndexProgress) -> None:
        events.append(progress)

    stats = run_index_once(
        temp_db,
        settings=settings,
        tagger_override=DummyTagger(),
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

    tag_events = [event for event in events if event.phase is IndexPhase.TAG]
    if stats["tagged"]:
        assert tag_events, "Expected TAG phase events when files are tagged"
        assert tag_events[-1].done == stats["tagged"]
    else:
        assert not tag_events

    fts_events = [event for event in events if event.phase is IndexPhase.FTS]
    if stats["signatures"]:
        assert fts_events, "Expected FTS phase events when rows are written"
        assert fts_events[-1].done == stats["signatures"]
    else:
        assert not fts_events
