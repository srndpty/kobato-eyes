"""Tests for the one-shot indexing pipeline."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from core.config import PipelineSettings, TaggerSettings
from core.pipeline import run_index_once
from db.connection import get_conn
from db.schema import apply_schema
from tagger.dummy import DummyTagger

pytestmark = pytest.mark.integration


@pytest.fixture()
def temp_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "kobato.db"
    conn = get_conn(db_path)
    apply_schema(conn)
    conn.close()
    return db_path


def _make_image(path: Path, colour: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (256, 256), color=colour)
    image.save(path, format="PNG")


def test_run_index_once_processes_images(tmp_path: Path, temp_db: Path) -> None:
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    _make_image(img1, (200, 20, 20))
    _make_image(img2, (20, 200, 20))

    settings = PipelineSettings(
        roots=[str(tmp_path)],
        excluded=[],
        allow_exts=[".png"],
        batch_size=2,
        tagger=TaggerSettings(name="dummy"),
        index_dir=str(tmp_path / "index"),
    )

    stats = run_index_once(
        temp_db,
        settings=settings,
        tagger_override=DummyTagger(),
    )

    assert stats["scanned"] == 2
    assert stats["tagged"] == 2
    assert stats["signatures"] == 2

    conn = get_conn(temp_db)
    conn.row_factory = sqlite3.Row
    assert conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"] == 2
    assert conn.execute("SELECT COUNT(*) AS c FROM file_tags").fetchone()["c"] >= 2
    assert conn.execute("SELECT COUNT(*) AS c FROM signatures").fetchone()["c"] == 2
    conn.close()

    # Second run should skip unchanged files.
    stats_second = run_index_once(
        temp_db,
        settings=settings,
        tagger_override=DummyTagger(),
    )
    assert stats_second["new_or_changed"] == 0
    assert stats_second["tagged"] == 0
