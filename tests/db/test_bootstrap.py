"""Tests covering database bootstrapping and initial indexing."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from core.config import AppPaths, PipelineSettings, TaggerSettings
from core.pipeline import run_index_once
from db.connection import bootstrap_if_needed, get_conn
from tagger.dummy import DummyTagger
from utils import paths

pytestmark = pytest.mark.integration


def _make_sample_image(path: Path) -> None:
    image = Image.new("RGB", (32, 32), color=(200, 100, 50))
    image.save(path, format="PNG")


def test_bootstrap_creates_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "fresh.db"
    bootstrap_if_needed(db_path)

    conn = get_conn(db_path)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')").fetchall()
        tables = {row["name"] for row in rows}
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()

    expected = {"files", "tags", "file_tags", "fts_files", "signatures"}
    assert expected.issubset(tables)
    assert int(version) >= 1


def test_run_index_once_bootstraps_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "library.db"
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _make_sample_image(image_dir / "one.png")

    app_paths = AppPaths(env={"KOE_DATA_DIR": str(tmp_path / "data")})
    monkeypatch.setattr(paths, "_APP_PATHS", app_paths)

    settings = PipelineSettings(
        roots=[str(image_dir)],
        excluded=[],
        allow_exts=[".png"],
        batch_size=1,
        tagger=TaggerSettings(name="dummy"),
        index_dir=str(tmp_path / "index"),
    )

    stats = run_index_once(
        db_path,
        settings=settings,
        tagger_override=DummyTagger(),
    )

    assert stats["scanned"] == 1

    conn = get_conn(db_path)
    try:
        files_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    finally:
        conn.close()

    assert int(files_count) == 1
