"""Tests for tag statistics model and threshold loading."""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, Iterator

import pytest

pytest.importorskip("PyQt6.QtCore", reason="PyQt6 core required", exc_type=ImportError)
from PyQt6.QtCore import QCoreApplication, Qt

from db.schema import apply_schema
from ui.tag_stats import _TagStatsModel, _load_thresholds

pytestmark = pytest.mark.gui


@pytest.fixture(scope="session", autouse=True)
def _headless_qapp() -> Iterable[None]:
    os.environ.setdefault("KOE_HEADLESS", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QCoreApplication.instance()
    if app is None:
        app = QCoreApplication([])
    yield


def _create_sample_connection(*, thresholds: dict[int, float] | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    apply_schema(conn)

    tag_ids: dict[str, int] = {}
    for name, category in (
        ("sunrise", 0),
        ("artistA", 1),
        ("metaTag", 5),
    ):
        cursor = conn.execute("INSERT INTO tags(name, category) VALUES (?, ?)", (name, category))
        tag_ids[name] = int(cursor.lastrowid)

    file_ids: dict[str, int] = {}
    for path, is_present in (
        ("a.jpg", 1),
        ("b.jpg", 1),
        ("c.jpg", 0),
    ):
        cursor = conn.execute("INSERT INTO files(path, is_present) VALUES (?, ?)", (path, is_present))
        file_ids[path] = int(cursor.lastrowid)

    conn.executemany(
        "INSERT INTO file_tags(file_id, tag_id, score) VALUES (?, ?, ?)",
        [
            (file_ids["a.jpg"], tag_ids["sunrise"], 0.40),
            (file_ids["b.jpg"], tag_ids["sunrise"], 0.55),
            (file_ids["c.jpg"], tag_ids["sunrise"], 0.80),
            (file_ids["a.jpg"], tag_ids["metaTag"], 0.15),
            (file_ids["b.jpg"], tag_ids["metaTag"], 0.05),
            (file_ids["b.jpg"], tag_ids["artistA"], 0.90),
        ],
    )

    conn.execute("DELETE FROM tagger_thresholds")
    for category, value in (thresholds or {0: 0.45, 5: 0.2}).items():
        conn.execute(
            "INSERT INTO tagger_thresholds(category, threshold) VALUES (?, ?)",
            (str(category), float(value)),
        )

    conn.commit()
    return conn


@pytest.fixture()
def sample_connection() -> Iterator[sqlite3.Connection]:
    conn = _create_sample_connection()
    try:
        yield conn
    finally:
        conn.close()


def test_load_thresholds_prefers_database_overrides() -> None:
    conn = sqlite3.connect(":memory:")
    try:
        apply_schema(conn)
        conn.execute("DELETE FROM tagger_thresholds")
        conn.executemany(
            "INSERT INTO tagger_thresholds(category, threshold) VALUES (?, ?)",
            [("0", 0.9), ("5", 0.2)],
        )

        thresholds = _load_thresholds(conn)

        assert thresholds[0] == pytest.approx(0.9)
        assert thresholds[5] == pytest.approx(0.2)
        assert thresholds[4] == pytest.approx(0.25)
        assert thresholds[2] == pytest.approx(0.0)
    finally:
        conn.close()


def test_tag_stats_model_respects_filters(sample_connection: sqlite3.Connection) -> None:
    model = _TagStatsModel()

    model.load(sample_connection, category=None, respect_thresholds=False)
    assert model.rowCount() == 3
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == "metaTag"
    assert model.data(model.index(0, 2), Qt.ItemDataRole.DisplayRole) == 2
    assert model.data(model.index(0, 3), Qt.ItemDataRole.DisplayRole) == "0.100"
    assert model.data(model.index(1, 3), Qt.ItemDataRole.UserRole) == pytest.approx(0.475)
    assert model.tag_at(1) == "sunrise"
    assert model.tag_at(42) == ""

    model.load(sample_connection, category=None, respect_thresholds=True)
    assert model.rowCount() == 2
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == "artistA"
    assert model.data(model.index(1, 2), Qt.ItemDataRole.DisplayRole) == 1
    assert model.data(model.index(1, 3), Qt.ItemDataRole.UserRole) == pytest.approx(0.55)

    model.load(sample_connection, category=0, respect_thresholds=True)
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "general"
    assert model.tag_at(0) == "sunrise"
