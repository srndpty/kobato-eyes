"""Shared helpers for database repository utilities."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any

_CATEGORY_KEY_LOOKUP = {
    "0": 0,
    "general": 0,
    "1": 1,
    "character": 1,
    "2": 2,
    "rating": 2,
    "3": 3,
    "copyright": 3,
    "4": 4,
    "artist": 4,
    "5": 5,
    "meta": 5,
}

_DEFAULT_CATEGORY_THRESHOLDS = {
    0: 0.35,
    1: 0.25,
    3: 0.25,
}

DEFAULT_CATEGORY_THRESHOLDS = dict(_DEFAULT_CATEGORY_THRESHOLDS)


def normalise_category(value: object) -> int | None:
    """Normalise category input into an integer code."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _CATEGORY_KEY_LOOKUP:
        return _CATEGORY_KEY_LOOKUP[lowered]
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def load_tag_thresholds(conn: sqlite3.Connection) -> dict[int, float]:
    """Load per-category tagger thresholds from the database."""

    thresholds: dict[int, float] = {}
    try:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tagger_thresholds'"
        ).fetchone()
    except sqlite3.Error:
        table_exists = None
    if table_exists is not None:
        cursor = conn.execute("SELECT category, threshold FROM tagger_thresholds")
        try:
            for row in cursor.fetchall():
                category = normalise_category(row["category"])
                if category is None:
                    continue
                try:
                    thresholds[category] = float(row["threshold"])
                except (TypeError, ValueError):
                    continue
        finally:
            cursor.close()
    if not thresholds:
        return dict(_DEFAULT_CATEGORY_THRESHOLDS)
    for category, default in _DEFAULT_CATEGORY_THRESHOLDS.items():
        thresholds.setdefault(category, default)
    return thresholds


def chunk(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield fixed-size chunks from *seq*."""

    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def fts_is_contentless(conn: sqlite3.Connection) -> bool:
    """Return True when the FTS table is configured in contentless mode."""

    try:
        row = conn.execute("SELECT value FROM pragma_fts5('fts_files','content')").fetchone()
        value = row[0] if row else None
        return not value
    except sqlite3.Error:
        return False


__all__ = [
    "DEFAULT_CATEGORY_THRESHOLDS",
    "chunk",
    "fts_is_contentless",
    "load_tag_thresholds",
    "normalise_category",
]
