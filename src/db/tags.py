"""Helpers for tag definitions and assignments."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any


def upsert_tags(conn: sqlite3.Connection, tags: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Ensure tags exist and return a mapping from tag name to identifier."""

    results: dict[str, int] = {}
    query = (
        "INSERT INTO tags (name, category) "
        "VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET category = excluded.category "
        "RETURNING id"
    )
    with conn:
        for tag in tags:
            name = str(tag["name"]).strip()
            category = int(tag.get("category", 0))
            cursor = conn.execute(query, (name, category))
            tag_id = cursor.fetchone()[0]
            results[name] = int(tag_id)
    return results


def replace_file_tags(conn: sqlite3.Connection, file_id: int, tag_scores: Iterable[tuple[int, float]]) -> None:
    """Replace tag assignments for a file with the provided tag/score pairs."""

    pairs = list(tag_scores)
    with conn:
        conn.execute("DELETE FROM file_tags WHERE file_id = ?", (file_id,))
        if pairs:
            conn.executemany(
                "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
                ((file_id, int(tag_id), float(score)) for tag_id, score in pairs),
            )


def replace_file_tags_many(
    conn: sqlite3.Connection,
    mapping: Mapping[int, Iterable[tuple[int, float]]],
) -> None:
    """Replace tag assignments for multiple files at once."""

    delete_rows = [(fid,) for fid in mapping.keys()]
    insert_rows: list[tuple[int, int, float]] = []
    for fid, pairs in mapping.items():
        for tag_id, score in pairs:
            insert_rows.append((fid, int(tag_id), float(score)))
    with conn:
        if delete_rows:
            conn.executemany("DELETE FROM file_tags WHERE file_id = ?", delete_rows)
        if insert_rows:
            conn.executemany(
                "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
                insert_rows,
            )


__all__ = ["replace_file_tags", "replace_file_tags_many", "upsert_tags"]
