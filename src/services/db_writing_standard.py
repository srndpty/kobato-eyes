"""Standard WAL batch writer used by :mod:`services.db_writing`."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from core.pipeline.contracts import DBItem


def chunked(seq: Sequence[int], size: int) -> list[list[int]]:
    """Split ``seq`` into fixed-size chunks."""

    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def upsert_tags_uncommitted(
    conn: sqlite3.Connection,
    tags: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Upsert tag definitions without committing the caller's transaction."""

    results: dict[str, int] = {}
    query = (
        "INSERT INTO tags (name, category) "
        "VALUES (?, ?) "
        "ON CONFLICT(name) DO UPDATE SET category = excluded.category "
        "RETURNING id"
    )
    for tag in tags:
        name = str(tag["name"]).strip()
        category = int(tag.get("category", 0))
        cursor = conn.execute(query, (name, category))
        tag_id = cursor.fetchone()[0]
        results[name] = int(tag_id)
    return results


def bulk_update_files_meta_by_id_uncommitted(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[int | None, int | None, str | None, float | None, int]],
    *,
    coalesce_wh: bool = True,
) -> None:
    """Update file metadata without committing the caller's transaction."""

    if not rows:
        return
    if coalesce_wh:
        sql = (
            "UPDATE files "
            "SET width = COALESCE(?, width), "
            "    height = COALESCE(?, height), "
            "    tagger_sig = COALESCE(?, tagger_sig), "
            "    last_tagged_at = COALESCE(?, last_tagged_at) "
            "WHERE id = ?"
        )
    else:
        sql = "UPDATE files SET width = ?, height = ?, tagger_sig = ?, last_tagged_at = ? WHERE id = ?"
    conn.executemany(sql, rows)


class StandardBatchWriter:
    """Persist batches directly to the primary tables under WAL."""

    def __init__(self, fts_replace_rows: Callable[[sqlite3.Connection, Sequence[tuple[int, str]]], None]) -> None:
        self._fts_replace_rows = fts_replace_rows

    def flush(
        self,
        conn: sqlite3.Connection,
        items: Sequence[DBItem],
        *,
        tag_cache: dict[str, int],
        default_tagger_sig: str | None,
        skip_fts: bool,
        fts_topk: int,
    ) -> dict[str, int]:
        """Flush ``items`` and return the updated tag cache."""

        conn.execute("BEGIN IMMEDIATE")
        next_tag_cache = dict(tag_cache)
        try:
            new_defs: list[dict[str, object]] = []
            for it in items:
                for name, _score, category in it.tags:
                    if name not in next_tag_cache:
                        new_defs.append({"name": name, "category": int(category)})
            if new_defs:
                next_tag_cache.update(upsert_tags_uncommitted(conn, new_defs))
            file_ids = [it.file_id for it in items]
            for chunk in chunked(file_ids, 900):
                if not chunk:
                    continue
                placeholders = ",".join(["?"] * len(chunk))
                conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({placeholders})", chunk)
            tag_rows: list[tuple[int, int, float]] = []
            for it in items:
                for name, score, _category in it.tags:
                    tag_id = next_tag_cache.get(name)
                    if tag_id is not None:
                        tag_rows.append((it.file_id, int(tag_id), float(score)))
            if tag_rows:
                conn.executemany(
                    "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
                    tag_rows,
                )
            if not skip_fts:
                fts_rows: list[tuple[int, str]] = []
                for it in items:
                    top = sorted(it.tags, key=lambda t: t[1], reverse=True)[:fts_topk]
                    text = " ".join([name for (name, _score, _category) in top])
                    if text:
                        fts_rows.append((it.file_id, text))
                if fts_rows:
                    self._fts_replace_rows(conn, fts_rows)
            now = time.time()
            meta_rows: list[tuple[int | None, int | None, str | None, float | None, int]] = []
            for it in items:
                sig = it.tagger_sig or default_tagger_sig
                ts = it.tagged_at or now
                meta_rows.append((it.width, it.height, sig, ts, it.file_id))
            bulk_update_files_meta_by_id_uncommitted(conn, meta_rows, coalesce_wh=True)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return next_tag_cache


__all__ = [
    "StandardBatchWriter",
    "bulk_update_files_meta_by_id_uncommitted",
    "chunked",
    "upsert_tags_uncommitted",
]
