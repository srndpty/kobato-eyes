"""Helpers for manipulating the ``files`` table."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence


def bulk_upsert_files_meta(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[int, object, object, str, float]],
    *,
    coalesce_wh: bool = True,
    chunk: int = 400,
) -> None:
    """Insert or update metadata rows in ``files`` using batched statements."""

    if not rows:
        return
    for start in range(0, len(rows), chunk):
        block = rows[start : start + chunk]
        flat: list[object] = []
        for fid, width, height, sig, ts in block:
            flat.extend((int(fid), width, height, sig, float(ts)))
        values = ",".join(["(?, ?, ?, ?, ?)"] * (len(flat) // 5))
        if coalesce_wh:
            sql = (
                "INSERT INTO files (id, width, height, tagger_sig, last_tagged_at) "
                f"VALUES {values} "
                "ON CONFLICT(id) DO UPDATE SET "
                "  width        = COALESCE(excluded.width,  files.width), "
                "  height       = COALESCE(excluded.height, files.height), "
                "  tagger_sig   = excluded.tagger_sig, "
                "  last_tagged_at = excluded.last_tagged_at"
            )
        else:
            sql = (
                "INSERT INTO files (id, width, height, tagger_sig, last_tagged_at) "
                f"VALUES {values} "
                "ON CONFLICT(id) DO UPDATE SET "
                "  width        = excluded.width, "
                "  height       = excluded.height, "
                "  tagger_sig   = excluded.tagger_sig, "
                "  last_tagged_at = excluded.last_tagged_at"
            )
        conn.execute(sql, flat)


def bulk_update_files_meta_by_id(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[int | None, int | None, str | None, float | None, int]],
    *,
    coalesce_wh: bool = True,
) -> None:
    """Update ``files`` rows by id using executemany for efficiency."""

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
        sql = (
            "UPDATE files "
            "SET width = ?, height = ?, tagger_sig = ?, last_tagged_at = ? "
            "WHERE id = ?"
        )
    with conn:
        conn.executemany(sql, rows)


__all__ = ["bulk_upsert_files_meta", "bulk_update_files_meta_by_id"]
