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
        ids = [int(row[0]) for row in block]
        existing: set[int] = set()
        if ids:
            placeholders = ",".join(["?"] * len(ids))
            rows_found = conn.execute(
                f"SELECT id FROM files WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            existing = {int(row[0]) for row in rows_found}

        to_update: list[tuple[int | None, int | None, str | None, float | None, int]] = []
        to_insert: list[object] = []
        for fid, width, height, sig, ts in block:
            fid_int = int(fid)
            ts_float = float(ts)
            if fid_int in existing:
                to_update.append((width, height, sig, ts_float, fid_int))
            else:
                existing.add(fid_int)
                to_insert.extend(
                    (fid_int, f"__bulk__:{fid_int}", width, height, sig, ts_float)
                )

        if to_insert:
            values = ",".join(["(?, ?, ?, ?, ?, ?)"] * (len(to_insert) // 6))
            conn.execute(
                "INSERT INTO files (id, path, width, height, tagger_sig, last_tagged_at) "
                f"VALUES {values}",
                to_insert,
            )

        if to_update:
            bulk_update_files_meta_by_id(
                conn,
                to_update,
                coalesce_wh=coalesce_wh,
            )


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
