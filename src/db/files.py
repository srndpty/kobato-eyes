"""Helpers for manipulating the ``files`` table."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from typing import SupportsFloat, SupportsInt


def _coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    if isinstance(value, (bytes, bytearray)):
        try:
            return int(value)
        except ValueError:
            return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, SupportsInt):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _coerce_float(value: object) -> float:
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, SupportsFloat):
        return float(value)
    if isinstance(value, str):
        return float(value.strip())
    raise TypeError(f"Unsupported timestamp value: {type(value)!r}")


def bulk_upsert_files_meta(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[int, object, object, object, object]],
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
            width_int = _coerce_optional_int(width)
            height_int = _coerce_optional_int(height)
            sig_text = _coerce_optional_str(sig)
            ts_float = _coerce_float(ts)
            if fid_int in existing:
                to_update.append((width_int, height_int, sig_text, ts_float, fid_int))
            else:
                existing.add(fid_int)
                to_insert.extend(
                    (fid_int, f"__bulk__:{fid_int}", width_int, height_int, sig_text, ts_float)
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
