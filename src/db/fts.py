"""FTS-related persistence helpers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence

from .common import chunk, fts_is_contentless


def fts_delete_rows(conn: sqlite3.Connection, ids: Sequence[int]) -> None:
    """Delete FTS rows identified by *ids*."""

    ids = [int(x) for x in ids]
    if not ids:
        return
    if fts_is_contentless(conn):
        for block in chunk(ids, 300):
            values = ",".join(["('delete', ?)"] * len(block))
            try:
                conn.execute(f"INSERT INTO fts_files(fts_files, rowid) VALUES {values}", list(block))
            except sqlite3.DatabaseError:
                placeholders = ",".join(["?"] * len(block))
                try:
                    conn.execute(
                        f"DELETE FROM fts_files WHERE rowid IN ({placeholders})",
                        list(block),
                    )
                except sqlite3.DatabaseError:
                    continue
    else:
        for block in chunk(ids, 900):
            placeholders = ",".join(["?"] * len(block))
            conn.execute(f"DELETE FROM fts_files WHERE rowid IN ({placeholders})", list(block))


def fts_replace_rows(conn: sqlite3.Connection, rows: Sequence[tuple[int, str]]) -> None:
    """Replace FTS entries with the provided rowid/text pairs."""

    if not rows:
        return
    if fts_is_contentless(conn):
        for block in chunk(rows, 300):
            ids = [rid for rid, _ in block]
            values = ",".join(["('delete', ?)"] * len(ids))
            try:
                conn.execute(f"INSERT INTO fts_files(fts_files, rowid) VALUES {values}", ids)
            except sqlite3.DatabaseError:
                placeholders = ",".join(["?"] * len(ids))
                try:
                    conn.execute(
                        f"DELETE FROM fts_files WHERE rowid IN ({placeholders})",
                        ids,
                    )
                except sqlite3.DatabaseError:
                    continue
        for block in chunk(rows, 400):
            insert_payload: list[object] = []
            for rid, text in block:
                insert_payload.extend((int(rid), str(text)))
            values = ",".join(["(?, ?)"] * (len(insert_payload) // 2))
            conn.execute(f"INSERT INTO fts_files(rowid, text) VALUES {values}", insert_payload)
    else:
        for block in chunk(rows, 400):
            replace_payload: list[object] = []
            for rid, text in block:
                replace_payload.extend((int(rid), str(text)))
            values = ",".join(["(?, ?)"] * (len(replace_payload) // 2))
            conn.execute(f"INSERT OR REPLACE INTO fts_files(rowid, text) VALUES {values}", replace_payload)


def update_fts(conn: sqlite3.Connection, file_id: int, text: str | None) -> None:
    with conn:
        fts_delete_rows(conn, [file_id])
        if text:
            conn.execute(
                "INSERT INTO fts_files (rowid, text) VALUES (?, ?)",
                (file_id, text),
            )


def update_fts_bulk(conn: sqlite3.Connection, entries: Iterable[tuple[int, str | None]]) -> None:
    delete_ids: list[int] = []
    insert_rows: list[tuple[int, str]] = []
    for fid, text in entries:
        delete_ids.append(fid)
        if text:
            insert_rows.append((fid, text))
    with conn:
        if delete_ids:
            fts_delete_rows(conn, delete_ids)
        if insert_rows:
            conn.executemany(
                "INSERT INTO fts_files (rowid, text) VALUES (?, ?)",
                insert_rows,
            )


__all__ = ["fts_delete_rows", "fts_replace_rows", "update_fts", "update_fts_bulk"]
