"""Data-access helpers wrapping SQLite operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _ensure_file_columns(conn: sqlite3.Connection) -> None:
    """Make sure optional columns exist on the files table."""
    rows = conn.execute("PRAGMA table_info(files)").fetchall()
    columns = {row[1] for row in rows}
    alterations: list[str] = []
    if "width" not in columns:
        alterations.append("ALTER TABLE files ADD COLUMN width INTEGER")
    if "height" not in columns:
        alterations.append("ALTER TABLE files ADD COLUMN height INTEGER")
    if "indexed_at" not in columns:
        alterations.append("ALTER TABLE files ADD COLUMN indexed_at REAL")
    if "tagger_sig" not in columns:
        alterations.append("ALTER TABLE files ADD COLUMN tagger_sig TEXT")
    if "last_tagged_at" not in columns:
        alterations.append("ALTER TABLE files ADD COLUMN last_tagged_at REAL")
    for statement in alterations:
        conn.execute(statement)
    if alterations:
        conn.commit()


def upsert_file(
    conn: sqlite3.Connection,
    *,
    path: str,
    size: int | None = None,
    mtime: float | None = None,
    sha256: str | None = None,
    width: int | None = None,
    height: int | None = None,
    indexed_at: float | None = None,
    tagger_sig: str | None = None,
    last_tagged_at: float | None = None,
) -> int:
    """Insert or update a file record and return its identifier."""
    _ensure_file_columns(conn)
    query = """
        INSERT INTO files (
            path,
            size,
            mtime,
            sha256,
            width,
            height,
            indexed_at,
            tagger_sig,
            last_tagged_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            size = excluded.size,
            mtime = excluded.mtime,
            sha256 = excluded.sha256,
            width = COALESCE(excluded.width, files.width),
            height = COALESCE(excluded.height, files.height),
            indexed_at = COALESCE(excluded.indexed_at, files.indexed_at),
            tagger_sig = COALESCE(excluded.tagger_sig, files.tagger_sig),
            last_tagged_at = COALESCE(excluded.last_tagged_at, files.last_tagged_at)
        RETURNING id
    """

    with conn:
        cursor = conn.execute(
            query,
            (
                path,
                size,
                mtime,
                sha256,
                width,
                height,
                indexed_at,
                tagger_sig,
                last_tagged_at,
            ),
        )
        file_id = cursor.fetchone()[0]
    return int(file_id)


def get_file_by_path(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    cursor = conn.execute("SELECT * FROM files WHERE path = ?", (path,))
    return cursor.fetchone()


def list_tag_names(conn: sqlite3.Connection, limit: int = 0) -> list[str]:
    """Return tag names ordered alphabetically, optionally limited."""

    sql = "SELECT name FROM tags ORDER BY name ASC"
    params: tuple[object, ...] = ()
    if limit > 0:
        sql += " LIMIT ?"
        params = (int(limit),)
    cursor = conn.execute(sql, params)
    return [str(row[0]) for row in cursor.fetchall()]


def upsert_tags(conn: sqlite3.Connection, tags: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Ensure tags exist and return a mapping from tag name to identifier."""
    results: dict[str, int] = {}
    query = """
        INSERT INTO tags (name, category)
        VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET
            category = excluded.category
        RETURNING id
    """
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
                ((file_id, tag_id, float(score)) for tag_id, score in pairs),
            )


def update_fts(conn: sqlite3.Connection, file_id: int, text: str | None) -> None:
    """Update the FTS5 index row for a given file."""
    with conn:
        conn.execute("DELETE FROM fts_files WHERE rowid = ?", (file_id,))
        if text:
            conn.execute(
                "INSERT INTO fts_files (rowid, file_id, text) VALUES (?, ?, ?)",
                (file_id, file_id, text),
            )


def upsert_signatures(conn: sqlite3.Connection, *, file_id: int, phash_u64: int, dhash_u64: int) -> None:
    """Store perceptual hash signatures for a file."""
    query = """
        INSERT INTO signatures (file_id, phash_u64, dhash_u64)
        VALUES (?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            phash_u64 = excluded.phash_u64,
            dhash_u64 = excluded.dhash_u64
    """
    with conn:
        conn.execute(query, (file_id, int(phash_u64), int(dhash_u64)))


def upsert_embedding(
    conn: sqlite3.Connection,
    *,
    file_id: int,
    model: str,
    dim: int,
    vector: bytes | memoryview,
) -> None:
    """Store or update an embedding vector for the given file/model pair."""
    query = """
        INSERT INTO embeddings (file_id, model, dim, vector)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(file_id, model) DO UPDATE SET
            dim = excluded.dim,
            vector = excluded.vector
    """
    payload = memoryview(vector) if not isinstance(vector, memoryview) else vector
    with conn:
        conn.execute(query, (file_id, model, int(dim), payload))


def search_files(
    conn: sqlite3.Connection,
    where_sql: str,
    params: list[object] | tuple[object, ...],
    *,
    order_by: str = "f.mtime DESC",
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, object]]:
    """Search files using a prebuilt WHERE clause and return enriched rows."""
    _ensure_file_columns(conn)

    limit = max(0, int(limit))
    offset = max(0, int(offset))

    query = (
        "SELECT f.id, f.path, f.size, f.mtime, f.width, f.height "
        "FROM files f WHERE "
        f"{where_sql} ORDER BY {order_by} LIMIT ? OFFSET ?"
    )

    cursor = conn.execute(query, (*params, limit, offset))
    rows = cursor.fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        file_id = row["id"]
        tag_rows = conn.execute(
            "SELECT t.name, ft.score FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
            "WHERE ft.file_id = ? ORDER BY ft.score DESC",
            (file_id,),
        ).fetchall()
        tags = [(tag_row["name"], float(tag_row["score"])) for tag_row in tag_rows]
        results.append(
            {
                "id": file_id,
                "path": row["path"],
                "width": row["width"],
                "height": row["height"],
                "size": row["size"],
                "mtime": row["mtime"],
                "tags": tags,
                "top_tags": tags,
            }
        )
    return results


def mark_indexed_at(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    indexed_at: float | None = None,
) -> None:
    """Update the ``indexed_at`` timestamp for the specified file."""
    with conn:
        conn.execute(
            "UPDATE files SET indexed_at = ? WHERE id = ?",
            (indexed_at, file_id),
        )


__all__ = [
    "upsert_file",
    "get_file_by_path",
    "upsert_tags",
    "replace_file_tags",
    "update_fts",
    "upsert_signatures",
    "upsert_embedding",
    "search_files",
    "mark_indexed_at",
]
