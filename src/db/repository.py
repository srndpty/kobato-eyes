"""Data-access helpers wrapping SQLite operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def upsert_file(
    conn: sqlite3.Connection,
    *,
    path: str,
    size: int | None = None,
    mtime: float | None = None,
    sha256: str | None = None,
) -> int:
    """Insert or update a file record and return its identifier."""
    query = """
        INSERT INTO files (path, size, mtime, sha256)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            size = excluded.size,
            mtime = excluded.mtime,
            sha256 = excluded.sha256
        RETURNING id
    """

    with conn:
        cursor = conn.execute(query, (path, size, mtime, sha256))
        file_id = cursor.fetchone()[0]
    return int(file_id)


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

    def _has_column(name: str) -> bool:
        cursor = conn.execute("PRAGMA table_info(files)")
        return any(row[1] == name for row in cursor.fetchall())

    has_width = _has_column("width")
    has_height = _has_column("height")

    select_parts = [
        "f.id",
        "f.path",
        "f.size",
        "f.mtime",
    ]
    select_parts.append("f.width AS width" if has_width else "NULL AS width")
    select_parts.append("f.height AS height" if has_height else "NULL AS height")
    select_clause = ", ".join(select_parts)

    limit = max(0, int(limit))
    offset = max(0, int(offset))

    query = f"SELECT {select_clause} FROM files f WHERE {where_sql} ORDER BY {order_by} LIMIT ? OFFSET ?"

    cursor = conn.execute(query, (*params, limit, offset))
    rows = cursor.fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        file_id = row["id"]
        tag_rows = conn.execute(
            "SELECT t.name, ft.score FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
            "WHERE ft.file_id = ? ORDER BY ft.score DESC LIMIT 5",
            (file_id,),
        ).fetchall()
        top_tags = [(tag_row["name"], float(tag_row["score"])) for tag_row in tag_rows]
        results.append(
            {
                "id": file_id,
                "path": row["path"],
                "width": row["width"] if has_width else None,
                "height": row["height"] if has_height else None,
                "size": row["size"],
                "mtime": row["mtime"],
                "top_tags": top_tags,
            }
        )
    return results


__all__ = [
    "upsert_file",
    "upsert_tags",
    "replace_file_tags",
    "update_fts",
    "upsert_signatures",
    "upsert_embedding",
    "search_files",
]
