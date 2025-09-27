"""Data-access helpers wrapping SQLite operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Iterator, Optional

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


def _normalise_category(value: object) -> int | None:
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


def _load_tag_thresholds(conn: sqlite3.Connection) -> dict[int, float]:
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
                category = _normalise_category(row["category"])
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


def _ensure_file_columns(conn: sqlite3.Connection) -> None:
    """Make sure optional columns exist on the files table."""
    rows = conn.execute("PRAGMA table_info(files)").fetchall()
    columns = {row[1] for row in rows}
    alterations: list[str] = []
    if "is_present" not in columns:
        alterations.append("ALTER TABLE files ADD COLUMN is_present INTEGER NOT NULL DEFAULT 1")
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
    if "is_present" not in columns:
        alterations.append("ALTER TABLE files ADD COLUMN is_present INTEGER NOT NULL DEFAULT 1")
    if "deleted_at" not in columns:
        alterations.append("ALTER TABLE files ADD COLUMN deleted_at TEXT")
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
    is_present: bool | int = True,
    deleted_at: object | None = None,
) -> int:
    """Insert or update a file record and return its identifier."""
    _ensure_file_columns(conn)
    query = """
        INSERT INTO files (
            path,
            size,
            mtime,
            sha256,
            is_present,
            width,
            height,
            indexed_at,
            tagger_sig,
            last_tagged_at,
            deleted_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            size = excluded.size,
            mtime = excluded.mtime,
            sha256 = excluded.sha256,
            is_present = COALESCE(excluded.is_present, files.is_present),
            width = COALESCE(excluded.width, files.width),
            height = COALESCE(excluded.height, files.height),
            indexed_at = COALESCE(excluded.indexed_at, files.indexed_at),
            tagger_sig = COALESCE(excluded.tagger_sig, files.tagger_sig),
            last_tagged_at = COALESCE(excluded.last_tagged_at, files.last_tagged_at),
            deleted_at = excluded.deleted_at
        RETURNING id
    """
    isp_val = None if is_present is None else int(bool(is_present))

    with conn:
        cursor = conn.execute(
            query,
            (
                path,
                size,
                mtime,
                sha256,
                isp_val,
                width,
                height,
                indexed_at,
                tagger_sig,
                last_tagged_at,
                deleted_at,
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


def list_untagged_under_path(conn: sqlite3.Connection, root_like: str) -> list[tuple[int, str]]:
    """Return untagged file identifiers and paths under the provided LIKE pattern."""

    query = """
        SELECT f.id, f.path
        FROM files AS f
        LEFT JOIN file_tags AS ft ON ft.file_id = f.id
        WHERE f.path LIKE ? AND f.is_present = 1
        GROUP BY f.id
        HAVING COUNT(ft.tag_id) = 0
        ORDER BY f.path ASC
    """
    cursor = conn.execute(query, (root_like,))
    try:
        return [(int(row[0]), str(row[1])) for row in cursor.fetchall()]
    finally:
        cursor.close()


def iter_files_for_dup(
    conn: sqlite3.Connection,
    path_like: Optional[str],
    *,
    model_name: Optional[str] = None,  # 使わない場合はそのまま None でOK（将来のcosine用）
) -> Iterator[Dict[str, Any]]:
    """
    Duplicatesスキャン用に、必ず **plain dict** を返す。
    DuplicateFile.from_row() が dict.get を使っても落ちないようにする。

    返すキー（DuplicateFile.from_row が期待する名前に合わせる）:
      - file_id, path, size, width, height, phash_u64, embedding
    embedding は重いので基本 None を返す（将来必要になったらJOIN）。
    """
    sql = """
      SELECT
        f.id         AS file_id,
        f.path       AS path,
        COALESCE(f.size, 0)    AS size,
        f.width      AS width,
        f.height     AS height,
        s.phash_u64  AS phash_u64
      FROM files f
      LEFT JOIN signatures s ON s.file_id = f.id
      WHERE f.is_present = 1
    """
    params: list[Any] = []
    if path_like:
        sql += " AND f.path LIKE ? ESCAPE '\\' "
        params.append(path_like)
    sql += " ORDER BY f.id"

    cur = conn.execute(sql, params)
    for r in cur:
        # ★ sqlite3.Row → 必ず plain dict に変換（.get が使える形）
        yield {
            "file_id": r["file_id"],
            "path": r["path"],
            "size": r["size"],
            "width": r["width"],
            "height": r["height"],
            "phash_u64": r["phash_u64"],
            "embedding": None,  # 将来 embeddings JOIN するならここで埋める
        }


def mark_files_absent(conn: sqlite3.Connection, file_ids: Sequence[int]) -> int:
    """Set ``is_present`` to ``0`` for the provided ``file_ids``."""

    if not file_ids:
        return 0
    placeholders = ", ".join("?" for _ in file_ids)
    sql = f"UPDATE files SET is_present = 0 WHERE id IN ({placeholders})"
    with conn:
        cursor = conn.execute(sql, tuple(int(file_id) for file_id in file_ids))
        return cursor.rowcount


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


def _resolve_relevance_thresholds(
    thresholds: Mapping[int, float] | None,
) -> tuple[float, float, float, float]:
    values: dict[int, float] = dict(_DEFAULT_CATEGORY_THRESHOLDS)
    if thresholds:
        for key, value in thresholds.items():
            category = _normalise_category(key)
            if category is None:
                continue
            try:
                values[category] = float(value)
            except (TypeError, ValueError):
                continue
    general = float(values.get(0, 0.0))
    character = float(values.get(1, 0.0))
    copyright = float(values.get(3, 0.0))
    default = float(values.get(-1, 0.0))
    return general, character, copyright, default


def search_files(
    conn: sqlite3.Connection,
    where_sql: str,
    params: list[object] | tuple[object, ...],
    *,
    tags_for_relevance: Sequence[str] | None = None,
    thresholds: Mapping[int, float] | None = None,
    order: str = "mtime",
    order_by: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict[str, object]]:
    """Search files using a prebuilt WHERE clause and return enriched rows."""
    _ensure_file_columns(conn)

    limit = max(0, int(limit))
    offset = max(0, int(offset))

    tag_terms = [str(tag) for tag in (tags_for_relevance or []) if str(tag)]
    order_mode = (order or "").lower()
    use_relevance = bool(tag_terms) and order_mode == "relevance" and not order_by

    base_params: list[object] = []
    if use_relevance:
        placeholders = ", ".join("?" for _ in tag_terms)
        general_thr, character_thr, copyright_thr, default_thr = _resolve_relevance_thresholds(thresholds)
        cte = (
            "WITH q AS ("
            "SELECT ft.file_id AS fid, SUM(ft.score) AS rel "
            "FROM file_tags ft "
            "JOIN tags t ON t.id = ft.tag_id "
            f"WHERE t.name IN ({placeholders}) "
            "AND ft.score >= CASE t.category "
            "WHEN 0 THEN ? "
            "WHEN 1 THEN ? "
            "WHEN 3 THEN ? "
            "ELSE ? "
            "END "
            "GROUP BY ft.file_id) "
        )
        query_prefix = cte
        base_params.extend(tag_terms)
        base_params.extend([general_thr, character_thr, copyright_thr, default_thr])
        relevance_select = "COALESCE(q.rel, 0.0) AS relevance"
        join_clause = "LEFT JOIN q ON q.fid = f.id "
    else:
        query_prefix = ""
        relevance_select = "0.0 AS relevance"
        join_clause = ""

    if order_by:
        order_clause = order_by
    elif use_relevance:
        order_clause = "relevance DESC, f.mtime DESC"
    else:
        order_clause = "f.mtime DESC"

    normalized_where = where_sql.strip() or "1=1"
    combined_where = f"({normalized_where}) AND f.is_present = 1"

    query = (
        f"{query_prefix}"
        "SELECT "
        "f.id, f.path, f.size, f.mtime, f.width, f.height, "
        f"{relevance_select} "
        "FROM files f "
        f"{join_clause}"
        "WHERE "
        f"{combined_where} "
        f"ORDER BY {order_clause} "
        "LIMIT ? OFFSET ?"
    )

    where_params = tuple(params or [])
    execute_params = (*base_params, *where_params, limit, offset)
    cursor = conn.execute(query, execute_params)
    rows = cursor.fetchall()

    results: list[dict[str, object]] = []
    for row in rows:
        file_id = row["id"]
        tag_rows = conn.execute(
            "SELECT t.name, t.category, ft.score FROM file_tags ft JOIN tags t ON t.id = ft.tag_id "
            "WHERE ft.file_id = ? ORDER BY ft.score DESC",
            (file_id,),
        ).fetchall()
        tags: list[tuple[str, float, int | None]] = []
        for tag_row in tag_rows:
            score = float(tag_row["score"])
            category = _normalise_category(tag_row["category"])
            tags.append((tag_row["name"], score, category))
        results.append(
            {
                "id": file_id,
                "path": row["path"],
                "width": row["width"],
                "height": row["height"],
                "size": row["size"],
                "mtime": row["mtime"],
                "relevance": float(row["relevance"] or 0.0),
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
    "iter_files_for_dup",
    "upsert_tags",
    "replace_file_tags",
    "update_fts",
    "upsert_signatures",
    "upsert_embedding",
    "search_files",
    "mark_files_absent",
    "mark_indexed_at",
    "list_tag_names",
    "list_untagged_under_path",
]
