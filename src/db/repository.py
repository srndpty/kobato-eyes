"""Data-access helpers wrapping SQLite operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .common import DEFAULT_CATEGORY_THRESHOLDS
from .common import chunk as _chunk
from .common import fts_is_contentless as _fts_is_contentless
from .common import load_tag_thresholds as _load_tag_thresholds
from .common import normalise_category as _normalise_category
from .files import bulk_update_files_meta_by_id, bulk_upsert_files_meta
from .fts import fts_delete_rows, fts_replace_rows
from .tags import replace_file_tags, replace_file_tags_many, upsert_tags

 


# ----------------------------------------
# files テーブル
# ----------------------------------------


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
    """
    Insert or update a file record and return its identifier.

    NOTE:
      - スキーマは INTEGER PRIMARY KEY（AUTOINCREMENT なし）想定。
      - 戻り値の id は RETURNING を使って取得。
    """
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


def update_files_metadata_bulk(
    conn: sqlite3.Connection,
    rows: Iterable[
        Tuple[
            int,
            Optional[int],
            Optional[float],
            Optional[str],
            Optional[int],
            Optional[int],
            Optional[str],
            Optional[float],
        ]
    ],
) -> int:
    """
    files のメタデータを複数行まとめて更新する。
    rows: Iterable of (file_id, size, mtime, sha256, width, height, tagger_sig, last_tagged_at)
    """
    sql = """
        UPDATE files
           SET size = ?,
               mtime = ?,
               sha256 = ?,
               width = COALESCE(?, width),
               height = COALESCE(?, height),
               tagger_sig = COALESCE(?, tagger_sig),
               last_tagged_at = COALESCE(?, last_tagged_at),
               is_present = 1
         WHERE id = ?
    """

    # executemany 用に順序を合わせる（? の並びに注意）
    def _adapt(
        it: Iterable[
            Tuple[
                int,
                Optional[int],
                Optional[float],
                Optional[str],
                Optional[int],
                Optional[int],
                Optional[str],
                Optional[float],
            ]
        ],
    ):
        for file_id, size, mtime, sha256, width, height, tagger_sig, last_tagged_at in it:
            yield (size, mtime, sha256, width, height, tagger_sig, last_tagged_at, file_id)

    with conn:
        cur = conn.executemany(sql, _adapt(rows))
        return cur.rowcount


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


# ----------------------------------------
# タグ／スコア
# ----------------------------------------
# ----------------------------------------
# FTS（contentless/detail=none 想定）
# ----------------------------------------


def update_fts(conn: sqlite3.Connection, file_id: int, text: str | None) -> None:
    with conn:
        # ここを置き換え
        # conn.execute("DELETE FROM fts_files WHERE rowid = ?", (file_id,))
        fts_delete_rows(conn, [file_id])
        if text:
            # contentless でも OK：rowid とインデックス対象の列（text）を挿入
            conn.execute(
                "INSERT INTO fts_files (rowid, text) VALUES (?, ?)",
                (file_id, text),
            )


def update_fts_bulk(conn: sqlite3.Connection, entries: Iterable[tuple[int, Optional[str]]]) -> None:
    """
    FTS エントリの一括更新。
    entries: Iterable[(file_id, text_or_None)]
    """
    delete_ids: list[int] = []
    insert_rows: list[tuple[int, str]] = []
    for fid, text in entries:
        delete_ids.append(fid)
        if text:
            insert_rows.append((fid, text))

    with conn:
        if delete_ids:
            # conn.executemany("DELETE FROM fts_files WHERE rowid = ?", delete_rows)
            fts_delete_rows(conn, delete_ids)
        if insert_rows:
            conn.executemany(
                "INSERT INTO fts_files (rowid, text) VALUES (?, ?)",
                insert_rows,
            )


# ----------------------------------------
# シグネチャ／埋め込み
# ----------------------------------------


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


# ----------------------------------------
# 検索
# ----------------------------------------


def _resolve_relevance_thresholds(
    thresholds: Mapping[int, float] | None,
) -> tuple[float, float, float, float]:
    values: dict[int, float] = dict(DEFAULT_CATEGORY_THRESHOLDS)
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


# ----------------------------------------
# Duplicate Scan 用
# ----------------------------------------


def iter_files_for_dup(
    conn: sqlite3.Connection,
    path_like: Optional[str],
) -> Iterator[Dict[str, Any]]:
    """
    Duplicatesスキャン用に、必ず **plain dict** を返す。
    DuplicateFile.from_row() が dict.get を使っても落ちないようにする。

    返すキー（DuplicateFile.from_row が期待する名前に合わせる）:
      - file_id, path, size, width, height, phash_u64
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
        yield {
            "file_id": r["file_id"],
            "path": r["path"],
            "size": r["size"],
            "width": r["width"],
            "height": r["height"],
            "phash_u64": r["phash_u64"],
        }


# ----------------------------------------
# クエリ構築ヘルパ
# ----------------------------------------


def build_where_and_params_for_query(
    conn: sqlite3.Connection,
    query: str,
    *,
    alias_file: str = "f",
) -> tuple[str, list[object]]:
    """
    UIのタグ検索クエリ文字列を WHERE 節とプレースホルダ値に変換する。
    - DBに保存されたタグしきい値(_load_tag_thresholds)を使う
    - core.query.translate_query を呼んで SQL 断片を得る
    """
    from core.query import translate_query  # 循環回避のためここで import

    try:
        thresholds = _load_tag_thresholds(conn)  # dict[int,float] を想定
    except Exception:
        thresholds = {}
    fragment = translate_query(query, file_alias=alias_file, thresholds=thresholds)
    where = (fragment.where or "").strip() or "1=1"
    params = list(fragment.params or [])
    return where, params


def iter_paths_for_search(conn: sqlite3.Connection, query: str) -> Iterator[str]:
    """
    検索クエリに一致する files.path を is_present=1 で列挙。
    """
    where, params = build_where_and_params_for_query(conn, query, alias_file="f")
    sql = f"SELECT f.path FROM files f WHERE f.is_present = 1 AND ({where}) ORDER BY f.id"
    cur = conn.execute(sql, params)
    for row in cur:
        yield (row["path"] if isinstance(row, sqlite3.Row) else row[0])


# ----------------------------------------
# バッチ書き込み（タグ付け）高速化 API（任意利用）
# ----------------------------------------


def write_tagging_batch(
    conn: sqlite3.Connection,
    items: Iterable[dict[str, Any]],
) -> int:
    """
    タグ付け一括書き込み。
    items の各要素は以下のキーを想定:

      {
        "file_id": int,                     # 必須
        "file_meta": {                      # 任意（更新したい場合）
            "size": int | None,
            "mtime": float | None,
            "sha256": str | None,
            "width": int | None,
            "height": int | None,
            "tagger_sig": str | None,
            "last_tagged_at": float | None,
        },
        "tags": list[(tag_id:int, score:float)],  # 置換するタグ一覧（空なら全削除）
        "fts_text": str | None,                    # FTS に入れるテキスト
      }

    戻り値: 処理したファイル件数
    """
    # files 更新
    meta_rows: List[
        Tuple[
            int,
            Optional[int],
            Optional[float],
            Optional[str],
            Optional[int],
            Optional[int],
            Optional[str],
            Optional[float],
        ]
    ] = []
    tag_map: Dict[int, List[Tuple[int, float]]] = {}
    fts_rows: List[Tuple[int, Optional[str]]] = []

    count = 0
    for it in items:
        fid = int(it["file_id"])
        count += 1

        meta = dict(it.get("file_meta") or {})
        meta_rows.append(
            (
                fid,
                meta.get("size"),
                meta.get("mtime"),
                meta.get("sha256"),
                meta.get("width"),
                meta.get("height"),
                meta.get("tagger_sig"),
                meta.get("last_tagged_at"),
            )
        )

        tags = [(int(tid), float(sc)) for (tid, sc) in (it.get("tags") or [])]
        tag_map[fid] = tags

        fts_rows.append((fid, (it.get("fts_text") or None)))

    # 1トランザクションで一気に流す
    with conn:
        if meta_rows:
            update_files_metadata_bulk(conn, meta_rows)
        if tag_map:
            replace_file_tags_many(conn, tag_map)
        if fts_rows:
            update_fts_bulk(conn, fts_rows)

    return count


def mark_files_absent(conn: sqlite3.Connection, file_ids: Sequence[int]) -> int:
    ids = [int(fid) for fid in file_ids if fid is not None]
    if not ids:
        return 0
    with conn:
        cur = conn.execute(
            f"UPDATE files SET is_present = 0, deleted_at = CURRENT_TIMESTAMP "
            f"WHERE id IN ({', '.join('?' for _ in ids)})",
            ids,
        )
        # ここを置き換え
        # conn.execute(f"DELETE FROM fts_files WHERE rowid IN ({placeholders})", ids)
        fts_delete_rows(conn, ids)
        return int(cur.rowcount or 0)


__all__ = [
    "upsert_file",
    "get_file_by_path",
    "iter_files_for_dup",
    "upsert_tags",
    "replace_file_tags",
    "replace_file_tags_many",
    "update_fts",
    "update_fts_bulk",
    "upsert_signatures",
    "search_files",
    "mark_indexed_at",
    "list_tag_names",
    "list_untagged_under_path",
    "build_where_and_params_for_query",
    "iter_paths_for_search",
    "update_files_metadata_bulk",
    "write_tagging_batch",
    "mark_files_absent",
    "fts_delete_rows",
    "fts_replace_rows",
    "bulk_upsert_files_meta",
    "bulk_update_files_meta_by_id",
]
