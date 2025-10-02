"""Data-access helpers wrapping SQLite operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ----------------------------------------
# 定数・ユーティリティ
# ----------------------------------------

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
    0: 0.35,  # general
    1: 0.25,  # character
    3: 0.25,  # copyright
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


# db/repository.py（抜粋・追加）


def _chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _fts_is_contentless(conn: sqlite3.Connection) -> bool:
    # 既存の実装を使っているはず。なければ簡易版：
    try:
        row = conn.execute("SELECT value FROM pragma_fts5('fts_files','content')").fetchone()
        val = row[0] if row else None
        return not val  # None or '' → contentless
    except sqlite3.Error:
        # 取得できない場合は contentless とみなさない
        return False


# db/repository.py に追加（既存の _chunk / _fts_is_contentless の下あたり）


def bulk_upsert_files_meta(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[int, object, object, str, float]],
    *,
    coalesce_wh: bool = True,
    chunk: int = 400,
) -> None:
    """
    files(id, width, height, tagger_sig, last_tagged_at) を一括置換。
    - rows: (id, width_or_None, height_or_None, tagger_sig, last_tagged_at)
    - width/height は None を渡せば既存値維持（COALESCE）
    - 1 ステートメント n 行の multi-VALUES を複数回に分けて出す
    - トランザクション開始/終了は呼び出し側に任せる
    """
    if not rows:
        return
    for chunk_rows in (rows[i : i + chunk] for i in range(0, len(rows), chunk)):
        flat: list[object] = []
        for fid, w, h, sig, ts in chunk_rows:
            flat.extend((int(fid), w, h, sig, float(ts)))
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


# db/repository.py に追加


def bulk_update_files_meta_by_id(
    conn: sqlite3.Connection,
    rows: Sequence[tuple[int | None, int | None, str | None, float | None, int]],
    *,
    coalesce_wh: bool = True,
) -> None:
    """
    files を id でまとめて UPDATE する。
    rows: (width, height, tagger_sig, last_tagged_at, file_id) のタプル列。

    coalesce_wh=True のとき:
      width/height が None の場合は既存値を保持（COALESCE）
    """
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
        sql = "UPDATE files " "SET width = ?, height = ?, tagger_sig = ?, last_tagged_at = ? " "WHERE id = ?"

    with conn:
        conn.executemany(sql, rows)


def fts_delete_rows(conn: sqlite3.Connection, ids: Sequence[int]) -> None:
    """
    fts_files から rowid ∈ ids を削除。
    - ここではトランザクションを開始/終了しない（呼び出し側に任せる）
    - contentless の場合は特殊INSERT ('delete', rowid) を multi-VALUES でまとめて投げる
    """
    ids = [int(x) for x in ids]
    if not ids:
        return

    if _fts_is_contentless(conn):
        # 1 ステートメント内に複数 VALUES を詰める（プレースホルダ上限を避けて分割）
        # SQLite の既定は 999 個までなので、少し余裕を見て ~300 件/ステートメント
        for chunk in _chunk(ids, 300):
            values = ",".join(["('delete', ?)"] * len(chunk))
            sql = f"INSERT INTO fts_files(fts_files, rowid) VALUES {values}"
            conn.execute(sql, chunk)
    else:
        for chunk in _chunk(ids, 900):
            placeholders = ",".join(["?"] * len(chunk))
            conn.execute(f"DELETE FROM fts_files WHERE rowid IN ({placeholders})", chunk)


def fts_replace_rows(conn: sqlite3.Connection, rows: Sequence[tuple[int, str]]) -> None:
    print("repository.fts_replace_rows")
    """
    rowid → text を「置換」する（=古い index を消して新しい index を入れる）。
    - トランザクションは呼び出し側に任せる
    - contentless: 'delete' の multi-VALUES → INSERT の multi-VALUES
    - 非 contentless: INSERT OR REPLACE の multi-VALUES 1発
    """
    if not rows:
        return

    if _fts_is_contentless(conn):
        # 先に delete をまとめて
        for chunk in _chunk(rows, 300):
            ids = [rid for rid, _ in chunk]
            values = ",".join(["('delete', ?)"] * len(ids))
            conn.execute(f"INSERT INTO fts_files(fts_files, rowid) VALUES {values}", ids)
        # 続けて insert
        for chunk in _chunk(rows, 400):
            flat: list[object] = []
            for rid, text in chunk:
                flat.extend((int(rid), str(text)))
            values = ",".join(["(?, ?)"] * (len(flat) // 2))
            conn.execute(f"INSERT INTO fts_files(rowid, text) VALUES {values}", flat)
    else:
        # 非 contentless は置換 1 回で済む
        for chunk in _chunk(rows, 400):
            flat: list[object] = []
            for rid, text in chunk:
                flat.extend((int(rid), str(text)))
            values = ",".join(["(?, ?)"] * (len(flat) // 2))
            conn.execute(f"INSERT OR REPLACE INTO fts_files(rowid, text) VALUES {values}", flat)


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


def upsert_tags(conn: sqlite3.Connection, tags: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    print("repository.upsert_tags")

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


def replace_file_tags_many(
    conn: sqlite3.Connection,
    mapping: Mapping[int, Iterable[tuple[int, float]]],
) -> None:
    """
    複数 file_id のタグを一括置換。
    mapping: {file_id: [(tag_id, score), ...], ...}
    """
    # DELETE をまとめて
    delete_rows = [(fid,) for fid in mapping.keys()]
    # INSERT 用にフラット化
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
                "INSERT INTO fts_files (rowid, file_id, text) VALUES (?, ?, ?)",
                (file_id, file_id, text),
            )


def update_fts_bulk(conn: sqlite3.Connection, entries: Iterable[tuple[int, Optional[str]]]) -> None:
    """
    FTS エントリの一括更新。
    entries: Iterable[(file_id, text_or_None)]
    """
    delete_rows: list[tuple[int]] = []
    insert_rows: list[tuple[int, str]] = []
    for fid, text in entries:
        delete_rows.append((fid,))
        if text:
            insert_rows.append((fid, text))

    with conn:
        if delete_rows:
            # conn.executemany("DELETE FROM fts_files WHERE rowid = ?", delete_rows)
            fts_delete_rows(conn, delete_rows)
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


# ----------------------------------------
# 検索
# ----------------------------------------


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
        yield {
            "file_id": r["file_id"],
            "path": r["path"],
            "size": r["size"],
            "width": r["width"],
            "height": r["height"],
            "phash_u64": r["phash_u64"],
            "embedding": None,
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
    "upsert_embedding",
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
