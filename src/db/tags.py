"""Helpers for tag definitions and assignments."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

# tags.name は一意でカテゴリは全画像で共有される. Danbooru 本家のカテゴリ変更に追従するため後勝ちで上書きする.
# 上書きは同名タグを持つ既存画像の category: 検索結果にも及ぶ（影響は find_category_conflicts で事前確認）.
_UPSERT_LATEST_CATEGORY = (
    "INSERT INTO tags (name, category) VALUES (?, ?) "
    "ON CONFLICT(name) DO UPDATE SET category = excluded.category "
    "RETURNING id"
)


def upsert_tags_uncommitted(conn: sqlite3.Connection, tags: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Upsert tag definitions (latest category wins); caller owns the transaction."""

    results: dict[str, int] = {}
    for tag in tags:
        name = str(tag["name"]).strip()
        category = int(tag.get("category", 0))
        results[name] = int(conn.execute(_UPSERT_LATEST_CATEGORY, (name, category)).fetchone()[0])
    return results


def upsert_tags(conn: sqlite3.Connection, tags: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Ensure tags exist and return a mapping from tag name to identifier."""

    with conn:
        return upsert_tags_uncommitted(conn, tags)


def find_category_conflicts(
    conn: sqlite3.Connection, definitions: Iterable[tuple[str, int]]
) -> list[tuple[str, int, int, int]]:
    """
    新モデル導入前の事前確認用. 既存 DB と異なるカテゴリを持つタグを
    (name, stored_category, new_category, 使用画像数) で返す.
    """

    rows = []
    for name, category in definitions:
        row = conn.execute(
            "SELECT t.category, (SELECT COUNT(*) FROM file_tags ft WHERE ft.tag_id = t.id) "
            "FROM tags t WHERE t.name = ?",
            (name,),
        ).fetchone()
        if row is not None and int(row[0]) != int(category):
            rows.append((name, int(row[0]), int(category), int(row[1])))
    return rows


def replace_file_tags(conn: sqlite3.Connection, file_id: int, tag_scores: Iterable[tuple[int, float]]) -> None:
    """Replace tag assignments for a file with the provided tag/score pairs."""

    pairs = list(tag_scores)
    with conn:
        conn.execute("DELETE FROM file_tags WHERE file_id = ?", (file_id,))
        if pairs:
            conn.executemany(
                "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
                ((file_id, int(tag_id), float(score)) for tag_id, score in pairs),
            )


def replace_file_tags_many(
    conn: sqlite3.Connection,
    mapping: Mapping[int, Iterable[tuple[int, float]]],
) -> None:
    """Replace tag assignments for multiple files at once."""

    delete_rows = [(fid,) for fid in mapping.keys()]
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


__all__ = [
    "find_category_conflicts",
    "replace_file_tags",
    "replace_file_tags_many",
    "upsert_tags",
    "upsert_tags_uncommitted",
]
