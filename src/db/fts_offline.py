"""Offline FTS rebuild helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from db.connection import get_conn
from db.repository import fts_replace_rows


def _detect_fts5_tables(conn: sqlite3.Connection) -> list[str]:
    """sqlite_master から fts5 仮想表名を列挙（複数あっても全消去対象にできる）。"""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND sql LIKE '%USING fts5%'").fetchall()
    return [str(r["name"]) for r in rows if r["name"]]


def _truncate_fts(conn: sqlite3.Connection) -> None:
    """検出した fts5 テーブルをすべて空にする。"""
    tables = _detect_fts5_tables(conn)
    for name in tables:
        conn.execute(
            f"INSERT INTO {name}({name}) VALUES('delete-all')"
        )  # 外部コンテンツ/コンテンツレスでも安全に空になる


def rebuild_fts_offline(
    db_path: str | Path,
    *,
    topk: int = 16,
    batch: int = 2000,
) -> int:
    """
    file_tags → tags を集計し、files の is_present=1 行について
    FTS を完全に作り直す。戻り値は REPLACE した件数。
    """
    normalized_topk = max(0, int(topk))
    normalized_batch = max(1, int(batch))
    total = 0
    with get_conn(db_path, allow_when_quiesced=True) as conn:
        conn.row_factory = sqlite3.Row
        # 安全寄り（WAL/NORMAL）で OK：ここは“まとめて 1 回だけ”耐久性を持たせたい
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.DatabaseError:
            pass

        # いったん FTS を空に
        _truncate_fts(conn)
        if normalized_topk <= 0:
            conn.commit()
            return 0

        buf: list[tuple[int, str]] = []
        cursor = conn.execute(
            """
            WITH ranked AS (
                SELECT
                    f.id AS file_id,
                    TRIM(t.name) AS tag_name,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.id
                        ORDER BY ft.score DESC, t.name ASC
                    ) AS rank
                FROM files AS f
                JOIN file_tags AS ft ON ft.file_id = f.id
                JOIN tags AS t ON t.id = ft.tag_id
                WHERE f.is_present = 1
                  AND TRIM(t.name) <> ''
            )
            SELECT file_id, GROUP_CONCAT(tag_name, ' ') AS text
            FROM (
                SELECT file_id, tag_name
                FROM ranked
                WHERE rank <= ?
                ORDER BY file_id ASC, rank ASC
            )
            GROUP BY file_id
            ORDER BY file_id ASC
            """,
            (normalized_topk,),
        )
        for row in cursor:
            text = str(row["text"] or "").strip()
            if not text:
                continue
            buf.append((int(row["file_id"]), text))
            if len(buf) >= normalized_batch:
                fts_replace_rows(conn, buf)  # 既存のユーティリティでまとめて投入
                total += len(buf)
                buf.clear()
        if buf:
            fts_replace_rows(conn, buf)
            total += len(buf)
        conn.commit()
        # 次回起動が軽くなるよう WAL を切り詰め
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.DatabaseError:
            pass
    return total
