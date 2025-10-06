# src/db/fts_offline.py
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple

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

        # 全ファイルを走査（必要なら ids を事前に取ってバッファリング）
        cursor = conn.execute("SELECT id FROM files WHERE is_present = 1")
        buf: List[Tuple[int, str]] = []
        for (fid,) in cursor.fetchall():
            rows = conn.execute(
                """
                SELECT t.name, ft.score
                FROM file_tags AS ft
                JOIN tags AS t ON t.id = ft.tag_id
                WHERE ft.file_id = ?
                ORDER BY ft.score DESC
                LIMIT ?
                """,
                (fid, int(topk)),
            ).fetchall()
            if not rows:
                continue
            text = " ".join([str(r["name"]).strip() for r in rows if str(r["name"]).strip()])
            if not text:
                continue
            buf.append((int(fid), text))
            if len(buf) >= batch:
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
