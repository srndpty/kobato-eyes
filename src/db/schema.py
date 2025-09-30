"""Database schema management for kobato-eyes."""

from __future__ import annotations

import sqlite3
from typing import Callable, Iterable

# v4: FTS を contentless（rowid = files.id）へ移行
# v5: 細かな index 調整（必要なら将来用）
CURRENT_SCHEMA_VERSION = 5

SCHEMA_STATEMENTS: tuple[str, ...] = (
    # files
    """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        size INTEGER,
        mtime REAL,
        sha256 TEXT,
        is_present INTEGER NOT NULL DEFAULT 1,
        width INTEGER,
        height INTEGER,
        indexed_at REAL,
        tagger_sig TEXT,
        last_tagged_at REAL,
        deleted_at TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    # tags
    """
    CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        category INTEGER DEFAULT 0
    );
    """,
    # file_tags
    """
    CREATE TABLE IF NOT EXISTS file_tags (
        file_id INTEGER NOT NULL,
        tag_id INTEGER NOT NULL,
        score REAL DEFAULT 1.0,
        PRIMARY KEY (file_id, tag_id),
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
        FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
    );
    """,
    # thresholds
    """
    CREATE TABLE IF NOT EXISTS tagger_thresholds (
        category TEXT PRIMARY KEY,
        threshold REAL NOT NULL
    );
    """,
    # FTS5 (contentless)
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_files USING fts5(
        text,
        content='',
        tokenize='unicode61'
    );
    """,
    # signatures
    """
    CREATE TABLE IF NOT EXISTS signatures (
        file_id INTEGER PRIMARY KEY,
        phash_u64 INTEGER NOT NULL,
        dhash_u64 INTEGER NOT NULL,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    """,
    # embeddings
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        file_id INTEGER NOT NULL,
        model TEXT NOT NULL,
        dim INTEGER NOT NULL,
        vector BLOB NOT NULL,
        PRIMARY KEY (file_id, model),
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    """,
    # minimal indexes （詳細は connection._ensure_indexes で補完）
    """
    CREATE INDEX IF NOT EXISTS idx_tags_category        ON tags(category);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id     ON file_tags(tag_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_embeddings_model     ON embeddings(model);
    """,
    """
    CREATE INDEX IF NOT EXISTS files_present_path_idx   ON files(is_present, path);
    """,
)


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    columns = {row[1] for row in rows}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _fts_has_file_id_column(conn: sqlite3.Connection) -> bool:
    if not _table_exists(conn, "fts_files"):
        return False
    try:
        rows = conn.execute("PRAGMA table_info(fts_files)").fetchall()
    except sqlite3.DatabaseError:
        # FTS5 は table_info で空が返ることがあるので、保守的に旧式とみなす
        rows = []
    cols = {r[1] for r in rows} if rows else set()
    return "file_id" in cols  # 旧スキーマの目印


# ---- Migrations ----


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "files", "tagger_sig", "TEXT")
    _add_column_if_missing(conn, "files", "last_tagged_at", "REAL")


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "files", "is_present", "INTEGER NOT NULL DEFAULT 1")
    _add_column_if_missing(conn, "files", "deleted_at", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS files_present_path_idx ON files(is_present, path)")


def _migrate_to_v4(conn: sqlite3.Connection) -> None:
    """
    FTS5 を contentless 形式に移行（rowid = files.id）
    旧: fts_files(file_id UNINDEXED, text)
    新: fts_files(text, content='')
    """
    # まず FTS が無ければ新規作成だけ
    if not _table_exists(conn, "fts_files"):
        conn.execute("CREATE VIRTUAL TABLE fts_files USING fts5(text, content='', tokenize='unicode61')")
        return

    # 旧スキーマのときはデータを退避→作り直し→復元
    if _fts_has_file_id_column(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS __tmp_fts (rowid INTEGER PRIMARY KEY, text TEXT)")
        try:
            # 旧テーブルから退避（rowid に file_id を入れておく）
            conn.execute("INSERT INTO __tmp_fts(rowid, text) SELECT file_id, text FROM fts_files")
        except sqlite3.DatabaseError:
            # 失敗しても進められるように空で続行
            conn.execute("DELETE FROM __tmp_fts")

        conn.execute("DROP TABLE IF EXISTS fts_files")
        conn.execute("CREATE VIRTUAL TABLE fts_files USING fts5(text, content='', tokenize='unicode61')")
        # 復元（rowid 指定で files.id と一致させる）
        conn.execute("INSERT INTO fts_files(rowid, text) SELECT rowid, text FROM __tmp_fts")
        conn.execute("DROP TABLE IF EXISTS __tmp_fts")
    else:
        # 既に contentless っぽい。必要なら再作成したいが、ここでは何もしない
        pass


def _migrate_to_v5(conn: sqlite3.Connection) -> None:
    """
    将来用の軽微調整（インデックス最適化など）。
    ここでは特に何もしないが、user_version の連番維持のために空実装。
    """
    return


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    2: _migrate_to_v2,
    3: _migrate_to_v3,
    4: _migrate_to_v4,
    5: _migrate_to_v5,
}


def ensure_schema(conn: sqlite3.Connection, statements: Iterable[str] | None = None) -> None:
    """Ensure all database tables and indexes exist with the latest version."""
    cursor = conn.cursor()
    try:
        version_row = cursor.execute("PRAGMA user_version").fetchone()
        current_version = int(version_row[0]) if version_row else 0

        # 初期スキーマ適用
        for statement in statements or SCHEMA_STATEMENTS:
            cursor.execute(statement)

        target_version = CURRENT_SCHEMA_VERSION
        # 段階的にマイグレーション
        while current_version < target_version:
            next_version = current_version + 1
            migration = MIGRATIONS.get(next_version)
            if migration is not None:
                migration(conn)
            current_version = next_version

        # 最終版に合わせる
        cursor.execute(f"PRAGMA user_version = {target_version}")
        conn.commit()
    finally:
        cursor.close()


def apply_schema(conn: sqlite3.Connection, statements: Iterable[str] | None = None) -> None:
    """Backward compatible wrapper for :func:`ensure_schema`."""
    ensure_schema(conn, statements=statements)


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "SCHEMA_STATEMENTS",
    "apply_schema",
    "ensure_schema",
]
