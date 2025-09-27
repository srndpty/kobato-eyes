"""Database schema management for kobato-eyes."""

from __future__ import annotations

import sqlite3
from typing import Callable, Iterable

CURRENT_SCHEMA_VERSION = 3

SCHEMA_STATEMENTS: tuple[str, ...] = (
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
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        category INTEGER DEFAULT 0
    );
    """,
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
    """
    CREATE TABLE IF NOT EXISTS tagger_thresholds (
        category TEXT PRIMARY KEY,
        threshold REAL NOT NULL
    );
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_files USING fts5(
        file_id UNINDEXED,
        text,
        tokenize = 'unicode61'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS signatures (
        file_id INTEGER PRIMARY KEY,
        phash_u64 INTEGER NOT NULL,
        dhash_u64 INTEGER NOT NULL,
        FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
    );
    """,
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
    """
    CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id ON file_tags(tag_id);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
    """,
    """
    CREATE INDEX IF NOT EXISTS files_path_idx ON files(path);
    """,
    # !!! ここに "files_is_present_path_idx" は置かない（v3 migration が作る）
)


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    cursor = conn.execute("PRAGMA table_info(%s)" % table)
    try:
        columns = {row[1] for row in cursor.fetchall()}
    finally:
        cursor.close()
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _migrate_to_v2(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "files", "tagger_sig", "TEXT")
    _add_column_if_missing(conn, "files", "last_tagged_at", "REAL")


def _migrate_to_v3(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "files", "is_present", "INTEGER NOT NULL DEFAULT 1")
    conn.execute("CREATE INDEX IF NOT EXISTS files_is_present_path_idx ON files(is_present, path)")


MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    2: _migrate_to_v2,
    3: _migrate_to_v3,
}


def ensure_schema(conn: sqlite3.Connection, statements: Iterable[str] | None = None) -> None:
    """Ensure all database tables and indexes exist with the latest version."""

    cursor = conn.cursor()
    try:
        version_row = cursor.execute("PRAGMA user_version").fetchone()
        current_version = int(version_row[0]) if version_row else 0
        for statement in statements or SCHEMA_STATEMENTS:
            cursor.execute(statement)
        target_version = CURRENT_SCHEMA_VERSION
        while current_version < target_version:
            next_version = current_version + 1
            migration = MIGRATIONS.get(next_version)
            if migration is not None:
                migration(conn)
            current_version = next_version
        if current_version != target_version:
            current_version = target_version
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
