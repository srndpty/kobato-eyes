"""Database schema management for kobato-eyes."""

from __future__ import annotations

import sqlite3
from typing import Iterable

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        size INTEGER,
        mtime REAL,
        sha256 TEXT,
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
)


def apply_schema(conn: sqlite3.Connection, statements: Iterable[str] | None = None) -> None:
    """Create database objects required by the application."""
    cursor = conn.cursor()
    try:
        for statement in statements or SCHEMA_STATEMENTS:
            cursor.execute(statement)
    finally:
        cursor.close()
    conn.commit()


__all__ = ["apply_schema", "SCHEMA_STATEMENTS"]
