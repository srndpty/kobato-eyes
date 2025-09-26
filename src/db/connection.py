"""Connection helpers for the kobato-eyes SQLite database."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from threading import Lock

from db.schema import ensure_schema

logger = logging.getLogger(__name__)

_BOOTSTRAP_LOCK = Lock()
_BOOTSTRAPPED: set[str] = set()


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    """本番DB/メモリDBともに必要な索引を作成する（存在すれば何もしない）"""
    conn.executescript("""
    -- 検索・集計高速化用
    CREATE INDEX IF NOT EXISTS idx_file_tags_tag           ON file_tags(tag_id);
    CREATE INDEX IF NOT EXISTS idx_file_tags_tag_score     ON file_tags(tag_id, score);
    CREATE INDEX IF NOT EXISTS idx_file_tags_tag_file      ON file_tags(tag_id, file_id);
    CREATE UNIQUE INDEX IF NOT EXISTS uq_tags_name         ON tags(name);
    CREATE INDEX IF NOT EXISTS idx_tags_category           ON tags(category);
    """)
    # 初回だけで十分だが、呼ばれても副作用は小さい
    try:
        conn.execute("ANALYZE")
    except sqlite3.DatabaseError:
        pass


def _resolve_db_target(db_path: str | Path) -> tuple[str, bool, bool, str, Path | None]:
    """Normalize database paths and derive connection options."""

    if isinstance(db_path, Path):
        candidate = db_path.expanduser()
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate.absolute()
        return str(resolved), False, False, str(resolved), resolved

    text_path = str(db_path)
    if text_path == ":memory":
        # Compatibility with sqlite's URI form "file::memory:".
        text_path = ":memory:"
    if text_path == ":memory:":
        return text_path, True, False, text_path, None
    if text_path.startswith("file:"):
        is_memory = "mode=memory" in text_path or text_path == "file::memory:" or text_path.startswith("file::memory:")
        return text_path, is_memory, True, text_path, None

    candidate = Path(text_path).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        resolved = candidate.absolute()
    resolved_str = str(resolved)
    return resolved_str, False, False, resolved_str, resolved


def _connect_to_target(
    target: str,
    *,
    timeout: float,
    uri: bool,
    is_memory: bool,
) -> sqlite3.Connection:
    conn = sqlite3.connect(
        target,
        detect_types=sqlite3.PARSE_DECLTYPES,
        timeout=timeout,
        uri=uri,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    if not is_memory:
        conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def bootstrap_if_needed(db_path: str | Path, *, timeout: float = 30.0) -> None:
    """Create the database file and ensure the schema exists for the given path."""

    target, is_memory, uri, display_path, fs_path = _resolve_db_target(db_path)

    if is_memory:
        logger.info("Bootstrapping schema at: %s", display_path)
        return

    with _BOOTSTRAP_LOCK:
        if target in _BOOTSTRAPPED:
            return
        if fs_path is not None:
            fs_path.parent.mkdir(parents=True, exist_ok=True)
        conn = _connect_to_target(target, timeout=timeout, uri=uri, is_memory=is_memory)
        try:
            logger.info("Bootstrapping schema at: %s", display_path)
            ensure_schema(conn)
            _ensure_indexes(conn)  # ★ ここで索引を作る
        finally:
            conn.close()
        _BOOTSTRAPPED.add(target)


def get_conn(
    db_path: str | Path,
    *,
    timeout: float = 30.0,
) -> sqlite3.Connection:
    """Create a SQLite connection with WAL and foreign-key support enabled."""

    target, is_memory, uri, _, _ = _resolve_db_target(db_path)
    bootstrap_if_needed(db_path, timeout=timeout)
    conn = _connect_to_target(target, timeout=timeout, uri=uri, is_memory=is_memory)
    if is_memory:
        ensure_schema(conn)
        _ensure_indexes(conn)  # ★ ここで索引を作る
    return conn


__all__ = ["bootstrap_if_needed", "get_conn"]
