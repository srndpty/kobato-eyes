"""Connection helpers for the kobato-eyes SQLite database."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import traceback
from pathlib import Path
from threading import Lock, RLock

from db.schema import ensure_schema

logger = logging.getLogger(__name__)

_BOOTSTRAP_LOCK = Lock()
_BOOTSTRAPPED: set[str] = set()

# ==== 接続トラッキング / 静穏化ゲート ====
_QUIESCE = threading.Event()  # True の間は新規 get_conn をブロック
_ACTIVE_LOCK = RLock()
_ACTIVE_COUNT = 0
_ACTIVE_CONNS: dict[int, dict] = {}  # id(conn) -> {thread, target, created_at, stack}


def _register_conn(conn: sqlite3.Connection, target: str) -> None:
    global _ACTIVE_COUNT
    info = {
        "thread": threading.current_thread().name,
        "target": target,
        "created_at": time.time(),
        "stack": "".join(traceback.format_stack(limit=12)),
    }
    with _ACTIVE_LOCK:
        _ACTIVE_COUNT += 1
        _ACTIVE_CONNS[id(conn)] = info


def _unregister_conn(conn: sqlite3.Connection) -> None:
    global _ACTIVE_COUNT
    with _ACTIVE_LOCK:
        _ACTIVE_COUNT = max(0, _ACTIVE_COUNT - 1)
        _ACTIVE_CONNS.pop(id(conn), None)


class _TrackedConnection(sqlite3.Connection):
    def close(self):  # type: ignore[override]
        try:
            super().close()
        finally:
            _unregister_conn(self)


def begin_quiesce(timeout: float = 15.0) -> bool:
    """新規接続を止め、既存接続が 0 になるのを待つ。"""
    _QUIESCE.set()
    t0 = time.time()
    while True:
        with _ACTIVE_LOCK:
            if _ACTIVE_COUNT == 0:
                return True
        if time.time() - t0 >= timeout:
            return False
        time.sleep(0.05)


def end_quiesce() -> None:
    _QUIESCE.clear()


def debug_dump_active_conns() -> str:
    with _ACTIVE_LOCK:
        lines = [f"QUIESCE={_QUIESCE.is_set()} active={_ACTIVE_COUNT}"]
        now = time.time()
        for cid, meta in _ACTIVE_CONNS.items():
            age = now - meta["created_at"]
            lines.append(f"- conn_id={cid} thr={meta['thread']} age={age:.1f}s")
            # スタックは先頭数行だけ
            stack_lines = meta["stack"].splitlines()
            for sl in stack_lines[-6:]:
                lines.append(f"    {sl}")
        return "\n".join(lines)


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    """必要な索引を作成（存在すれば何もしない）"""
    conn.executescript(
        """
        -- 検索・集計高速化用
        CREATE UNIQUE INDEX IF NOT EXISTS uq_tags_name          ON tags(name);
        CREATE INDEX IF NOT EXISTS idx_tags_category            ON tags(category);

        -- file_tags は (file_id, tag_id) の PRIMARY KEY があるので file_id 先頭の探索は担保される
        CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id         ON file_tags(tag_id);
        CREATE INDEX IF NOT EXISTS idx_file_tags_tag_score      ON file_tags(tag_id, score);

        -- files
        CREATE INDEX IF NOT EXISTS files_present_path_idx       ON files(is_present, path);
        CREATE INDEX IF NOT EXISTS files_present_mtime_idx      ON files(is_present, mtime DESC);

        -- embeddings
        CREATE INDEX IF NOT EXISTS idx_embeddings_model         ON embeddings(model);
        """
    )
    # 遅いので除外、別途どこかでやりたい
    # try:
    #     conn.execute("ANALYZE")
    # except sqlite3.DatabaseError:
    #     pass


def _resolve_db_target(db_path: str | Path) -> tuple[str, bool, bool, str, Path | None]:
    """Normalize database paths and derive connection options."""
    if isinstance(db_path, Path):
        candidate = db_path.expanduser()
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate.absolute()
        s = str(resolved)
        return s, False, False, s, resolved

    text_path = str(db_path)
    if text_path == ":memory":
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


def _apply_runtime_pragmas(conn: sqlite3.Connection, *, is_memory: bool) -> None:
    """接続ごとに適用する安全寄りの高速化 PRAGMA"""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    if is_memory:
        # テストや一時 DB
        try:
            conn.execute("PRAGMA journal_mode = MEMORY;")
        except sqlite3.DatabaseError:
            pass
    else:
        # 本番 DB：WAL + NORMAL で安全＆高速
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
        except sqlite3.DatabaseError:
            pass
        try:
            conn.execute("PRAGMA synchronous = NORMAL;")
        except sqlite3.DatabaseError:
            pass

    # 共有設定
    for pragma in (
        "PRAGMA temp_store = MEMORY;",
        # cache_size: 負数=KB 指定。-65536=64MB
        "PRAGMA cache_size = -65536;",
        # 256MB の mmap（対応環境のみ）
        "PRAGMA mmap_size = 268435456;",
    ):
        try:
            conn.execute(pragma)
        except sqlite3.DatabaseError:
            pass


def _apply_pragmas(conn: sqlite3.Connection, *, is_memory: bool, force_wal: bool = True) -> None:
    cur = conn.cursor()
    try:
        # 既存
        cur.execute("PRAGMA foreign_keys=ON;")
        if not is_memory and force_wal:
            # WAL はDBに永続化される
            cur.execute("PRAGMA journal_mode=WAL;")
        # ★ここから毎接続で効く高速化系（永続ではない）
        cur.execute("PRAGMA synchronous=NORMAL;")  # COMMIT の fsync を軽く
        cur.execute("PRAGMA temp_store=MEMORY;")  # tempをメモリへ
        cur.execute("PRAGMA cache_size=-200000;")  # 約200MBのページキャッシュ
        cur.execute("PRAGMA wal_autocheckpoint=50000;")  # WAL約200MBで自動チェックポイント
        try:
            cur.execute("PRAGMA mmap_size=1073741824;")  # 1GBのmmap（対応環境のみ）
        except sqlite3.DatabaseError:
            pass
        # さらに攻めるなら（任意）
        # cur.execute("PRAGMA cache_spill=FALSE;")
    finally:
        cur.close()


def _connect_to_target(
    target: str,
    *,
    timeout: float,
    uri: bool,
    is_memory: bool,
    force_wal: bool = True,
    allow_when_quiesced: bool = False,
) -> sqlite3.Connection:
    # 静穏期間中は原則ブロック（DBWriterなど例外のみ通す）
    if _QUIESCE.is_set() and not allow_when_quiesced:
        # quiesce解除まで待機
        while _QUIESCE.is_set():
            time.sleep(0.05)
    conn = sqlite3.connect(
        target,
        detect_types=sqlite3.PARSE_DECLTYPES,
        timeout=timeout,  # busy_timeout 互換
        uri=uri,
        factory=_TrackedConnection,
    )
    # _apply_runtime_pragmas(conn, is_memory=is_memory)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn, is_memory=is_memory, force_wal=force_wal)  # ★追加
    _register_conn(conn, target)
    return conn


def bootstrap_if_needed(db_path: str | Path, *, timeout: float = 30.0, force_wal: bool = False) -> None:
    """Create the database file and ensure the schema exists for the given path."""
    target, is_memory, uri, display_path, fs_path = _resolve_db_target(db_path)

    # :memory: / file::memory: は起動都度スキーマ適用のみ
    if is_memory:
        logger.info("Bootstrapping schema at: %s", display_path)
        return

    with _BOOTSTRAP_LOCK:
        if target in _BOOTSTRAPPED:
            return

        is_new = False
        if fs_path is not None:
            fs_path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not fs_path.exists() or (fs_path.exists() and os.path.getsize(fs_path) == 0)

        # ブートストラップ時は安全寄り（WALを有効）で OK
        conn = _connect_to_target(target, timeout=timeout, uri=uri, is_memory=is_memory, force_wal=force_wal)
        try:
            logger.info("Bootstrapping schema at: %s", display_path)

            # 新規 DB っぽいときに page_size を先に指定（テーブル作成前が安全）
            if is_new:
                try:
                    conn.execute("PRAGMA page_size = 8192;")
                except sqlite3.DatabaseError:
                    pass

            ensure_schema(conn)  # ここで必要なテーブル・インデックス・マイグレーション
            _ensure_indexes(conn)  # 追加の実運用向け索引
            try:
                conn.execute("PRAGMA optimize;")
            except sqlite3.DatabaseError:
                pass
        finally:
            conn.close()
        _BOOTSTRAPPED.add(target)


def get_conn(
    db_path: str | Path,
    *,
    timeout: float = 30.0,
    force_wal: bool = True,
    allow_when_quiesced=False,
) -> sqlite3.Connection:
    """Create a SQLite connection with WAL and foreign-key support enabled."""
    target, is_memory, uri, _, _ = _resolve_db_target(db_path)
    bootstrap_if_needed(db_path, timeout=timeout, force_wal=force_wal)
    conn = _connect_to_target(
        target,
        timeout=timeout,
        uri=uri,
        is_memory=is_memory,
        force_wal=force_wal,
        allow_when_quiesced=allow_when_quiesced,
    )
    # :memory: は都度スキーマ適用
    if is_memory:
        ensure_schema(conn)
        _ensure_indexes(conn)
    return conn


__all__ = ["bootstrap_if_needed", "get_conn", "begin_quiesce", "end_quiesce"]
