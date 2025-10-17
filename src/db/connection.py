"""Connection helpers for the kobato-eyes SQLite database."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path
from threading import Lock

from db.schema import ensure_schema

logger = logging.getLogger(__name__)

_BOOTSTRAP_LOCK = Lock()
_BOOTSTRAPPED: set[str] = set()

_QUIESCE = False
_QUIESCE_LOCK = Lock()


def begin_quiesce() -> None:
    """UNSAFE 区間に入る。通常の新規接続を禁止する。"""
    global _QUIESCE
    with _QUIESCE_LOCK:
        _QUIESCE = True
    logger.info("begin_quiesce(): quiesce=%s", _QUIESCE)


def end_quiesce() -> None:
    """UNSAFE 区間を抜ける。通常接続を許可する。"""
    global _QUIESCE
    with _QUIESCE_LOCK:
        _QUIESCE = False
    logger.info("end_quiesce(): quiesce=%s", _QUIESCE)


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    """
    必要な索引を作成（存在すれば何もしない）。
    ※起動遅延を避けるため、環境変数でスキップ可能＆1本ずつ時間をログ出力。
    """
    import os
    import time

    if os.environ.get("KE_SKIP_INDEX_BUILD") == "1":
        logger.warning("Skip building indexes (KE_SKIP_INDEX_BUILD=1)")
        return

    # “重い”候補（巨大 file_tags で特に時間がかかる）
    HEAVY = {"idx_file_tags_tag_score"}
    skip_heavy = os.environ.get("KE_SKIP_HEAVY_INDEXES") == "1"

    def exists(name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def build(name: str, sql: str) -> None:
        if exists(name):
            logger.info("index: %-28s already exists; skip", name)
            return
        if skip_heavy and name in HEAVY:
            logger.warning("index: %-28s skipped (KE_SKIP_HEAVY_INDEXES=1)", name)
            return
        t0 = time.perf_counter()
        logger.info("index: %-28s begin", name)
        try:
            # インデックス作成を速める軽量PRAGMA（この接続内のみ）
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-262144")  # ~256MB
            conn.execute(sql)
            conn.commit()
        finally:
            dt = time.perf_counter() - t0
            logger.info("index: %-28s end (%.2fs)", name, dt)

    build("uq_tags_name", "CREATE UNIQUE INDEX IF NOT EXISTS uq_tags_name          ON tags(name)")
    build("idx_tags_category", "CREATE INDEX IF NOT EXISTS idx_tags_category            ON tags(category)")
    build("idx_file_tags_tag_id", "CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id         ON file_tags(tag_id)")
    build(
        "idx_file_tags_tag_score", "CREATE INDEX IF NOT EXISTS idx_file_tags_tag_score      ON file_tags(tag_id, score)"
    )
    build(
        "files_present_path_idx", "CREATE INDEX IF NOT EXISTS files_present_path_idx       ON files(is_present, path)"
    )
    build(
        "files_present_mtime_idx",
        "CREATE INDEX IF NOT EXISTS files_present_mtime_idx      ON files(is_present, mtime DESC)",
    )


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


def _exec_pragma_retry(cur: sqlite3.Cursor, sql: str, retries: int = 40, sleep_sec: float = 0.1) -> None:
    last: sqlite3.OperationalError | None = None
    for _ in range(retries):
        try:
            cur.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            last = exc
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            time.sleep(sleep_sec)
    if last is not None:
        raise last
    raise sqlite3.OperationalError(f"Failed to execute pragma after {retries} attempts: {sql}")


def _apply_pragmas(conn: sqlite3.Connection, *, is_memory: bool) -> None:
    cur = conn.cursor()
    try:
        # 既存
        _exec_pragma_retry(cur, "PRAGMA foreign_keys=ON;")
        if not is_memory:
            # WAL はDBに永続化される
            _exec_pragma_retry(cur, "PRAGMA journal_mode=WAL;")
        # ★ここから毎接続で効く高速化系（永続ではない）
        _exec_pragma_retry(cur, "PRAGMA synchronous=NORMAL;")  # COMMIT の fsync を軽く
        _exec_pragma_retry(cur, "PRAGMA temp_store=MEMORY;")  # tempをメモリへ
        _exec_pragma_retry(cur, "PRAGMA cache_size=-200000;")  # 約200MBのページキャッシュ
        _exec_pragma_retry(cur, "PRAGMA wal_autocheckpoint=50000;")  # WAL約200MBで自動チェックポイント
        try:
            _exec_pragma_retry(cur, "PRAGMA mmap_size=1073741824;")  # 1GBのmmap（対応環境のみ）
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
) -> sqlite3.Connection:
    conn = sqlite3.connect(
        target,
        detect_types=sqlite3.PARSE_DECLTYPES,
        timeout=timeout,  # busy_timeout 互換
        uri=uri,
    )
    # _apply_runtime_pragmas(conn, is_memory=is_memory)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn, is_memory=is_memory)  # ★追加
    return conn


def bootstrap_if_needed(db_path: str | Path, *, timeout: float = 30.0) -> None:
    """Create the database file and ensure the schema exists for the given path."""
    target, is_memory, uri, display_path, fs_path = _resolve_db_target(db_path)

    # :memory: / file::memory: は起動都度スキーマ適用のみ
    if is_memory:
        logger.info("Bootstrapping schema at: %s", display_path)
        # t0 = time.perf_counter()
        wal = f"{display_path}-wal"
        shm = f"{display_path}-shm"
        try:
            logger.info(
                "DB sizes: db=%s wal=%s shm=%s",
                os.path.getsize(display_path) if os.path.exists(display_path) else 0,
                os.path.getsize(wal) if os.path.exists(wal) else 0,
                os.path.getsize(shm) if os.path.exists(shm) else 0,
            )
        except Exception:
            pass
        return

    with _BOOTSTRAP_LOCK:
        if target in _BOOTSTRAPPED:
            return

        is_new = False
        if fs_path is not None:
            fs_path.parent.mkdir(parents=True, exist_ok=True)
            is_new = not fs_path.exists() or (fs_path.exists() and os.path.getsize(fs_path) == 0)

        conn = _connect_to_target(target, timeout=timeout, uri=uri, is_memory=is_memory)
        try:
            logger.info("Bootstrapping schema at: %s", display_path)

            # 新規 DB っぽいときに page_size を先に指定（テーブル作成前が安全）
            if is_new:
                try:
                    conn.execute("PRAGMA page_size = 8192;")
                except sqlite3.DatabaseError:
                    pass

            t1 = time.perf_counter()
            logger.info("ensure_schema: begin")
            ensure_schema(conn)
            logger.info("ensure_schema: end (%.2fs)", time.perf_counter() - t1)

            t2 = time.perf_counter()
            logger.info("_ensure_indexes: begin")
            _ensure_indexes(conn)
            logger.info("_ensure_indexes: end (%.2fs)", time.perf_counter() - t2)

            # optimize は重くなる個体があるので、一旦オフにできるスイッチ
            if os.environ.get("KE_SKIP_OPTIMIZE") != "1":
                t3 = time.perf_counter()
                try:
                    conn.execute("PRAGMA optimize;")
                except sqlite3.DatabaseError:
                    pass
                logger.info("optimize: end (%.2fs)", time.perf_counter() - t3)
            else:
                logger.info("optimize: skipped by KE_SKIP_OPTIMIZE=1")
        finally:
            conn.close()
        _BOOTSTRAPPED.add(target)


def get_conn(
    db_path: str | Path,
    *,
    timeout: float = 30.0,
    allow_when_quiesced: bool = False,
) -> sqlite3.Connection:
    """Create a SQLite connection with WAL and foreign-key support enabled."""
    # quiesce 中は通常の新規接続を拒否（DBWriter など専用だけ allow）
    if _QUIESCE and not allow_when_quiesced:
        raise RuntimeError("DB is quiesced (UNSAFE fast mode active)")
    target, is_memory, uri, _, _ = _resolve_db_target(db_path)
    bootstrap_if_needed(db_path, timeout=timeout)
    conn = _connect_to_target(target, timeout=timeout, uri=uri, is_memory=is_memory)
    # :memory: は都度スキーマ適用
    if is_memory:
        ensure_schema(conn)
        _ensure_indexes(conn)
    return conn


__all__ = ["bootstrap_if_needed", "get_conn", "begin_quiesce", "end_quiesce"]
