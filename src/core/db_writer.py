# src/core/db_writer.py
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from db.connection import get_conn
from db.repository import fts_replace_rows, upsert_tags

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DBItem:
    # per file: (file_id, [(tag_name, score, category), ...]), width, height
    file_id: int
    tags: List[Tuple[str, float, int]]
    width: Optional[int]
    height: Optional[int]
    tagger_sig: Optional[str]
    tagged_at: Optional[float]


@dataclass(frozen=True)
class DBFlush:
    # 明示的フラッシュ要求（進捗同期用）
    pass


@dataclass(frozen=True)
class DBStop:
    # 終了シグナル
    pass


class DBWriter(threading.Thread):
    """
    推論スレッドから DBItem を受け取り、まとめて書く。
    - 専用 SQLite コネクション（WAL）
    - 大きめトランザクションでバルク書き
    """

    def __init__(
        self,
        db_path,
        flush_chunk=256,
        fts_topk=32,
        queue_size=4096,
        *,
        default_tagger_sig: str | None = None,
        sub_txn_size: int = 128,
        defer_fts: bool = True,  # FTS 遅延更新
        fts_batch: int = 512,  # まとめ書き件数
        fts_interval: float = 2.0,  # まとめ書き間隔
        unsafe_fastmode: bool = False,
        skip_fts: bool = False,
    ):
        super().__init__(name="DBWriter", daemon=True)
        self._db_path = db_path
        self._flush_chunk = int(flush_chunk)
        self._fts_topk = int(fts_topk)
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)
        self._exc: Optional[BaseException] = None
        self._stop_evt = threading.Event()
        self._written = 0
        self._tag_cache: Dict[str, int] = {}
        # files 更新用の既定 tagger_sig（アイテム未指定時に使用）
        self._default_tagger_sig = default_tagger_sig
        self._flush_count = 0
        self._sub_txn = max(32, int(sub_txn_size))
        # 診断用
        self._in_flush_evt = threading.Event()
        self._last_flush_summary: Dict[str, Any] = {}
        self._last_flush_started_at = 0.0
        # FTS 遅延
        self._defer_fts = bool(defer_fts)
        self._fts_buffer: Dict[int, str] = {}
        self._fts_batch = int(fts_batch)
        self._fts_interval = float(fts_interval)
        self._last_fts_flush = 0.0
        self._unsafe_fast = bool(unsafe_fastmode)
        print(f"self._unsafe_fast:{self._unsafe_fast}")
        self._skip_fts = bool(skip_fts)
        print(f"self._skip_fts:{self._skip_fts}")
        self._fts_backlog: set[int] = set()

    def put(self, item: object, block=True, timeout=None):
        self._q.put(item, block=block, timeout=timeout)

    def raise_if_failed(self):
        if self._exc:
            raise self._exc

    @property
    def written(self) -> int:
        return self._written

    @property
    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except Exception:
            return -1

    def stop(self, *, flush: bool = True, timeout: float | None = 10.0) -> None:
        """
        キューを閉じ、残りを flush してから停止。join まで行う。
        """
        try:
            if flush:
                # 先にフラッシュ要求を入れてから停止トークン
                self._q.put(DBFlush())
            self._q.put(DBStop())
        except Exception:
            # すでに終了/破棄されている等。join だけ試す。
            pass
        self._stop_evt.set()
        try:
            self.join(timeout=timeout)
        except RuntimeError:
            # すでに join 済みなど
            pass
        # スレッド内例外をメイン側へ
        self.raise_if_failed()

    def run(self):
        try:
            # ★ unsafe のときは最初から WAL を強制しない接続を取得
            conn = get_conn(self._db_path, force_wal=not self._unsafe_fast, allow_when_quiesced=True)

            logger.info("dbw: unsafe_fast=%s", self._unsafe_fast)
            # 書き込み向け PRAGMA
            if self._unsafe_fast:
                before = conn.execute("PRAGMA journal_mode").fetchone()[0]
                logger.info("dbw: unsafe_fast=True")
                logger.info("dbw: journal(before)=%s", before)
                ok = False
                for _ in range(60):
                    try:
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
                        conn.execute("PRAGMA journal_mode=DELETE")
                        mode = conn.execute("PRAGMA journal_mode=MEMORY").fetchone()[0]
                        if str(mode).lower() == "memory":
                            ok = True
                            break
                    except sqlite3.OperationalError as e:
                        logger.warning("dbw: switch to MEMORY retry: %s", e)
                        time.sleep(0.05)
                after = conn.execute("PRAGMA journal_mode").fetchone()[0]
                logger.info("dbw: journal(after)=%s", after)
                if not ok:
                    from db.connection import debug_dump_active_conns

                    logger.error("dbw: MEMORY切替失敗。オープン接続:\n%s", debug_dump_active_conns())
                    # フォールバック（高速だけど安全寄り）
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=OFF")
                else:
                    logger.warning("DBWriter running in UNSAFE FAST mode (journal=MEMORY, sync=OFF)")
                    # ★危険だが速い（FTS重め対策で RAM を厚めに）
                    conn.execute("PRAGMA journal_mode=MEMORY")
                    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
                    conn.execute("PRAGMA synchronous=OFF")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("PRAGMA cache_spill=OFF")
                    conn.execute("PRAGMA cache_size=-524288")  # 512MB ページキャッシュ
                    conn.execute("PRAGMA mmap_size=536870912")  # 512MB（対応環境のみ）
            else:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=OFF")
                conn.execute("PRAGMA wal_autocheckpoint=0")  # 自前で回す
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=1073741824")  # 1GB（対応環境のみ）
            conn.execute("PRAGMA cache_size=-524288")  # 512MB ページキャッシュ
            try:
                self._loop(conn)
            finally:
                conn.close()
        except BaseException as e:
            self._exc = e
            self._stop_evt.set()

    def _wal_size_mb(self) -> int:
        import os

        try:
            return os.path.getsize(str(self._db_path) + "-wal") // (1024 * 1024)
        except OSError:
            return 0

    def _maybe_checkpoint(self, conn: sqlite3.Connection) -> None:
        wal_mb = self._wal_size_mb()
        # 閾値: 大きめに（例 1024MB）。必要なら設定で変数化してもOK。
        if wal_mb >= 1024:
            try:
                logger.info("dbw.checkpoint: passive wal_mb>=%d", wal_mb)
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")  # 速い
            except Exception:
                pass
            return
        # 周期的に PASSIVE
        self._flush_count += 1
        if (self._flush_count % 2) == 0:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
        # アイドル時のみ TRUNCATE + optimize
        if (self._flush_count % 32) == 0 and self._q.empty():
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.execute("PRAGMA optimize")
            except Exception:
                pass

    def snapshot(self) -> Dict[str, Any]:
        """外部（TagStage ウォッチドッグ）から状態を読むための軽量 API。"""
        try:
            last = dict(self._last_flush_summary) if self._last_flush_summary else {}
        except Exception:
            last = {}
        return {
            "qsize": self._q.qsize(),
            "written": self._written,
            "wal_mb": self._wal_size_mb(),
            "in_flush": self._in_flush_evt.is_set(),
            "last_flush": last,
            "last_started_at": self._last_flush_started_at,
            "flush_count": self._flush_count,
        }

    def _flush_fts_buffer(self, conn: sqlite3.Connection):
        if not self._fts_buffer:
            return
        now = time.perf_counter()
        if (now - self._last_fts_flush) < self._fts_interval and not self._q.empty():
            return
        items = list(self._fts_buffer.items())[: self._fts_batch]
        for fid, _ in items:
            self._fts_buffer.pop(fid, None)
        if not items:
            return
        conn.execute("BEGIN IMMEDIATE")
        fts_replace_rows(conn, items)
        conn.commit()
        self._last_fts_flush = now

    def _loop(self, conn: sqlite3.Connection):
        batch: List[DBItem] = []
        last_report = time.perf_counter()

        while not self._stop_evt.is_set():
            try:
                t0 = time.perf_counter()
                msg = self._q.get(timeout=0.5)
                wait = time.perf_counter() - t0
                if wait > 0.5:
                    logger.info("dbw.loop: waited %.3fs for queue (qsize=%d)", wait, self.qsize)
            except queue.Empty:
                msg = None
            if msg is None:
                # タイムアウト：十分溜まっていればフラッシュ
                if batch:
                    self._flush(conn, batch)
                    batch.clear()
                if (time.perf_counter() - last_report) >= 5.0:
                    last_report = time.perf_counter()
                    logger.debug("dbw.loop: idle qsize=%d batch=%d written=%d", self.qsize, len(batch), self._written)
                continue
            if isinstance(msg, DBStop):
                if batch:
                    self._flush(conn, batch)
                    batch.clear()
                break
            if isinstance(msg, DBFlush):
                if batch:
                    self._flush(conn, batch)
                    batch.clear()
                continue
            if isinstance(msg, DBItem):
                batch.append(msg)
                if len(batch) >= self._flush_chunk:
                    self._flush(conn, batch)
                    batch.clear()

    def _flush(self, conn: sqlite3.Connection, items: Sequence[DBItem]):
        """大きなフラッシュを sub_txn ごとに分割してコミットする。"""
        if not items:
            return
        self._in_flush_evt.set()
        self._last_flush_started_at = time.perf_counter()
        wal_before = self._wal_size_mb()
        T_up = T_del = T_ins = T_fts = T_files = T_cmt = 0.0
        total_n = 0

        def _chunk(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i : i + n]

        # 1) tags 定義 upsert（全体で1回）
        t0 = time.perf_counter()
        new_defs = []
        for it in items:
            for n, _s, c in it.tags:
                if n not in self._tag_cache:
                    new_defs.append({"name": n, "category": int(c)})
        if new_defs:
            logger.info("dbw.upsert_tags: defs=%d cache=%d", len(new_defs), len(self._tag_cache))
            # ★ しきい値超でオープン接続のダンプ
            slow_guard_start = time.perf_counter()
            created = upsert_tags(conn, new_defs)  # ← 現状どおり
            took = time.perf_counter() - slow_guard_start
            if took > 2.0:
                try:
                    from db.connection import dump_open_conns

                    dump_open_conns("slow upsert_tags")
                except Exception:
                    pass
            self._tag_cache.update(created)
        T_up += time.perf_counter() - t0

        # 2) サブトランザクション単位で差分マージ
        for part in _chunk(list(items), self._sub_txn):
            total_n += len(part)
            file_ids = [it.file_id for it in part]
            # 2-1) 既存タグを一括取得
            existing: Dict[int, set] = {}
            for i in range(0, len(file_ids), 900):
                sub = file_ids[i : i + 900]
                ph = ",".join("?" for _ in sub)
                for fid, tid, sc in conn.execute(
                    f"SELECT file_id, tag_id, score FROM file_tags WHERE file_id IN ({ph})", sub
                ):
                    m = existing.setdefault(int(fid), {})
                    m[int(tid)] = float(sc)

            # 2-2) 新タグを構築
            new_map: Dict[int, List[Tuple[int, float]]] = {}
            new_ids: Dict[int, set] = {}
            for it in part:
                rows: List[Tuple[int, float]] = []
                ids = set()
                for n, s, _c in it.tags:
                    tid = self._tag_cache.get(n)
                    if tid is None:
                        continue
                    rows.append((int(tid), float(s)))
                    ids.add(int(tid))
                new_map[it.file_id] = rows
                new_ids[it.file_id] = ids

            SLOW = 2.0
            # ---- ここから計測対象 ----
            # BEGIN IMMEDIATE の待ち時間
            t0 = time.perf_counter()
            busy_logged = False
            while True:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as e:
                    # ロック以外はそのまま投げる
                    if "locked" not in str(e).lower():
                        raise
                    # 最初の一回だけ“誰が掴んでいるか”を吐く
                    if not busy_logged:
                        try:
                            from db.connection import debug_dump_active_conns

                            logger.warning("dbw.wait BEGIN: locked; open conns:\n%s", debug_dump_active_conns())
                        except Exception:
                            logger.warning("dbw.wait BEGIN: locked (no dump)")
                        busy_logged = True
                    time.sleep(0.05)
            begin_wait = time.perf_counter() - t0
            if begin_wait > SLOW:
                logger.warning("dbw.flush.wait BEGIN=%.3fs part=%d", begin_wait, len(part))

            # 2-3) 差分 DELETE（現状のまま）
            t_del0 = time.perf_counter()
            for fid in file_ids:
                old = set(existing.get(fid, {}).keys())
                keep = new_ids.get(fid, set())
                extra = old - keep
                if extra:
                    ph = ",".join("?" for _ in extra)
                    conn.execute(
                        f"DELETE FROM file_tags WHERE file_id = ? AND tag_id IN ({ph})",
                        (fid, *list(extra)),
                    )
            T_del += time.perf_counter() - t_del0

            # UPSERT の待ち時間
            up_rows: List[Tuple[int, int, float]] = []
            for fid, rows in new_map.items():
                old = existing.get(fid, {})
                for tid, score in rows:
                    prev = old.get(tid)
                    if prev is None or abs(prev - score) > 1e-6:
                        up_rows.append((fid, tid, score))

            t_ins_wait0 = time.perf_counter()
            if up_rows:
                conn.executemany(
                    "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)"
                    " ON CONFLICT(file_id, tag_id) DO UPDATE SET score=excluded.score",
                    up_rows,
                )
            ins_wait = time.perf_counter() - t_ins_wait0
            T_ins += ins_wait
            if ins_wait > SLOW:
                logger.warning("dbw.flush.wait INSERT=%.3fs rows=%d part=%d", ins_wait, len(up_rows), len(part))

            # 2-4) FTS／スキップ処理（現状のまま）
            t_fts0 = time.perf_counter()
            if not self._skip_fts and self._fts_topk > 0:
                ...
            else:
                for it in part:
                    self._fts_backlog.add(it.file_id)
            T_fts += time.perf_counter() - t_fts0

            # 2-5) files メタ（現状のまま）
            t_files0 = time.perf_counter()
            ...
            T_files += time.perf_counter() - t_files0

            # COMMIT の待ち時間
            t_cmt0 = time.perf_counter()
            conn.commit()
            cmt_wait = time.perf_counter() - t_cmt0
            T_cmt += cmt_wait
            if cmt_wait > SLOW:
                logger.warning("dbw.flush.wait COMMIT=%.3fs part=%d", cmt_wait, len(part))
            # ---- 計測ここまで ----

            # まとめ FTS の機会があればここで
            if self._defer_fts and not self._skip_fts:
                self._flush_fts_buffer(conn)
            time.sleep(0.002)
            self._maybe_checkpoint(conn)

        # 最後に残りの FTS バッファを吐き切る
        if self._defer_fts and not self._skip_fts:
            self._flush_fts_buffer(conn)

        self._written += len(items)
        total = time.perf_counter() - self._last_flush_started_at
        wal_after = self._wal_size_mb()
        self._last_flush_summary = {
            "n": total_n,
            "tag_upsert": round(T_up, 3),
            "del": round(T_del, 3),
            "ins": round(T_ins, 3),
            "fts": round(T_fts, 3),
            "files": round(T_files, 3),
            "commit": round(T_cmt, 3),
            "total": round(total, 3),
            "wal_mb_from": wal_before,
            "wal_mb_to": wal_after,
        }
        logger.info(
            "dbw.flush: n=%d tag_upsert=%.3f del=%.3f ins=%.3f fts=%.3f files=%.3f commit=%.3f total=%.3f wal_mb %d -> %d",
            total_n,
            T_up,
            T_del,
            T_ins,
            T_fts,
            T_files,
            T_cmt,
            total,
            wal_before,
            wal_after,
        )
        self._in_flush_evt.clear()
