# src/core/db_writer.py
from __future__ import annotations

import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from db.connection import get_conn
from db.repository import bulk_update_files_meta_by_id, fts_replace_rows, upsert_tags


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
        flush_chunk=1024,
        fts_topk=128,
        queue_size=16384,  # ← メモリに余裕があれば大きめ推奨（必要なら元に戻してOK）
        *,
        default_tagger_sig: str | None = None,
        wal_soft_cap_mb: int = 256,  # ← ここで上限を小さめに（状況で上げ下げ可）
        cache_mb: int = 256,  # ← SQLite ページキャッシュ（RAM）を厚めに
        checkpoint_every: int = 2,  # ← 2 フラッシュごとに PASSIVE
        truncate_idle_every: int = 32,  # ← アイドル時 TRUNCATE 周期
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
        # checkpoint / pacing
        self._flush_count = 0
        self._checkpoint_every = int(checkpoint_every)
        self._truncate_idle_every = int(truncate_idle_every)
        self._wal_soft_cap_mb = int(wal_soft_cap_mb)
        self._cache_kb = int(cache_mb) * 1024
        self._base_flush_chunk = int(flush_chunk)
        self._queue_capacity = queue_size

    def put(self, item: object, block=True, timeout=None):
        self._q.put(item, block=block, timeout=timeout)

    def raise_if_failed(self):
        if self._exc:
            raise self._exc

    @property
    def written(self) -> int:
        return self._written

    # パイプラインが水位監視に使うための簡易API
    def qsize_safe(self) -> int | None:
        try:
            return self._q.qsize()
        except NotImplementedError:
            return None

    @property
    def queue_capacity(self) -> int:
        return self._queue_capacity

    # 明示フラッシュ（DBFlush を投入するだけ）
    def flush(self, force: bool = True) -> None:
        try:
            self._q.put(DBFlush())
        except Exception:
            pass

    def run(self):
        try:
            # ここで書き込み向け PRAGMA（順番バグ修正：conn 取得後に設定）
            try:
                conn = get_conn(self._db_path)
                # 書き込み向け PRAGMA（ここで確実に設定）
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=OFF")
                conn.execute("PRAGMA wal_autocheckpoint=0")  # -- 自前で checkpoint
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA mmap_size=268435456")  # -- 256MB
                conn.execute(f"PRAGMA cache_size={-self._cache_kb}")  # -- KB 指定（負値）
                conn.execute("PRAGMA journal_size_limit=1073741824")  # -- 1GB cap.
            except Exception:
                pass
            try:
                self._loop(conn)
            finally:
                conn.close()
        except BaseException as e:
            self._exc = e
            self._stop_evt.set()

    def _loop(self, conn: sqlite3.Connection):
        batch: List[DBItem] = []
        while not self._stop_evt.is_set():
            try:
                msg = self._q.get(timeout=0.5)
            except queue.Empty:
                msg = None
            if msg is None:
                # タイムアウト：十分溜まっていればフラッシュ
                if batch:
                    self._flush(conn, batch)
                    batch.clear()
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

    def _flush(self, conn: sqlite3.Connection, items: Sequence[DBItem]):
        if not items:
            return
        conn.execute("BEGIN IMMEDIATE")

        # 1) tags upsert（新規だけ抽出）＋ cache 更新
        new_defs = []
        for it in items:
            for n, _s, c in it.tags:
                if n not in self._tag_cache:
                    new_defs.append({"name": n, "category": int(c)})
        if new_defs:
            created = upsert_tags(conn, new_defs)
            self._tag_cache.update(created)

        # 2) file_tags: DELETE → INSERT（チャンク分割）
        file_ids = [it.file_id for it in items]
        if file_ids:
            for i in range(0, len(file_ids), 900):
                chunk = file_ids[i : i + 900]
                ph = ",".join(["?"] * len(chunk))
                conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({ph})", chunk)

        tag_rows: List[Tuple[int, int, float]] = []
        for it in items:
            for n, s, _c in it.tags:
                tid = self._tag_cache.get(n)
                if tid is not None:
                    tag_rows.append((it.file_id, int(tid), float(s)))
        if tag_rows:
            conn.executemany(
                "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
                tag_rows,
            )

        # 3) FTS: REPLACE 相当
        fts_rows: List[Tuple[int, str]] = []
        for it in items:
            # 事前に tags はスコア順で入れておく or ここで簡易 sort
            top = sorted(it.tags, key=lambda t: t[1], reverse=True)[: self._fts_topk]
            text = " ".join([n for (n, _s, _c) in top])
            if text:
                fts_rows.append((it.file_id, text))
        if fts_rows:
            fts_replace_rows(conn, fts_rows)

        # 4) files: width/height/tagger_sig/last_tagged_at
        #   - width/height は None → 据え置き
        #   - tagger_sig はアイテムに無ければ default を使う（必ず埋める）
        now = time.time()
        rows: list[tuple[int | None, int | None, str | None, float | None, int]] = []
        for it in items:
            sig = getattr(it, "tagger_sig", None) or self._default_tagger_sig
            ts = getattr(it, "tagged_at", None) or now
            # bulk_update_files_meta_by_id の期待順序:
            # (width, height, tagger_sig, last_tagged_at, file_id)
            rows.append((it.width, it.height, sig, ts, it.file_id))
        bulk_update_files_meta_by_id(conn, rows, coalesce_wh=True)

        conn.commit()
        self._written += len(items)
        self._flush_count += 1
        self._maybe_checkpoint(conn)
        self._maybe_retarget_flush_chunk(conn)
        # 任意の観測ログ（うるさければ削除）
        try:
            if (self._flush_count % 8) == 0:
                qsz = self._q.qsize()
                wal = self._wal_size_mb(conn)
                import logging

                logging.getLogger(__name__).info(
                    "FLUSH #%d written=%d q=%d wal=%dMB chunk=%d",
                    self._flush_count,
                    self._written,
                    qsz,
                    wal,
                    self._flush_chunk,
                )
        except Exception:
            pass

    def _wal_size_mb(self, conn: sqlite3.Connection) -> int:
        # wal ファイルサイズ（存在しなければ 0）
        wal_path = str(self._db_path) + "-wal"
        try:
            return int(os.path.getsize(wal_path) // (1024 * 1024))
        except OSError:
            return 0

    def _maybe_checkpoint(self, conn: sqlite3.Connection) -> None:
        wal_now = self._wal_size_mb(conn)
        # 1) ソフト上限超えなら即 PASSIVE（reader が居ても進む範囲で消化）
        if wal_now >= self._wal_soft_cap_mb:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
            return
        # 2) 周期的に PASSIVE
        if self._checkpoint_every > 0 and (self._flush_count % self._checkpoint_every) == 0:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except Exception:
                pass
        # 3) さらに長周期でアイドルなら TRUNCATE
        if self._truncate_idle_every > 0 and (self._flush_count % self._truncate_idle_every) == 0:
            # キューが空＝アイドル時だけ TRUNCATE（ブロッキングを避ける）
            if self._q.empty():
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
                try:
                    conn.execute("PRAGMA optimize")
                except Exception:
                    pass

    def _maybe_retarget_flush_chunk(self, conn: sqlite3.Connection) -> None:
        """WAL の膨張に応じて flush_chunk を一時的に絞る/戻す。"""
        try:
            wal = self._wal_size_mb(conn)
            if wal >= self._wal_soft_cap_mb:
                # バースト気味：半分に絞る（下限 256）
                self._flush_chunk = max(256, self._flush_chunk // 2)
            elif self._q.empty() and (self._flush_count % 16 == 0):
                # 余裕がある：ゆっくり元の値へ
                if self._flush_chunk < self._base_flush_chunk:
                    self._flush_chunk = min(self._base_flush_chunk, self._flush_chunk * 2)
        except Exception:
            pass
