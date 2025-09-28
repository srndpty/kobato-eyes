# src/core/db_writer.py
from __future__ import annotations

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

    def __init__(self, db_path, flush_chunk=1024, fts_topk=128, queue_size=4096):
        super().__init__(name="DBWriter", daemon=True)
        self._db_path = db_path
        self._flush_chunk = int(flush_chunk)
        self._fts_topk = int(fts_topk)
        self._q: "queue.Queue[object]" = queue.Queue(maxsize=queue_size)
        self._exc: Optional[BaseException] = None
        self._stop_evt = threading.Event()
        self._written = 0
        self._tag_cache: Dict[str, int] = {}

    def put(self, item: object, block=True, timeout=None):
        self._q.put(item, block=block, timeout=timeout)

    def raise_if_failed(self):
        if self._exc:
            raise self._exc

    @property
    def written(self) -> int:
        return self._written

    def run(self):
        try:
            conn = get_conn(self._db_path)
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
        #   width/height は None なら据え置き（bulk_update_files_meta_by_id が coalesce）
        now = time.time()
        rows = []
        for it in items:
            rows.append((it.file_id, it.width, it.height, None, now))
        bulk_update_files_meta_by_id(conn, rows, coalesce_wh=True)

        conn.commit()
        self._written += len(items)
        # ログ
