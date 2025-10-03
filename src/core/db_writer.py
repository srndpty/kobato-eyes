# src/core/db_writer.py
from __future__ import annotations

import logging
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
        queue_size=1024,
        *,
        default_tagger_sig: str | None = None,
        unsafe_fast: bool = False,
        skip_fts: bool = False,
    ):
        super().__init__(name="DBWriter", daemon=True)
        self._log = logging.getLogger(__name__)
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
        self._unsafe_fast = bool(unsafe_fast)
        self._skip_fts = bool(skip_fts)
        # 環境変数で簡易デバッグ
        import os

        self._debug = os.environ.get("KE_DBWRITER_DEBUG") == "1"
        self._skip_fts = os.environ.get("KE_SKIP_FTS_DURING_TAG") == "1" or (self._fts_topk <= 0)
        self._stage_tags_in_temp = True

    # 現状値の観測用
    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except Exception:
            return -1

    def put(self, item: object, block=True, timeout=None):
        self._q.put(item, block=block, timeout=timeout)

    def raise_if_failed(self):
        if self._exc:
            raise self._exc

    @property
    def written(self) -> int:
        return self._written

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
            # quiesce 中でも取得できる特別接続
            conn = get_conn(self._db_path, allow_when_quiesced=True, timeout=120.0)

            if self._unsafe_fast:
                # ★ UNSAFE FAST MODE
                # 1) 充分に待つ設定を先に
                conn.execute("PRAGMA busy_timeout=60000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA cache_size=-262144")  # ~256MB
                try:
                    conn.execute("PRAGMA mmap_size=268435456")  # 256MB（対応環境でのみ）
                except Exception as e:
                    self._log.warning("DBWriter: pragma failed: (%s)", e)

                # 2) まず占有ロックを掴む（ここで待って良い）
                #    BEGIN EXCLUSIVE を挟むと確実に占有が取れる
                conn.execute("PRAGMA locking_mode=EXCLUSIVE")
                conn.execute("BEGIN EXCLUSIVE")
                conn.execute("COMMIT")

                # 3) WAL が残っていると journal 切替に失敗するので、念のため掃除
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception as e:
                    self._log.warning("DBWriter: wal_checkpoint failed: %s", e)

                # 4) ここでようやく MEMORY に切替（排他を握っているので通る）
                #    万一競合で失敗しても数回リトライ
                for _ in range(5):
                    try:
                        conn.execute("PRAGMA journal_mode=MEMORY")
                        break
                    except sqlite3.OperationalError as e:
                        if "locked" not in str(e).lower():
                            raise
                        time.sleep(0.25)
                # 5) 最後に fsync を切る
                conn.execute("PRAGMA synchronous=OFF")
            else:
                # 既存の WAL 高速設定（参考：以前の挙動を維持）
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=OFF")
                conn.execute("PRAGMA wal_autocheckpoint=0")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("PRAGMA temp_store=MEMORY")
                conn.execute("PRAGMA mmap_size=268435456")
                conn.execute("PRAGMA cache_size=-262144")

            # run() の UNSAFEブロック末尾あたり（PRAGMA設定の後）に：
            if self._unsafe_fast and self._stage_tags_in_temp:
                self._create_temp_staging(conn)

            try:
                self._loop(conn)
                # ここで一括マージ（UNSAFE+TEMPの時だけ）
                if self._unsafe_fast and self._stage_tags_in_temp:
                    self._merge_staging_into_persistent(conn)
            finally:
                conn.close()
        except BaseException as e:
            self._exc = e
            # ここで必ずログに残す（従来は沈黙していた）
            try:
                self._log.exception("DBWriter: crashed in run(): %s", e)
            finally:
                self._exc = e
            self._stop_evt.set()

    def _create_temp_staging(self, conn: sqlite3.Connection) -> None:
        # TEMP は temp_store=MEMORY ならRAM上。索引は後で作るのでここでは無し。
        conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS tmp_file_tags(
                file_id   INTEGER,
                tag_name  TEXT,
                score     REAL,
                category  INTEGER
            )
        """)
        conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS tmp_files_meta(
                file_id     INTEGER PRIMARY KEY,
                width       INTEGER,
                height      INTEGER,
                tagger_sig  TEXT,
                tagged_at   REAL
            )
        """)
        conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS tmp_tag_defs(
                name      TEXT PRIMARY KEY,
                category  INTEGER
            )
        """)

    def _wal_size_mb(self) -> int:
        import os

        try:
            return os.path.getsize(str(self._db_path) + "-wal") // (1024 * 1024)
        except OSError:
            return 0

    def _maybe_checkpoint(self, conn: sqlite3.Connection) -> None:
        wal_mb = self._wal_size_mb()
        # 256MB 超えたら即 PASSIVE
        if wal_mb >= 256:
            try:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
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
        """
        フラッシュ動作の全体像：
        - UNSAFE_FAST + TEMPステージング有効:
            └ 本テーブル(file_tags等)には一切触れず、TEMPテーブルに積むだけ
                → ディスクI/O最小、GPUを止めない
        - それ以外(従来/WAL):
            └ これまで通り tags upsert → file_tags DELETE→INSERT → (必要なら)FTS → files更新
        """
        if not items:
            return

        # ---------- ① UNSAFE_FAST + TEMPステージング ----------
        if self._unsafe_fast and getattr(self, "_stage_tags_in_temp", False):
            # TEMP だけをまとめてコミット（BEGIN IMMEDIATE でもよいが、TEMPのみなので通常 BEGIN で十分）
            conn.execute("BEGIN")
            now = time.time()

            # a) タグ定義（name, category）だけを集約（重複は IGNORE）
            defs_set: set[tuple[str, int]] = set()
            for it in items:
                for n, _s, c in it.tags:
                    defs_set.add((n, int(c)))
            if defs_set:
                conn.executemany(
                    "INSERT OR IGNORE INTO temp.tmp_tag_defs(name, category) VALUES(?, ?)",
                    list(defs_set),
                )

            # b) 画像ごとのタグ行（file_id, tag_name, score, category）
            tag_rows: list[tuple[int, str, float, int]] = []
            for it in items:
                for n, s, c in it.tags:
                    tag_rows.append((it.file_id, n, float(s), int(c)))
            if tag_rows:
                conn.executemany(
                    "INSERT INTO temp.tmp_file_tags(file_id, tag_name, score, category) " "VALUES(?, ?, ?, ?)",
                    tag_rows,
                )

            # c) files メタ（幅/高は None のときは据え置き。PK=file_id なので UPSERT）
            metas: list[tuple[int, int | None, int | None, str, float]] = []
            for it in items:
                sig = getattr(it, "tagger_sig", None) or self._default_tagger_sig
                ts = getattr(it, "tagged_at", None) or now
                metas.append((it.file_id, it.width, it.height, sig, ts))
            if metas:
                conn.executemany(
                    "INSERT INTO temp.tmp_files_meta(file_id, width, height, tagger_sig, tagged_at) "
                    "VALUES(?, ?, ?, ?, ?) "
                    "ON CONFLICT(file_id) DO UPDATE SET "
                    "  width       = COALESCE(excluded.width,  width), "
                    "  height      = COALESCE(excluded.height, height), "
                    "  tagger_sig  = excluded.tagger_sig, "
                    "  tagged_at   = excluded.tagged_at",
                    metas,
                )

            # d) FTS は完全スキップ（後段でオフライン再構築）
            conn.commit()
            self._written += len(items)
            if self._debug:
                self._log.debug(
                    "DBWriter: staged %d items to TEMP (written=%d, q=%d)", len(items), self._written, self.qsize()
                )
            return

        # ---------- ② 従来/WAL パス ----------
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

        tag_rows2: list[tuple[int, int, float]] = []
        for it in items:
            for n, s, _c in it.tags:
                tid = self._tag_cache.get(n)
                if tid is not None:
                    tag_rows2.append((it.file_id, int(tid), float(s)))
        if tag_rows2:
            conn.executemany(
                "INSERT INTO file_tags (file_id, tag_id, score) VALUES (?, ?, ?)",
                tag_rows2,
            )

        # 3) FTS（必要なら）
        if not self._skip_fts:
            fts_rows: list[tuple[int, str]] = []
            for it in items:
                top = sorted(it.tags, key=lambda t: t[1], reverse=True)[: self._fts_topk]
                text = " ".join([n for (n, _s, _c) in top])
                if text:
                    fts_rows.append((it.file_id, text))
            if fts_rows:
                fts_replace_rows(conn, fts_rows)

        # 4) files メタ更新（width/height は None なら据え置き）
        now = time.time()
        rows: list[tuple[int | None, int | None, str | None, float | None, int]] = []
        for it in items:
            sig = getattr(it, "tagger_sig", None) or self._default_tagger_sig
            ts = getattr(it, "tagged_at", None) or now
            rows.append((it.width, it.height, sig, ts, it.file_id))
        bulk_update_files_meta_by_id(conn, rows, coalesce_wh=True)

        conn.commit()
        self._written += len(items)
        self._maybe_checkpoint(conn)  # WALのときだけ意味がある
        if self._debug:
            self._log.debug(
                "DBWriter: flushed %d items (written=%d, qsize=%d)", len(items), self._written, self.qsize()
            )

    def _merge_staging_into_persistent(self, conn: sqlite3.Connection) -> None:
        # temp.tmp_file_tags が無ければ何もしない
        try:
            row = conn.execute("SELECT count(*) FROM temp.tmp_file_tags").fetchone()
        except sqlite3.OperationalError:
            return
        if not row or int(row[0]) == 0:
            return

        self._log.info("DBWriter: offline merge start (tmp->disk)")

        # 重い永続側インデックスは一旦落とす
        try:
            conn.execute("DROP INDEX IF EXISTS idx_file_tags_tag_score")
            conn.execute("DROP INDEX IF EXISTS idx_file_tags_tag_id")
        except Exception as e:
            self._log.warning("drop index failed: %s", e)

        conn.execute("BEGIN IMMEDIATE")
        try:
            # 1) tags の upsert（tmp_tag_defs から）
            rows = conn.execute("SELECT name, MAX(category) FROM temp.tmp_tag_defs GROUP BY name").fetchall()
            if rows:
                from db.repository import upsert_tags

                defs = [{"name": r[0], "category": int(r[1] or 0)} for r in rows]
                upsert_tags(conn, defs)

            # 2) 影響対象ファイルの既存タグを一括削除
            # ★ 先に消してから temp スキーマに明示して作成（TEMPは使わない）
            conn.execute("DROP TABLE IF EXISTS temp.tmp_file_ids")
            conn.execute("CREATE TABLE temp.tmp_file_ids AS " "SELECT DISTINCT file_id FROM temp.tmp_file_tags")
            conn.execute("DELETE FROM file_tags WHERE file_id IN (SELECT file_id FROM temp.tmp_file_ids)")

            # 3) 一時テーブル用の補助インデックス（temp を明示）
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_ft_name ON tmp_file_tags(tag_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_ft_file ON tmp_file_tags(file_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_fm_file ON tmp_files_meta(file_id)")

            # 4) tmp の名前→id を JOIN で解決して永続 file_tags に流し込む
            conn.execute("""
                INSERT INTO file_tags (file_id, tag_id, score)
                SELECT ft.file_id, t.id, ft.score
                FROM temp.tmp_file_tags AS ft
                JOIN tags AS t ON t.name = ft.tag_name
            """)

            # 5) files メタ一括更新
            conn.execute("""
                UPDATE files
                SET width          = COALESCE(width,  (SELECT m.width  FROM temp.tmp_files_meta m WHERE m.file_id = files.id)),
                    height         = COALESCE(height, (SELECT m.height FROM temp.tmp_files_meta m WHERE m.file_id = files.id)),
                    tagger_sig     = (SELECT m.tagger_sig FROM temp.tmp_files_meta m WHERE m.file_id = files.id),
                    last_tagged_at = (SELECT m.tagged_at  FROM temp.tmp_files_meta m WHERE m.file_id = files.id)
                WHERE id IN (SELECT file_id FROM temp.tmp_files_meta)
            """)

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            # 後始末（例外/キャンセルでも確実に消す）
            for ddl in (
                "DROP INDEX IF EXISTS temp.idx_tmp_ft_name",
                "DROP INDEX IF EXISTS temp.idx_tmp_ft_file",
                "DROP INDEX IF EXISTS temp.idx_tmp_fm_file",
                "DROP TABLE IF EXISTS temp.tmp_file_ids",
            ):
                try:
                    conn.execute(ddl)
                except Exception:
                    pass

        # 永続側インデックスを再作成
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id    ON file_tags(tag_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_tag_score ON file_tags(tag_id, score)")
            conn.commit()
        except Exception as e:
            self._log.warning("recreate index failed: %s", e)

        self._log.info("DBWriter: offline merge done")
