# src/core/db_writer.py
from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from db.connection import get_conn
from db.repository import bulk_update_files_meta_by_id, fts_replace_rows, upsert_tags

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
        flush_chunk=1024,
        fts_topk=128,
        queue_size=1024,
        *,
        default_tagger_sig: str | None = None,
        unsafe_fast: bool = False,
        skip_fts: bool = False,
        progress_cb: Optional[Callable[[str, int, int], None]] = None,
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
        self._progress_cb = progress_cb

    def _progress(self, kind: str, done: int, total: int) -> None:
        cb = self._progress_cb
        if cb:
            try:
                cb(kind, int(done), int(max(total, 1)))
            except Exception:
                pass

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

    def stop(self, *, flush: bool = True, timeout: float | None = 10.0, wait_forever: bool = False) -> None:
        """キューを閉じ、残りを flush してから停止。join まで行う。"""
        print("DBWriter: stopping...")
        try:
            if flush:
                self._q.put(DBFlush())
            self._q.put(DBStop())
        except Exception:
            pass
        self._stop_evt.set()

        if wait_forever:
            # 必ず止まるまで待つ（大規模マージ用）
            while self.is_alive():
                try:
                    self.join(timeout=1.0)
                except RuntimeError:
                    break
        else:
            try:
                self.join(timeout=timeout)
            except RuntimeError:
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
                # ← ここを追加
                if self._unsafe_fast:
                    try:
                        self._restore_normal_mode(conn)
                    except Exception as e:
                        self._log.warning("DBWriter: restore_normal_mode failed: %s", e)
                conn.close()
        except BaseException as e:
            self._exc = e
            # ここで必ずログに残す（従来は沈黙していた）
            try:
                self._log.exception("DBWriter: crashed in run(): %s", e)
            finally:
                self._exc = e
            self._stop_evt.set()

    def _restore_normal_mode(self, conn: sqlite3.Connection) -> None:
        # ここでは例外は潰す（best-effort）
        try:
            conn.execute("END")  # 念のためトランザクション強制終了
        except Exception:
            pass
        try:
            conn.execute("PRAGMA locking_mode=NORMAL")
        except Exception:
            pass
        try:
            # MEMORY は接続ローカルだが、いったんDELETEに戻してからWALへ
            conn.execute("PRAGMA journal_mode=DELETE")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass

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
        logger.debug("_flush called with", len(items), "items")
        if not items:
            return

        # ---------- ① UNSAFE_FAST + TEMPステージング ----------
        if self._unsafe_fast and getattr(self, "_stage_tags_in_temp", False):
            logger.debug("... using UNSAFE_FAST + TEMP staging path")
            # TEMP だけをまとめてコミット（BEGIN IMMEDIATE でもよいが、TEMPのみなので通常 BEGIN で十分）
            conn.execute("BEGIN")
            now = time.time()

            # a) タグ定義（name, category）だけを集約（重複は IGNORE）
            logger.debug("... inserting tag definitions into temp.tmp_tag_defs")
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
            logger.debug("... inserting file tags into temp.tmp_file_tags")
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
            logger.debug("... updating file metadata into temp.tmp_files_meta")
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
            logger.debug("... skipping FTS update (to be done offline later)")
            conn.commit()
            self._written += len(items)
            if self._debug:
                self._log.debug(
                    "DBWriter: staged %d items to TEMP (written=%d, q=%d)", len(items), self._written, self.qsize()
                )
            return

        # ---------- ② 従来/WAL パス ----------
        conn.execute("BEGIN IMMEDIATE")
        logger.debug("... using standard WAL path")

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

        total_tag_rows = int(row[0])
        total_file_rows = int(conn.execute("SELECT count(DISTINCT file_id) FROM temp.tmp_file_tags").fetchone()[0])
        total_meta_rows = int(conn.execute("SELECT count(*) FROM temp.tmp_files_meta").fetchone()[0])

        self._log.info("DBWriter: offline merge start (tmp->disk)")
        self._progress("merge.start", 0, total_tag_rows)

        # 永続側の重い索引は一旦落とす
        try:
            conn.execute("DROP INDEX IF EXISTS idx_file_tags_tag_score")
            conn.execute("DROP INDEX IF EXISTS idx_file_tags_tag_id")  # 存在しなくてもOK
        except Exception as e:
            self._log.warning("drop index failed: %s", e)

        conn.execute("BEGIN IMMEDIATE")
        try:
            # 1) tags upsert
            rows = conn.execute("SELECT name, MAX(category) FROM temp.tmp_tag_defs GROUP BY name").fetchall()
            if rows:
                from db.repository import upsert_tags

                defs = [{"name": r[0], "category": int(r[1] or 0)} for r in rows]
                upsert_tags(conn, defs)

            # 2) 影響ファイル抽出
            conn.execute("DROP TABLE IF EXISTS temp.tmp_file_ids")
            conn.execute("CREATE TABLE temp.tmp_file_ids AS " "SELECT DISTINCT file_id FROM temp.tmp_file_tags")

            # 3) 既存 file_tags の一括削除（チャンク実行）
            self._progress("merge.delete", 0, total_file_rows)
            CHUNK_F = 5_000
            done_f = 0
            cur = conn.execute("SELECT file_id FROM temp.tmp_file_ids")
            buf = []
            for (fid,) in cur:
                buf.append(fid)
                if len(buf) >= CHUNK_F:
                    ph = ",".join("?" * len(buf))
                    conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({ph})", buf)
                    done_f += len(buf)
                    self._progress("merge.delete", done_f, total_file_rows)
                    buf.clear()
            if buf:
                ph = ",".join("?" * len(buf))
                conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({ph})", buf)
                done_f += len(buf)
                self._progress("merge.delete", done_f, total_file_rows)

            # 4) temp 側補助インデックス（速やか）
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_ft_name ON tmp_file_tags(tag_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_ft_file ON tmp_file_tags(file_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS temp.idx_tmp_fm_file ON tmp_files_meta(file_id)")

            # 5) INSERT（JOIN）をチャンク実行（rowid 窓）
            self._progress("merge.insert", 0, total_tag_rows)
            CHUNK_T = 100_000
            done_t = 0
            # rowid は TEMP テーブルなら使える
            max_rowid = conn.execute("SELECT max(rowid) FROM temp.tmp_file_tags").fetchone()[0] or 0
            start = 0
            while start < max_rowid:
                end = min(start + CHUNK_T, max_rowid)
                conn.execute(
                    """
                    INSERT INTO file_tags (file_id, tag_id, score)
                    SELECT ft.file_id, t.id, ft.score
                    FROM temp.tmp_file_tags AS ft
                    JOIN tags AS t ON t.name = ft.tag_name
                    WHERE ft.rowid > ? AND ft.rowid <= ?
                """,
                    (start, end),
                )
                # 実際の増加行数は changes() でも取れるが、おおまかにチャンク単位で前進表示
                done_t = min(done_t + CHUNK_T, total_tag_rows)
                self._progress("merge.insert", done_t, total_tag_rows)
                start = end

            # 6) files メタ更新（チャンク：id で分割）
            self._progress("merge.update", 0, total_meta_rows)
            CHUNK_M = 20_000
            done_m = 0
            # rowid 窓で tmp_files_meta を流す
            max_mid = conn.execute("SELECT max(rowid) FROM temp.tmp_files_meta").fetchone()[0] or 0
            start = 0
            while start < max_mid:
                end = min(start + CHUNK_M, max_mid)
                conn.execute(
                    """
                    UPDATE files
                    SET width          = COALESCE(width,  (SELECT m.width  FROM temp.tmp_files_meta m WHERE m.file_id = files.id AND m.rowid > ? AND m.rowid <= ?)),
                        height         = COALESCE(height, (SELECT m.height FROM temp.tmp_files_meta m WHERE m.file_id = files.id AND m.rowid > ? AND m.rowid <= ?)),
                        tagger_sig     = (SELECT m.tagger_sig FROM temp.tmp_files_meta m WHERE m.file_id = files.id AND m.rowid > ? AND m.rowid <= ?),
                        last_tagged_at = (SELECT m.tagged_at  FROM temp.tmp_files_meta m WHERE m.file_id = files.id AND m.rowid > ? AND m.rowid <= ?)
                    WHERE id IN (SELECT file_id FROM temp.tmp_files_meta WHERE rowid > ? AND rowid <= ?)
                """,
                    (start, end, start, end, start, end, start, end, start, end),
                )
                done_m = min(done_m + CHUNK_M, total_meta_rows)
                self._progress("merge.update", done_m, total_meta_rows)
                start = end

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            # 後始末
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

        # 永続側インデックス再作成（進捗1/2 → 2/2 の形で通知）
        try:
            self._progress("merge.index", 0, 2)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_tag_id    ON file_tags(tag_id)")
            self._progress("merge.index", 1, 2)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_tags_tag_score ON file_tags(tag_id, score)")
            self._progress("merge.index", 2, 2)
            conn.commit()
        except Exception as e:
            self._log.warning("recreate index failed: %s", e)

        self._progress("merge.done", 1, 1)
        self._log.info("DBWriter: offline merge done")
