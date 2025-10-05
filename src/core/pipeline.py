"""File processing pipeline ensuring watcher events update indices."""

from __future__ import annotations

import hashlib
import logging
import os
import queue
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

from core.db_writer import DBWriter
from db.fts_offline import rebuild_fts_offline
from utils.env import is_headless

if is_headless():

    class QObject:  # type: ignore[too-many-ancestors]
        """Minimal stub used when Qt is unavailable."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - Qt-compatible signature
            pass

        def deleteLater(self) -> None:  # noqa: D401 - Qt-compatible signature
            pass


else:  # pragma: no branch - trivial import guard
    from PyQt6.QtCore import QObject

from core.config import load_settings
from core.jobs import BatchJob, JobManager
from core.scanner import DEFAULT_EXTENSIONS, iter_images
from core.settings import PipelineSettings
from core.tag_job import TagJobConfig, run_tag_job
from core.watcher import DirectoryWatcher
from db.connection import bootstrap_if_needed, get_conn
from db.repository import fts_delete_rows, get_file_by_path, list_untagged_under_path, mark_indexed_at, upsert_file
from dup.indexer import DuplicateIndexer, EmbedderProtocol, add_embeddings_to_hnsw, load_hnsw_index, save_hnsw_index
from index.hnsw import HNSWIndex
from sig.phash import dhash, phash
from tagger.base import ITagger, TagCategory, TagResult
from utils.hash import compute_sha256
from utils.paths import ensure_dirs, get_db_path, get_index_dir

logger = logging.getLogger(__name__)


# ============================================================
# PrefetchLoader: 画像パス列を「(paths, images)」のバッチにして先読み
# ============================================================
class PrefetchLoader:
    def __init__(
        self,
        paths: Sequence[str],
        *,
        batch_size: int,
        prefetch_batches: int = 2,
        io_workers: int | None = None,
    ) -> None:
        self._paths = paths
        self._B = int(batch_size)
        self._depth = max(1, int(prefetch_batches))
        cpu = os.cpu_count() or 4
        self._workers = max(1, int(io_workers or min(8, cpu)))
        self._q: "queue.Queue[Tuple[list[str], list[Image.Image]] | None]" = queue.Queue(self._depth)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._producer, name="ke-prefetch", daemon=True)
        self._th.start()
        logger.info("PrefetchLoader: start (B=%d, depth=%d, io_workers=%d)", self._B, self._depth, self._workers)

    def _load_one(self, p: str) -> Image.Image | None:
        try:
            # ファイルハンドルは即クローズ。decode を完了させてRAMへ載せる
            with Image.open(p) as im:
                im.load()  # デコードをここで完了させる
                return im.convert("RGB")  # 透明合成は tagger 側でも行うので RGBでOK
        except Exception as e:
            logger.warning("PrefetchLoader: failed to load %s: %s", p, e)
            return None

    def _producer(self) -> None:
        try:
            with ThreadPoolExecutor(max_workers=self._workers, thread_name_prefix="ke-io") as ex:
                N = len(self._paths)
                for i in range(0, N, self._B):
                    if self._stop.is_set():
                        break
                    batch_paths = list(self._paths[i : i + self._B])
                    # I/O を並列に投げる
                    futs = {ex.submit(self._load_one, p): p for p in batch_paths}
                    imgs: list[Image.Image] = []
                    for fut in as_completed(futs):
                        img = fut.result()
                        if img is not None:
                            imgs.append(img)
                    # ここで順序は問いません（推論的に問題なし）。順序維持したい場合は gather 方式に変える
                    self._q.put((batch_paths, imgs))
                    if self._stop.is_set():
                        break
        finally:
            # 終端マーカー
            try:
                self._q.put(None, timeout=1)
            except Exception:
                pass

    def __iter__(self) -> Iterator[Tuple[list[str], list[Image.Image]]]:
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        try:
            # 早期終了時にキューを開放
            while True:
                self._q.get_nowait()
        except Exception:
            pass
        if self._th.is_alive():
            self._th.join(timeout=2.0)
        logger.info("PrefetchLoader: stop")


class PrefetchLoaderPrepared:
    """
    画像を並列ロードして前処理（PIL→RGB、正方形化＋リサイズ）まで済ませ、
    (paths, np_batch, sizes) を事前に用意して供給するローダ。
      - paths: List[str]（バッチ内のファイルパス。順序は元の順）
      - np_batch: np.ndarray 形状 (B, H, W, 3), float32
      - sizes: List[Tuple[int,int]] 元画像の (width, height)
    """

    def __init__(
        self,
        paths: List[str],
        *,
        tagger,  # WD14Tagger インスタンス（prepare_batch_pil を呼ぶ）
        batch_size: int,
        prefetch_batches: int = 2,
        io_workers: int | None = None,
    ) -> None:
        self._paths = list(paths)
        self._B = int(batch_size)
        self._depth = max(1, int(prefetch_batches))
        cpu = os.cpu_count() or 4
        self._io_workers: int = max(1, int(io_workers or min(8, cpu)))  # ← int のまま保持
        self._tagger = tagger

        # (paths, np_batch, sizes) or None(sentinal)
        self._q: "queue.Queue[tuple[list[str], np.ndarray, list[tuple[int,int]]] | None]" = queue.Queue(self._depth)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._producer, name="PL-Feeder", daemon=True)
        self._th.start()
        logger.info(
            "PrefetchLoaderPrepared: start (B=%d, depth=%d, io_workers=%d)", self._B, self._depth, self._io_workers
        )

    # 個別ロード（PIL）
    def _load_one(self, p: str) -> tuple[str, Image.Image | None, tuple[int, int] | None]:
        try:
            with Image.open(p) as im:
                im.load()  # デコードを完了（ハンドル即解放）
                w, h = im.size
                rgb = im.convert("RGB")  # 透明合成は tagger 側でOKだが、簡便にRGBへ
                return (p, rgb, (w, h))
        except Exception as e:
            logger.warning("PrefetchLoaderPrepared: failed to load %s: %s", p, e)
            return (p, None, None)

    def _producer(self) -> None:
        try:
            N = len(self._paths)
            with ThreadPoolExecutor(max_workers=self._io_workers, thread_name_prefix="ke-io") as ex:
                for i in range(0, N, self._B):
                    if self._stop.is_set():
                        break

                    batch_paths = self._paths[i : i + self._B]
                    # 並列ロード（順序は futures から辞書に集めて後で整える）
                    futs = [ex.submit(self._load_one, p) for p in batch_paths]
                    # 収集（まずは path -> (PIL or None, size or None)）
                    tmp: dict[str, tuple[Image.Image | None, tuple[int, int] | None]] = {}
                    for fut in as_completed(futs):
                        p, pil, sz = fut.result()
                        tmp[p] = (pil, sz)

                    # 入力量産（順序を batch_paths で揃える）
                    pil_list: list[Image.Image] = []
                    sizes: list[tuple[int, int]] = []
                    kept_paths: list[str] = []
                    for p in batch_paths:
                        pil, sz = tmp.get(p, (None, None))
                        if pil is None or sz is None:
                            continue
                        pil_list.append(pil)
                        sizes.append(sz)
                        kept_paths.append(p)

                    if not pil_list:
                        # 空でも終端はまだ先なので継続
                        continue

                    # 前処理（PIL -> np.ndarray (B,H,W,3) float32）
                    try:
                        np_batch = self._tagger.prepare_batch_pil(pil_list)
                    finally:
                        # PIL バッファ解放（np_batchは tagger 側で確保済み）
                        for im in pil_list:
                            try:
                                im.close()
                            except Exception:
                                pass

                    # キューへ
                    self._q.put((kept_paths, np_batch, sizes))
                    if self._stop.is_set():
                        break
        finally:
            try:
                self._q.put(None, timeout=1)
            except Exception:
                pass

    def __iter__(self) -> Iterator[Tuple[list[str], np.ndarray, list[tuple[int, int]]]]:
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except Exception:
            pass
        if self._th.is_alive():
            self._th.join(timeout=2.0)
        logger.info("PrefetchLoaderPrepared: stop")


class IndexPhase(Enum):
    """Phases reported during ``run_index_once`` progress updates."""

    SCAN = "scan"
    TAG = "tag"
    EMBED = "embed"
    HNSW = "hnsw"
    FTS = "fts"
    DONE = "done"


@dataclass
class IndexProgress:
    """Structured payload describing indexing progress."""

    phase: IndexPhase
    done: int
    total: int
    message: str | None = None


class _FileProcessJob(BatchJob):
    def __init__(
        self,
        paths: Iterable[Path],
        *,
        db_path: Path,
        tagger: ITagger,
        embedder: EmbedderProtocol,
        hnsw_index: HNSWIndex,
        model_name: str,
        tag_config: TagJobConfig,
    ) -> None:
        super().__init__(list(paths))
        self._db_path = db_path
        self._tagger = tagger
        self._embedder = embedder
        self._index = hnsw_index
        self._model_name = model_name
        self._tag_config = tag_config
        self._conn = None
        self._indexer: DuplicateIndexer | None = None

    def prepare(self) -> None:
        self._conn = get_conn(self._db_path)
        self._indexer = DuplicateIndexer(
            self._conn,
            self._embedder,
            self._index,
            model_name=self._model_name,
        )
        logger.debug("Prepared processing job for %d files", len(self.items))

    def load_item(self, item: Path) -> Path:
        return item

    def process_item(self, item: Path, loaded: Path) -> tuple[Path, Optional[int]] | None:
        if self._conn is None or self._indexer is None:
            logger.error("Job not prepared before processing")
            return None
        if not loaded.exists():
            logger.info("Skipping missing file: %s", loaded)
            return None
        try:
            output = run_tag_job(self._tagger, loaded, self._conn, config=self._tag_config)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Tagging failed for %s", loaded)
            return None
        file_id = output.file_id if output else None
        try:
            indexed_ids = self._indexer.index_paths([loaded])
            if indexed_ids and file_id is None:
                file_id = indexed_ids[0]
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Indexing failed for %s", loaded)
        return (loaded, file_id)

    def write_item(self, item: Path, processed: tuple[Path, Optional[int]] | None) -> None:
        if processed is None:
            return
        _, file_id = processed
        if file_id is None:
            logger.warning("No file ID resolved after processing %s", item)
        else:
            logger.debug("Processed file %s (id=%s)", item, file_id)

    def cleanup(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        logger.debug("Processing job finished")


class ProcessingPipeline(QObject):
    """Drive watcher events through tagging and duplicate indexing."""

    def __init__(
        self,
        *,
        db_path: Path,
        tagger: ITagger,
        embedder: EmbedderProtocol,
        hnsw_index: HNSWIndex,
        job_manager: JobManager,
        settings: PipelineSettings | None = None,
    ) -> None:
        super().__init__()
        resolved_db = Path(db_path).expanduser()
        db_literal = str(resolved_db)
        if not db_literal.startswith("file:") and db_literal != ":memory:":
            ensure_dirs()
            resolved_db.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = resolved_db
        self._tagger = tagger
        self._embedder = embedder
        self._index = hnsw_index
        self._job_manager = job_manager
        self._settings = settings or PipelineSettings()
        self._watcher: DirectoryWatcher | None = None
        self._scheduled: Set[Path] = set()
        self._tag_config = TagJobConfig()
        self._tagger_sig: str | None = None
        self._refresh_tag_config()

    def _refresh_tag_config(self) -> None:
        thresholds = _build_threshold_map(self._settings.tagger.thresholds)
        max_tags_map = _build_max_tags_map(getattr(self._settings.tagger, "max_tags", None))
        self._tagger_sig = current_tagger_sig(
            self._settings,
            thresholds=thresholds,
            max_tags=max_tags_map,
        )
        self._tag_config = TagJobConfig(
            thresholds=thresholds or None,
            max_tags=max_tags_map or None,
            tagger_sig=self._tagger_sig,
        )

    def update_settings(self, settings: PipelineSettings) -> None:
        logger.info("Updating pipeline settings: %s", settings)
        was_running = self._watcher is not None and self._watcher.is_running()
        if self._watcher is not None:
            self.stop()
        self._settings = settings
        self._refresh_tag_config()
        if self._settings.auto_index and (was_running or self._watcher is None):
            self.start()

    def start(self) -> None:
        if self._watcher is not None:
            return
        if not self._settings.auto_index:
            logger.info("Auto indexing disabled; watcher not started")
            return
        if not self._settings.roots:
            logger.warning("Cannot start pipeline without roots")
            return
        self._watcher = DirectoryWatcher(
            self._settings.roots,
            callback=self.enqueue_index,
            excluded=self._settings.excluded,
            extensions=self._settings.allow_exts or DEFAULT_EXTENSIONS,
            parent=self,
        )
        self._watcher.start()
        logger.info("Processing pipeline started")

    def stop(self) -> None:
        if self._watcher is not None:
            self._watcher.flush_pending()
            self._watcher.stop()
            self._watcher.deleteLater()
            self._watcher = None
        self._scheduled.clear()
        logger.info("Processing pipeline stopped")

    def shutdown(self) -> None:
        self.stop()
        self._job_manager.shutdown()

    def enqueue_path(self, path: Path) -> None:
        self.enqueue_index([path])

    def enqueue_index(self, paths: Iterable[Path]) -> None:
        bootstrap_if_needed(self._db_path)
        allow_exts = {ext.lower() for ext in (self._settings.allow_exts or DEFAULT_EXTENSIONS)}
        scheduled_now: set[Path] = set()
        resolved_paths: list[Path] = []
        for path in paths:
            candidate = Path(path)
            suffix = candidate.suffix.lower()
            if not suffix or suffix not in allow_exts:
                continue
            try:
                resolved = candidate.expanduser().resolve()
            except OSError:
                resolved = candidate.expanduser().absolute()
            if resolved in self._scheduled or resolved in scheduled_now:
                continue
            scheduled_now.add(resolved)
            resolved_paths.append(resolved)
        if not resolved_paths:
            return
        self._scheduled.update(scheduled_now)
        logger.debug("Scheduling processing for %d file(s)", len(resolved_paths))
        job = _FileProcessJob(
            resolved_paths,
            db_path=self._db_path,
            tagger=self._tagger,
            embedder=self._embedder,
            hnsw_index=self._index,
            model_name=self._settings.model_name,
            tag_config=self._tag_config,
        )
        signals = self._job_manager.submit(job)

        def _on_finished() -> None:
            for resolved in resolved_paths:
                self._scheduled.discard(resolved)

        signals.finished.connect(_on_finished)


def scan_and_tag(
    root: Path,
    *,
    recursive: bool = True,
    batch_size: int = 8,
    skip_hnsw: bool = True,
    hard_delete_missing: bool = False,
) -> dict[str, object]:
    """Tag unprocessed images within ``root`` without updating embeddings or HNSW."""

    start_time = time.perf_counter()
    resolved_root = Path(root).expanduser()
    try:
        resolved_root = resolved_root.resolve(strict=False)
    except OSError:
        resolved_root = resolved_root.absolute()
    stats_out: dict[str, object] = {
        "queued": 0,
        "tagged": 0,
        "elapsed_sec": 0.0,
        "missing": 0,
        "soft_deleted": 0,
        "hard_deleted": 0,
    }

    if not resolved_root.exists():
        logger.info("Manual tag refresh skipped; path does not exist: %s", resolved_root)
        return stats_out

    settings = load_settings()
    allow_exts = {ext.lower() for ext in (settings.allow_exts or DEFAULT_EXTENSIONS)}
    if resolved_root.is_file() and resolved_root.suffix.lower() not in allow_exts:
        elapsed = time.perf_counter() - start_time
        stats_out["elapsed_sec"] = elapsed
        logger.info("Manual tag refresh skipped; unsupported file type: %s", resolved_root)
        return stats_out
    thresholds = _build_threshold_map(settings.tagger.thresholds)
    max_tags_map = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))
    tagger_sig = current_tagger_sig(settings, thresholds=thresholds, max_tags=max_tags_map)
    db_path = get_db_path()
    bootstrap_if_needed(db_path)
    conn = get_conn(db_path, allow_when_quiesced=True)
    tagger: ITagger | None = None

    def _like_pattern(path: Path) -> str:
        literal = str(path)
        if path.is_dir():
            if literal.endswith(("/", "\\")):
                return f"{literal}%"
            separator = "\\" if "\\" in literal and "/" not in literal else "/"
            return f"{literal}{separator}%"
        return literal

    def _within_scope(candidate: Path) -> bool:
        if resolved_root.is_file():
            return candidate == resolved_root
        if candidate == resolved_root:
            return False
        try:
            relative = candidate.relative_to(resolved_root)
        except ValueError:
            return False
        if not recursive and len(relative.parts) > 1:
            return False
        return True

    try:
        if not skip_hnsw:
            logger.info("scan_and_tag operates in tagging-only mode; embeddings/HNSW are skipped.")

        queued_paths: list[Path] = []
        seen: set[str] = set()
        fs_paths: set[str] = set()

        for _, stored_path in list_untagged_under_path(conn, _like_pattern(resolved_root)):
            path_obj = Path(stored_path)
            if not path_obj.exists():
                continue
            if not _within_scope(path_obj):
                continue
            if path_obj.suffix.lower() not in allow_exts:
                continue
            literal = str(path_obj)
            fs_paths.add(literal)
            if literal in seen:
                continue
            queued_paths.append(path_obj)
            seen.add(literal)

        if resolved_root.is_file():
            fs_iterable: Iterable[Path] = [resolved_root]
        else:
            fs_iterable = iter_images([resolved_root], excluded=[], extensions=allow_exts)

        for candidate in fs_iterable:
            path_obj = Path(candidate)
            if not _within_scope(path_obj):
                continue
            if path_obj.suffix.lower() not in allow_exts:
                continue
            literal = str(path_obj)
            fs_paths.add(literal)
            if literal in seen:
                continue
            row = get_file_by_path(conn, literal)
            if row is None:
                queued_paths.append(path_obj)
                seen.add(literal)
                continue
            cursor = conn.execute(
                "SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1",
                (row["id"],),
            )
            try:
                if cursor.fetchone() is None:
                    queued_paths.append(path_obj)
                    seen.add(literal)
            finally:
                cursor.close()

        total = len(queued_paths)
        stats_out["queued"] = total

        def _chunked(items: Sequence[int], size: int = 900) -> Iterator[list[int]]:
            for index in range(0, len(items), size):
                yield list(items[index : index + size])

        missing_ids: list[int] = []
        cursor = None
        try:
            if resolved_root.is_file():
                cursor = conn.execute(
                    "SELECT id, path FROM files WHERE is_present = 1 AND path = ?",
                    (str(resolved_root),),
                )
            else:
                cursor = conn.execute(
                    "SELECT id, path FROM files WHERE is_present = 1 AND path LIKE ?",
                    (_like_pattern(resolved_root),),
                )
            for row in cursor.fetchall():
                path_text = str(row["path"])
                candidate = Path(path_text)
                if not _within_scope(candidate):
                    continue
                if path_text not in fs_paths:
                    missing_ids.append(int(row["id"]))
        finally:
            if cursor is not None:
                cursor.close()

        missing_count = len(missing_ids)
        stats_out["missing"] = missing_count
        soft_deleted = 0
        hard_deleted_count = 0
        if missing_ids:
            logger.info("Manual tag refresh: marking %d missing file(s) under %s", missing_count, resolved_root)
            if hard_delete_missing:
                with conn:
                    for chunk in _chunked(missing_ids):
                        placeholders = ", ".join("?" for _ in chunk)
                        conn.execute(
                            f"DELETE FROM file_tags WHERE file_id IN ({placeholders})",
                            chunk,
                        )
                        conn.execute(
                            f"DELETE FROM embeddings WHERE file_id IN ({placeholders})",
                            chunk,
                        )
                        conn.execute(
                            f"DELETE FROM signatures WHERE file_id IN ({placeholders})",
                            chunk,
                        )
                        # conn.execute(
                        #     f"DELETE FROM fts_files WHERE rowid IN ({placeholders})",
                        #     chunk,
                        # )
                        # countless対応版
                        fts_delete_rows(conn, chunk)
                        conn.execute(
                            f"DELETE FROM files WHERE id IN ({placeholders})",
                            chunk,
                        )
                hard_deleted_count = missing_count
            else:
                with conn:
                    for chunk in _chunked(missing_ids):
                        placeholders = ", ".join("?" for _ in chunk)
                        conn.execute(
                            f"UPDATE files SET is_present = 0, deleted_at = CURRENT_TIMESTAMP WHERE id IN ({', '.join('?' for _ in chunk)})",
                            tuple(chunk),
                        )
                        # conn.execute(
                        #     f"DELETE FROM fts_files WHERE rowid IN ({placeholders})",
                        #     chunk,
                        # )
                        # countless対応
                        fts_delete_rows(conn, chunk)
                soft_deleted = missing_count
        stats_out["soft_deleted"] = soft_deleted
        stats_out["hard_deleted"] = hard_deleted_count

        if total == 0:
            elapsed = time.perf_counter() - start_time
            stats_out["elapsed_sec"] = elapsed
            if missing_count:
                logger.info(
                    "Manual tag refresh: removed %d missing file(s) (hard_delete=%s) in %.2fs",
                    missing_count,
                    hard_delete_missing,
                    elapsed,
                )
            else:
                logger.info(
                    "Manual tag refresh: no untagged files under %s (elapsed %.2fs)",
                    resolved_root,
                    elapsed,
                )
            return stats_out

        logger.info(
            "Manual tag refresh: tagging %d file(s) under %s (recursive=%s)",
            total,
            resolved_root,
            recursive,
        )

        tagger = _resolve_tagger(
            settings,
            None,
            thresholds=thresholds,
            max_tags=max_tags_map,
        )
        config = TagJobConfig(
            thresholds=thresholds or None,
            max_tags=max_tags_map or None,
            tagger_sig=tagger_sig,
        )

        tagged = 0
        for index, path_obj in enumerate(queued_paths, start=1):
            logger.info(
                "Manual tag refresh progress: %d/%d %s",
                index,
                total,
                path_obj,
            )
            try:
                result = run_tag_job(tagger, path_obj, conn, config=config)
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Tagging failed during refresh for %s", path_obj)
                continue
            if result is None:
                continue
            tagged += 1
            if batch_size > 0 and tagged % max(batch_size, 1) == 0:
                logger.info("Manual tag refresh tagged %d/%d file(s)", tagged, total)

        elapsed = time.perf_counter() - start_time
        stats_out["tagged"] = tagged
        stats_out["elapsed_sec"] = elapsed
        logger.info(
            "Manual tag refresh complete: tagged %d of %d file(s) in %.2fs (missing removed=%d, hard=%s)",
            tagged,
            total,
            elapsed,
            missing_count,
            hard_delete_missing,
        )
        return stats_out
    finally:
        conn.close()
        if tagger is not None:
            closer = getattr(tagger, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - defensive logging
                    logger.exception("Failed to close tagger after manual refresh")


@dataclass
class _FileRecord:
    """Metadata captured for a file during a one-shot indexing run."""

    file_id: int
    path: Path
    size: int
    mtime: float
    sha: str
    is_new: bool
    changed: bool
    tag_exists: bool
    embed_exists: bool
    needs_tagging: bool
    needs_embedding: bool
    stored_tagger_sig: str | None = None
    current_tagger_sig: str | None = None
    last_tagged_at: float | None = None
    image: Image.Image | None = None
    width: int | None = None
    height: int | None = None
    load_failed: bool = False


def wait_for_unlock(db_path: str, timeout: float = 15.0) -> bool:
    """短命接続で軽く叩き、ロックが抜けるのを待つ（最大 timeout 秒）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with sqlite3.connect(db_path, timeout=1.0) as c:
                # 何か軽いクエリで OK（PRAGMA でも SELECT でも可）
                c.execute("SELECT 1")
                return True
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower():
                # ロック以外は待っても無意味
                return True
        time.sleep(0.25)
    return False


def _settle_after_quiesce(db_path: str, progress_cb=None) -> None:
    logger.info("_settle_after_quiesce (best-effort)")
    # 1) ロックが抜けるのを最大15秒だけ待つ（抜けなくても続行）
    ok = wait_for_unlock(db_path, timeout=15.0)
    if not ok:
        logger.warning("settle: DB still locked; proceeding best-effort")

    # 2) 掃除専用の短命接続で実行（既存接続に PRAGMA を当てない）
    try:
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            cur = conn.cursor()
            # 少し広めに
            cur.execute("PRAGMA busy_timeout=30000")
            # MEMORY→WAL への切替は強要しない。チェックポイント＆最適化だけ
            try:
                cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError as e:
                logger.warning("settle: wal_checkpoint failed: %s", e)
            try:
                cur.execute("PRAGMA optimize")
            except sqlite3.OperationalError as e:
                logger.warning("settle: optimize failed: %s", e)
            conn.commit()
    except Exception as e:
        logger.warning("settle: sweep connection failed: %s", e)

    # 3) ほんの少しだけ間を空けると Windows だと安定することが多い
    time.sleep(0.2)


def run_index_once(
    db_path: str | Path,
    settings: PipelineSettings | None = None,
    *,
    tagger_override: ITagger | None = None,
    embedder_override: EmbedderProtocol | None = None,
    progress_cb: Callable[[IndexProgress], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Perform a full indexing pass across all configured roots (fast batched version)."""

    start_time = time.perf_counter()
    settings = settings or load_settings()
    bootstrap_if_needed(db_path)
    ensure_dirs()

    stats: dict[str, object] = {
        "scanned": 0,
        "new_or_changed": 0,
        "tagged": 0,
        "signatures": 0,
        "embedded": 0,
        "hnsw_added": 0,
        "elapsed_sec": 0.0,
        "tagger_name": settings.tagger.name,
        "retagged": 0,
        "cancelled": False,
    }

    thresholds = _build_threshold_map(settings.tagger.thresholds)
    max_tags_map = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))
    tagger_sig = current_tagger_sig(settings, thresholds=thresholds, max_tags=max_tags_map)
    # serialised_thresholds = _serialise_thresholds(thresholds)
    # serialised_max_tags = _serialise_max_tags(max_tags_map)
    stats["tagger_sig"] = tagger_sig

    last_emit: dict[IndexPhase, tuple[int, float]] = {}
    # last_messages: dict[IndexPhase, str | None] = {}
    progress_sink = progress_cb

    def _emit(progress: IndexProgress, *, force: bool = False) -> None:
        nonlocal progress_sink
        if progress_sink is None:
            return
        now = time.perf_counter()
        last = last_emit.get(progress.phase)
        should = force or last is None
        if not should and last is not None:
            last_done, last_time = last
            if progress.total > 0:
                step = max(1, progress.total // 100)
                should = (progress.done >= progress.total) or ((progress.done - last_done) >= step)
            should = should or ((now - last_time) >= 0.1)
        if should:
            last_emit[progress.phase] = (progress.done, now)
            try:
                progress_sink(progress)
            except RuntimeError as e:
                # PyQt側QObject破棄後など。以後emit無効化
                logger.warning("Progress callback is gone: %s; stop emitting further updates.", e)
                progress_sink = None
            except Exception:
                # 予期せぬ例外も“以後emitしない”で安全側に
                logger.exception("Progress callback raised; disabling further updates.")
                progress_sink = None

    def _should_cancel() -> bool:
        if is_cancelled is None:
            return False
        try:
            return bool(is_cancelled())
        except Exception:
            logger.exception("Cancellation callback failed")
            return False

    roots = [Path(root).expanduser() for root in settings.roots if root]
    roots = [r for r in roots if r.exists()]
    if not roots:
        logger.info("No valid roots configured; skipping indexing run.")
        stats["elapsed_sec"] = time.perf_counter() - start_time
        return stats

    excluded_paths = [Path(p).expanduser() for p in settings.excluded if p]
    allow_exts = {ext.lower() for ext in (settings.allow_exts or DEFAULT_EXTENSIONS)}
    batch_size = max(1, 32)
    # batch_size = max(1, settings.batch_size)
    # commit_every = max(1, getattr(settings, "commit_every", 512))  # 追加: バッチ書き込み閾値

    _emit(IndexProgress(phase=IndexPhase.SCAN, done=0, total=-1, message="start"), force=True)

    records: list[_FileRecord] = []
    hnsw_additions: list[tuple[int, np.ndarray]] = []

    db_literal = str(db_path)
    conn = get_conn(
        db_literal if (db_literal.startswith("file:") or db_literal == ":memory:") else Path(db_path).expanduser(),
        allow_when_quiesced=True,
    )
    cancelled = False
    dbw: "DBWriter | None" = None
    quiesced = False

    try:
        logger.info("Scanning %d root(s) for eligible images", len(roots))
        # より速い「変更判定」: size/mtime が一致なら SHA を計算しない
        for image_path in iter_images(roots, excluded=excluded_paths, extensions=allow_exts):
            if cancelled or _should_cancel():
                cancelled = True
                break
            stats["scanned"] += 1
            _emit(IndexProgress(phase=IndexPhase.SCAN, done=stats["scanned"], total=-1, message=str(image_path)))

            try:
                stat = image_path.stat()
            except OSError as exc:
                logger.warning("Failed to stat %s: %s", image_path, exc)
                continue

            row = get_file_by_path(conn, str(image_path))
            is_new = row is None

            # size/mtime が一致していれば changed=False とみなして SHA 計算スキップ
            if row is not None:
                size_changed = int(row["size"] or 0) != stat.st_size
                mtime_changed = float(row["mtime"] or 0.0) != stat.st_mtime
            else:
                size_changed = True
                mtime_changed = True

            if is_new or size_changed or mtime_changed:
                try:
                    sha = compute_sha256(image_path)
                except OSError as exc:
                    logger.warning("Failed to hash %s: %s", image_path, exc)
                    continue
                changed = True if is_new else (str(row["sha256"] or "") != sha)  # type: ignore[index]
            else:
                sha = str(row["sha256"] or "")  # type: ignore[index]
                changed = False

            indexed_at = None if changed else (row["indexed_at"] if row else None)
            file_id = upsert_file(
                conn,
                path=str(image_path),
                size=stat.st_size,
                mtime=stat.st_mtime,
                sha256=sha,
                indexed_at=indexed_at,
            )

            # 既存フラグの取得は軽量クエリで
            tag_exists = (
                conn.execute("SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1", (file_id,)).fetchone() is not None
            )
            embed_exists = (
                conn.execute(
                    "SELECT 1 FROM embeddings WHERE file_id = ? AND model = ? LIMIT 1",
                    (file_id, settings.embed_model.name),
                ).fetchone()
                is not None
            )

            stored_sig = str(row["tagger_sig"]) if (row is not None and row["tagger_sig"] is not None) else None
            stored_tagged_at = row["last_tagged_at"] if row is not None else None
            last_tagged_at = float(stored_tagged_at) if stored_tagged_at is not None else None
            needs_tagging = is_new or changed or (not tag_exists) or (stored_sig != tagger_sig)

            records.append(
                _FileRecord(
                    file_id=file_id,
                    path=image_path,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    sha=sha,
                    is_new=is_new,
                    changed=changed,
                    tag_exists=tag_exists,
                    embed_exists=embed_exists,
                    needs_tagging=needs_tagging,
                    needs_embedding=is_new or changed or not embed_exists,
                    stored_tagger_sig=stored_sig,
                    current_tagger_sig=tagger_sig,
                    last_tagged_at=last_tagged_at,
                )
            )

            if _should_cancel():
                cancelled = True
                break

        conn.commit()

        stats["new_or_changed"] = sum(1 for r in records if r.is_new or r.changed)
        logger.info("Scan complete: %d file(s) seen, %d new or changed", stats["scanned"], stats["new_or_changed"])
        if not cancelled:
            _emit(IndexProgress(phase=IndexPhase.SCAN, done=stats["scanned"], total=stats["scanned"]), force=True)

        # --- 画像ロードヘルパ ---
        def ensure_image_loaded(rec: _FileRecord) -> bool:
            if rec.load_failed:
                return False
            if rec.image is None:
                from utils.image_io import safe_load_image

                image = safe_load_image(
                    rec.path,
                    max_side=4096,
                    bomb_pixel_cap=350_000_000,
                    hard_skip_pixels=220_000_000,  # ← これが効く
                    skip_on_bomb=True,  # ← 爆弾判定はスキップ
                )
                if image is None:
                    rec.load_failed = True
                    return False
                rec.image = image
                rec.width = image.width
                rec.height = image.height
            return True

        def log_progress(stage: str, processed: int, total: int, last_value: int) -> int:
            if total <= 0:
                return last_value
            step = max(1, total // 5)
            if processed >= total or processed - last_value >= step:
                logger.info("%s progress: %d/%d", stage, processed, total)
                return processed
            return last_value

        # ===== TAGGING =====
        tag_records = [r for r in records if r.needs_tagging and not r.load_failed]
        _emit(IndexProgress(phase=IndexPhase.TAG, done=0, total=len(tag_records)), force=True)
        # FTS はタグ付け中は止めるので、ここでの FTS 進捗はダミーのまま
        _emit(IndexProgress(phase=IndexPhase.FTS, done=0, total=len(tag_records)), force=True)

        if tag_records and not cancelled:
            # ★ スキャナ用接続を閉じる → quiesce 開始 → DBWriter 起動
            try:
                conn.close()
            except Exception:
                pass
            finally:
                conn = None
            from db.connection import begin_quiesce, end_quiesce
            from db.connection import get_conn as _get_conn

            processed_tags = 0
            fts_processed = 0
            last_logged = 0
            quiesced = False
            merge_stage_label = {
                "merge.delete": "Deleting old tags",
                "merge.insert": "Writing new tags",
                "merge.update": "Updating files",
                "merge.index": "Rebuilding indexes",
            }

            def _dbw_progress(kind: str, done: int, total: int) -> None:
                # バーは FTS を流用（ラベルは Finalizing... に切替）
                nonlocal fts_processed
                fts_processed = done  # 単純に done を反映
                _emit(IndexProgress(phase=IndexPhase.FTS, done=done, total=total))
                # ステータスバーに文言を出す（任意。なければ省略）
                try:
                    label = merge_stage_label.get(kind.split(".")[0], "Finalizing...")
                    logger.info("finalizing: %s %d/%d", label, done, total)
                except Exception:
                    pass

            try:
                dbw = None
                begin_quiesce()
                quiesced = True
                tagger = _resolve_tagger(
                    settings,
                    tagger_override,
                    thresholds=thresholds or None,
                    max_tags=max_tags_map or None,
                )

                stats["tagger_name"] = type(tagger).__name__ if tagger_override is not None else settings.tagger.name
                logger.info("Tagging %d image(s)", len(tag_records))

                processed_tags = 0
                fts_processed = 0
                last_logged = 0
                idx = 0
                current_batch = batch_size

                # --- DB ライター起動 ---
                import os
                import time as _time

                from core.db_writer import DBItem, DBWriter

                # ★ ここから UNSAFE 区間へ（UI は接続を閉じておくこと）
                dbw = DBWriter(
                    db_path,
                    flush_chunk=getattr(settings, "db_flush_chunk", 1024),
                    fts_topk=getattr(settings, "fts_topk", 128),
                    queue_size=int(os.environ.get("KE_DB_QUEUE", "1024")),
                    default_tagger_sig=tagger_sig,
                    unsafe_fast=True,  # ← WAL を使わず MEMORY/EXCLUSIVE/OFF
                    skip_fts=True,  # ← FTS 更新は完全停止
                    progress_cb=_dbw_progress,
                )

                dbw.start()
                _time.sleep(0.2)
                dbw.raise_if_failed()

                # ===== PrefetchLoader を使った重畳ループ =====
                # 1) レコード ↔ パス対応
                rec_by_path: dict[str, _FileRecord] = {str(r.path): r for r in tag_records}
                tag_paths: list[str] = list(rec_by_path.keys())

                # 2) パラメータ
                current_batch = int(batch_size)
                prefetch_depth = int(os.environ.get("KE_PREFETCH_DEPTH", "12"))
                io_workers = int(os.environ.get("KE_IO_WORKERS", "16") or "16")
                if io_workers <= 0:
                    io_workers = None  # 自動

                # 3) PrefetchLoader 起動
                loader = PrefetchLoaderPrepared(
                    tag_paths,
                    batch_size=current_batch,
                    prefetch_batches=prefetch_depth,
                    io_workers=io_workers,
                    tagger=tagger,
                )

                def _infer_with_retry_np(rgb_list: list[np.ndarray]) -> list[TagResult]:
                    """
                    OOM 等で落ちたらバッチを分割して再試行（prepared 経路）。
                    rgb_list: list of np.ndarray (H,W,3, uint8, RGB)
                    """
                    try:
                        np_batch = tagger.prepare_batch_from_rgb_np(rgb_list)
                        return tagger.infer_batch_prepared(
                            np_batch,
                            thresholds=thresholds or None,
                            max_tags=max_tags_map or None,
                        )
                    except Exception:
                        if len(rgb_list) <= 1:
                            raise
                        mid = len(rgb_list) // 2
                        left = _infer_with_retry_np(rgb_list[:mid])
                        right = _infer_with_retry_np(rgb_list[mid:])
                        return left + right

                for batch in loader:
                    if cancelled or _should_cancel():
                        cancelled = True
                        break

                    if not batch:
                        continue
                    try:
                        batch_paths, batch_np_rgb, sizes = batch  # list[str], list[np.ndarray | None]
                    except Exception:
                        logger.error("PrefetchLoaderPrepared batch has unexpected shape: %r", type(batch))
                        raise

                    # 成功ロードだけ拾ってレコードに対応付け
                    batch_recs: list[_FileRecord] = []
                    rgb_list: list[np.ndarray] = []
                    wh_needed: list[tuple[int | None, int | None]] = []  # (w,h) or (None,None)
                    for p, arr, (_w, _h) in zip(batch_paths, batch_np_rgb, sizes):
                        if arr is None:
                            continue
                        rec = rec_by_path.get(p)
                        if rec is None:
                            continue
                        batch_recs.append(rec)
                        rgb_list.append(arr)
                        # 幅・高さが必要なら np.shape から拾う（PIL 不要）
                        need_wh = rec.is_new or rec.changed or rec.width is None or rec.height is None
                        if need_wh:
                            h, w = arr.shape[:2]
                            wh_needed.append((int(w), int(h)))
                        else:
                            wh_needed.append((None, None))

                    if not rgb_list:
                        continue

                    # --- 推論（prepared 経路） ---
                    try:
                        results = _infer_with_retry_np(rgb_list)
                    except Exception as exc:
                        logger.exception("Tagger failed for a prefetch-prepared batch: %s", exc)
                        continue

                    # --- 推論結果 → DBWriter ---
                    per_file_items: list[DBItem] = []
                    now_ts = time.time()
                    for rec, result, (w_opt, h_opt) in zip(batch_recs, results, wh_needed):
                        merged: dict[str, tuple[float, TagCategory]] = {}
                        for pred in result.tags:
                            name = pred.name.strip()
                            if not name:
                                continue
                            s = float(pred.score)
                            cur = merged.get(name)
                            if (cur is None) or (s > cur[0]):
                                merged[name] = (s, pred.category)

                        items = [(n, s, int(c)) for n, (s, c) in merged.items()]
                        per_file_items.append(
                            DBItem(rec.file_id, items, w_opt, h_opt, tagger_sig=tagger_sig, tagged_at=now_ts)
                        )

                        # UI統計更新
                        prev_sig = rec.stored_tagger_sig
                        rec.needs_tagging = False
                        rec.tag_exists = True
                        rec.stored_tagger_sig = tagger_sig
                        rec.current_tagger_sig = tagger_sig
                        rec.last_tagged_at = now_ts
                        if (not rec.is_new) and (not rec.changed) and (prev_sig != tagger_sig):
                            stats["retagged"] = int(stats.get("retagged", 0)) + 1

                    # enqueue（DBWriter がまとめる）
                    for it in per_file_items:
                        dbw.put(it)

                    # 進捗（推論完了ベース）
                    done_now = len(per_file_items)
                    processed_tags += done_now
                    fts_processed += done_now
                    last_logged = log_progress("Tagging", processed_tags, len(tag_records), last_logged)
                    _emit(IndexProgress(phase=IndexPhase.TAG, done=processed_tags, total=len(tag_records)))
                    _emit(IndexProgress(phase=IndexPhase.FTS, done=fts_processed, total=len(tag_records)))

                    # メモリ開放（PIL）
                    for _im in rgb_list:
                        try:
                            _im.close()
                        except Exception:
                            pass
                    rgb_list.clear()
                    per_file_items.clear()

                    # 断片化対策でたまにGC
                    if (processed_tags % 4096) == 0:
                        import gc

                        gc.collect()

            finally:
                print("finalizing tagging phase...")
                # Prefetch を必ず停止
                try:
                    loader.close()
                except Exception:
                    pass

                if dbw is not None:
                    try:
                        dbw.stop(flush=True, wait_forever=True)
                    finally:
                        dbw = None
                # ★ 必ず quiesce を解除（例外やキャンセルでも）
                if quiesced:
                    try:
                        end_quiesce()
                    except Exception:
                        logger.exception("end_quiesce failed")
                    quiesced = False

                # ★ ここで“整地”を実行
                try:
                    _settle_after_quiesce(db_path)
                except Exception:
                    logger.warning("settle_after_quiesce failed; continuing")

            # （以降は元の後処理と同じ）
            stats["tagged"] = processed_tags
            logger.info("Tagging complete: %d image(s) processed", processed_tags)
            _emit(IndexProgress(phase=IndexPhase.TAG, done=processed_tags, total=len(tag_records)), force=True)
            _emit(IndexProgress(phase=IndexPhase.FTS, done=fts_processed, total=len(tag_records)), force=True)
            print("... tagging phase finalized")

        # ★ ここで安全接続を再オープン（以降の EMBED/HNSW/FTS 再構築で使用）
        if conn is None:
            conn = get_conn(db_path)

        # ===== EMBEDDING =====（ここは従来どおりでOKだが、必要なら同様に一括 executemany 化できる）
        embed_records = [r for r in records if r.needs_embedding and not r.load_failed]
        _emit(IndexProgress(phase=IndexPhase.EMBED, done=0, total=len(embed_records)), force=True)

        if embed_records and not cancelled:
            try:
                embedder = embedder_override or _resolve_embedder(settings)
            except Exception as exc:
                logger.exception("Failed to instantiate embedder: %s", exc)
            else:
                logger.info("Embedding %d image(s)", len(embed_records))
                processed_embeddings = 0
                last_logged = 0
                idx = 0
                current_batch = batch_size

                while idx < len(embed_records):
                    if cancelled or _should_cancel():
                        cancelled = True
                        break
                    batch_slice = embed_records[idx : idx + current_batch]

                    images: list[Image.Image] = []
                    valid: list[_FileRecord] = []
                    for rec in batch_slice:
                        if ensure_image_loaded(rec) and rec.image is not None:
                            images.append(rec.image)
                            valid.append(rec)
                    if not images:
                        idx += len(batch_slice)
                        continue

                    try:
                        vectors = embedder.embed_images(images)
                    except Exception as exc:
                        if current_batch > 1:
                            current_batch = max(1, current_batch // 2)
                            logger.warning("Embedding batch failed (%s); reducing batch size to %d", exc, current_batch)
                            continue
                        logger.exception("Embedding failed for batch starting with %s", valid[0].path)
                        idx += len(batch_slice)
                        continue

                    if len(vectors) != len(valid):
                        logger.error(
                            "Embedder returned %d vectors for %d images; skipping batch", len(vectors), len(valid)
                        )
                        idx += len(batch_slice)
                        continue

                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        up_rows_embed: list[tuple[int, str, int, memoryview]] = []
                        up_rows_file: list[tuple[int, int, float, str, int, int]] = []
                        sig_rows: list[tuple[int, int, int]] = []

                        for rec, image, vector in zip(valid, images, vectors):
                            # 使う属性は先に取り出してからクローズできるようにする  # ← 追加
                            _w, _h = getattr(image, "width", None), getattr(image, "height", None)  # ← 追加

                            array = np.asarray(vector, dtype=np.float32)
                            if array.ndim != 1:
                                array = np.reshape(array, (-1,))
                            if not array.flags.c_contiguous:
                                array = np.ascontiguousarray(array, dtype=np.float32)
                            else:
                                array = array.astype(np.float32, copy=True)
                            norm = float(np.linalg.norm(array))
                            if norm > 0:
                                array /= norm

                            # files の基本属性更新（念のため）
                            up_rows_file.append(  # ← 変更: image.width/height をローカルにした値で
                                (rec.size, rec.mtime, rec.sha, str(rec.path), _w or 0, _h or 0)
                            )
                            # signatures
                            sig_rows.append((rec.file_id, int(phash(image)), int(dhash(image))))
                            # embeddings
                            up_rows_embed.append(
                                (
                                    rec.file_id,
                                    settings.embed_model.name,
                                    int(array.shape[0]),
                                    memoryview(array.tobytes()),
                                )
                            )

                            was_missing = not rec.embed_exists
                            mark_indexed_at(conn, rec.file_id, indexed_at=time.time())
                            rec.needs_embedding = False
                            rec.embed_exists = True
                            if rec.is_new or was_missing:
                                hnsw_additions.append((rec.file_id, array.copy()))
                            processed_embeddings += 1

                            # ここで画像を即解放し、参照も切る  # ← 追加
                            try:
                                if image is not None:
                                    image.close()
                            except Exception:
                                pass
                            rec.image = None
                            del array  # ベクトル一時配列も解放しやすく  # ← 追加
                        # bulk upserts
                        if up_rows_file:
                            conn.executemany(
                                "UPDATE files SET size=?, mtime=?, sha256=? , width=?, height=? WHERE path=?",
                                # ↑順番間違い注意: 下のタプルと合わせる
                                [(sz, mt, sha, w, h, p) for (sz, mt, sha, p, w, h) in up_rows_file],
                            )
                        if sig_rows:
                            conn.executemany(
                                "INSERT INTO signatures (file_id, phash_u64, dhash_u64) VALUES (?, ?, ?) "
                                "ON CONFLICT(file_id) DO UPDATE SET phash_u64=excluded.phash_u64, dhash_u64=excluded.dhash_u64",
                                sig_rows,
                            )
                        if up_rows_embed:
                            conn.executemany(
                                "INSERT INTO embeddings (file_id, model, dim, vector) VALUES (?, ?, ?, ?) "
                                "ON CONFLICT(file_id, model) DO UPDATE SET dim=excluded.dim, vector=excluded.vector",
                                up_rows_embed,
                            )
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise

                    # バッチ終了時にリスト参照を切って早めに回収  # ← 追加
                    try:
                        images.clear()
                    except Exception:
                        pass
                    del images
                    try:
                        vectors.clear()
                    except Exception:
                        pass
                    del vectors

                    # 数千件ごとに軽くGCして断片化を抑える  # ← 追加
                    if (processed_embeddings % 4096) == 0:
                        import gc

                        gc.collect()
                    last_logged = log_progress("Embedding", processed_embeddings, len(embed_records), last_logged)
                    _emit(IndexProgress(phase=IndexPhase.EMBED, done=processed_embeddings, total=len(embed_records)))

                    idx += len(batch_slice)

                conn.commit()
                stats["signatures"] = processed_embeddings
                stats["embedded"] = processed_embeddings
                logger.info("Embedding complete: %d image(s) processed", processed_embeddings)
                _emit(
                    IndexProgress(phase=IndexPhase.EMBED, done=processed_embeddings, total=len(embed_records)),
                    force=True,
                )

        # メモリ解放
        for rec in records:
            rec.image = None

    finally:
        try:
            if conn is not None:
                conn.close()
            if dbw is not None:
                dbw.stop(flush=not cancelled, timeout=30.0)
        except Exception:
            logger.exception("dbwriter stop failed")
        finally:
            dbw = None
            conn = None
        # ★ quiesce を解除し、安全接続に戻す（以降の埋め込み/HNSW/FTSは通常モード）
        # ★ オフライン FTS 再構築（UNSAFE 区間内でやってしまう）
        try:
            if quiesced and not cancelled:
                logger.info("rebuild_fts_offline: start")
                rebuilt = rebuild_fts_offline(db_path, topk=int(getattr(settings, "fts_topk", 128) or 128))
                logger.info("rebuild_fts_offline: done (%d rows)", rebuilt)
        finally:
            if quiesced:
                end_quiesce()
                conn = _get_conn(db_path)  # 安全(WAL)接続に戻す
                quiesced = False

    # ===== HNSW 追記 =====
    _emit(IndexProgress(phase=IndexPhase.HNSW, done=0, total=len(hnsw_additions)), force=True)
    if hnsw_additions and not cancelled:
        dim = hnsw_additions[0][1].shape[0]
        ensure_dirs()
        index_dir = Path(settings.index_dir).expanduser() if settings.index_dir else get_index_dir()
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "hnsw_cosine.bin"
        try:
            index = load_hnsw_index(index_path, dim=dim)
            added = add_embeddings_to_hnsw(index, hnsw_additions, dim=dim)
            if added:
                save_hnsw_index(index, index_path)
            stats["hnsw_added"] = added
            logger.info("HNSW index updated with %d new vector(s)", added)
            _emit(IndexProgress(phase=IndexPhase.HNSW, done=added, total=len(hnsw_additions)), force=True)
        except Exception as exc:
            logger.exception("Failed to update HNSW index: %s", exc)

    stats["elapsed_sec"] = time.perf_counter() - start_time
    stats["cancelled"] = cancelled
    _emit(IndexProgress(phase=IndexPhase.DONE, done=1, total=1, message="cancelled" if cancelled else None), force=True)
    logger.info(
        "Indexing complete: scanned=%d, new=%d, tagged=%d, embedded=%d, hnsw_added=%d (%.2fs)",
        stats["scanned"],
        stats["new_or_changed"],
        stats["tagged"],
        stats["embedded"],
        stats["hnsw_added"],
        stats["elapsed_sec"],
    )
    return stats


def current_tagger_sig(
    settings: PipelineSettings,
    *,
    thresholds: Mapping[TagCategory, float] | None = None,
    max_tags: Mapping[TagCategory, int] | None = None,
) -> str:
    """Derive a stable signature that captures the active tagger configuration."""

    threshold_map = thresholds or _build_threshold_map(settings.tagger.thresholds)
    max_tags_map = max_tags or _build_max_tags_map(getattr(settings.tagger, "max_tags", None))
    serialised_thresholds = _serialise_thresholds(threshold_map)
    serialised_max_tags = _serialise_max_tags(max_tags_map)

    tagger_name = str(getattr(settings.tagger, "name", "") or "").lower()
    model_path = getattr(settings.tagger, "model_path", None)
    tags_csv = getattr(settings.tagger, "tags_csv", None)

    model_digest = _digest_identifier(model_path)
    csv_digest = _digest_identifier(tags_csv)
    thresholds_part = _format_sig_mapping(serialised_thresholds)
    max_tags_part = _format_sig_mapping(serialised_max_tags)
    return f"{tagger_name}:{model_digest}:csv={csv_digest}:" f"thr={thresholds_part}:max={max_tags_part}"


def retag_query(
    db_path: str | Path,
    where_sql: str,
    params: Sequence[object] | None,
) -> int:
    """Mark files matching the provided WHERE clause for re-tagging."""

    predicate = where_sql.strip() or "1=1"
    arguments: tuple[object, ...] = tuple(params or ())
    conn = get_conn(db_path, allow_when_quiesced=True)
    try:
        # predicate 内で f.* を参照できるように、外側の UPDATE にもエイリアスを付ける
        sql = "UPDATE files AS f " "SET tagger_sig = NULL, last_tagged_at = NULL " f"WHERE {predicate}"
        conn.execute(sql, arguments)
        affected = conn.execute("SELECT changes()").fetchone()[0] or 0
        conn.commit()
        logger.info("Flagged %d file(s) for re-tagging (predicate=%s)", affected, predicate)
        return int(affected)
    finally:
        conn.close()


def retag_all(
    db_path: str | Path,
    *,
    force: bool = False,
    settings: PipelineSettings | None = None,
) -> int:
    """Reset tagger signatures across the library to trigger re-tagging."""

    effective_settings = settings or load_settings()
    signature = current_tagger_sig(effective_settings)
    if force:
        predicate = "1=1"
        params: tuple[object, ...] = ()
    else:
        predicate = "tagger_sig = ?"
        params = (signature,)
    affected = retag_query(db_path, predicate, params)
    logger.info(
        "Scheduled %d file(s) for re-tagging (force=%s, signature=%s)",
        affected,
        force,
        signature,
    )
    return affected


def _resolve_tagger(
    settings: PipelineSettings,
    override: ITagger | None,
    *,
    thresholds: Mapping[TagCategory, float] | None = None,
    max_tags: Mapping[TagCategory, int] | None = None,
) -> ITagger:
    serialised_thresholds = _serialise_thresholds(thresholds)
    serialised_max_tags = _serialise_max_tags(max_tags)
    if override is not None:
        name = type(override).__name__
        model_path = getattr(override, "model_path", None)
        tags_csv = getattr(override, "tags_csv", None)
        logger.info(
            "Tagger in use: %s, model=%s, tags_csv=%s, thresholds=%s, max_tags=%s",
            name,
            model_path,
            tags_csv,
            serialised_thresholds,
            serialised_max_tags,
        )
        return override

    tagger_name = settings.tagger.name
    lowered = tagger_name.lower()
    model_path_value = settings.tagger.model_path
    tags_csv_value = getattr(settings.tagger, "tags_csv", None)
    if lowered == "dummy":
        from tagger.dummy import DummyTagger

        tagger_instance: ITagger = DummyTagger()
    elif lowered == "wd14-onnx":
        from tagger.wd14_onnx import WD14Tagger

        if not settings.tagger.model_path:
            raise ValueError("WD14: model_path is required")
        model_path_obj = Path(settings.tagger.model_path)
        tagger_instance = WD14Tagger(model_path_obj, tags_csv=settings.tagger.tags_csv)
        model_path_value = str(model_path_obj)
    else:
        raise ValueError(f"Unknown tagger '{settings.tagger.name}'")

    logger.info(
        "Tagger in use: %s, model=%s, tags_csv=%s, thresholds=%s, max_tags=%s",
        tagger_name,
        model_path_value,
        tags_csv_value,
        serialised_thresholds,
        serialised_max_tags,
    )
    return tagger_instance


def _resolve_embedder(settings: PipelineSettings) -> EmbedderProtocol:
    from sig.embedder import OpenClipEmbedder

    return OpenClipEmbedder(
        settings.embed_model.name,
        settings.embed_model.pretrained,
        device=settings.embed_model.device,
        batch_size=settings.batch_size,
    )


def _format_sig_mapping(mapping: Mapping[str, float | int]) -> str:
    if not mapping:
        return "none"
    parts: list[str] = []
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, float):
            formatted = format(value, ".6f").rstrip("0").rstrip(".")
        else:
            formatted = str(int(value))
        parts.append(f"{key}={formatted}")
    return ",".join(parts)


def _digest_identifier(value: str | Path | None) -> str:
    normalised = _normalise_sig_source(value)
    if normalised is None:
        return "none"
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _normalise_sig_source(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, Path):
        candidate = value
    else:
        try:
            candidate = Path(str(value))
        except (TypeError, ValueError):
            return str(value)
    expanded = candidate.expanduser()
    try:
        resolved = expanded.resolve(strict=False)
    except OSError:
        resolved = expanded.absolute()
    return str(resolved)


def _build_threshold_map(thresholds: dict[str, float]) -> dict[TagCategory, float]:
    mapping: dict[TagCategory, float] = {}
    for key, value in thresholds.items():
        category = _CATEGORY_KEY_LOOKUP.get(key.lower())
        if category is not None:
            mapping[category] = float(value)
    return mapping


def _build_max_tags_map(max_tags: Mapping[str, int] | None) -> dict[TagCategory, int]:
    mapping: dict[TagCategory, int] = {}
    if not max_tags:
        return mapping
    for key, value in max_tags.items():
        category = _CATEGORY_KEY_LOOKUP.get(str(key).lower())
        if category is None:
            continue
        try:
            mapping[category] = int(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _serialise_thresholds(
    thresholds: Mapping[TagCategory, float] | None,
) -> dict[str, float]:
    if not thresholds:
        return {}
    return {category.name.lower(): float(value) for category, value in thresholds.items()}


def _serialise_max_tags(
    max_tags: Mapping[TagCategory, int] | None,
) -> dict[str, int]:
    if not max_tags:
        return {}
    return {category.name.lower(): int(value) for category, value in max_tags.items()}


_CATEGORY_KEY_LOOKUP = {
    "general": TagCategory.GENERAL,
    "character": TagCategory.CHARACTER,
    "copyright": TagCategory.COPYRIGHT,
    "artist": TagCategory.ARTIST,
    "meta": TagCategory.META,
    "rating": TagCategory.RATING,
}


__all__ = [
    "IndexPhase",
    "IndexProgress",
    "PipelineSettings",
    "ProcessingPipeline",
    "current_tagger_sig",
    "scan_and_tag",
    "retag_all",
    "retag_query",
    "run_index_once",
]
