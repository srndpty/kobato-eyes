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

import cv2
import numpy as np
from PIL import Image

from core.db_writer import DBItem, DBWriter
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
from db.connection import bootstrap_if_needed, get_conn
from db.repository import fts_delete_rows, get_file_by_path, list_untagged_under_path, upsert_file
from tagger.base import ITagger, TagCategory, TagResult
from utils.hash import compute_sha256
from utils.paths import ensure_dirs, get_db_path

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


# --- TurboJPEG はあれば使う（無ければ自動でオフ） ---
_USE_TJ = os.getenv("KE_USE_TURBOJPEG", "1") != "0"
_TJ = None
if _USE_TJ:
    try:
        from turbojpeg import TJPF, TurboJPEG  # type: ignore

        _TJ = TurboJPEG()
    except Exception as e:
        logger.info("TurboJPEG not available (%s); falling back to OpenCV/PIL", e)
        _TJ = None

_TARGET = 448  # tagger の入力サイズ


def _choose_tj_scale(w: int, h: int) -> tuple[int, int]:
    # TurboJPEG の縮小段階は (1/1, 1/2, 1/4, 1/8, 1/16)。
    side = max(w, h)
    # ざっくり target に近づける（余裕を持って大きめ→後で1回だけ resize）
    if side >= _TARGET * 8:
        return (1, 8)
    if side >= _TARGET * 4:
        return (1, 4)
    if side >= _TARGET * 2:
        return (1, 2)
    return (1, 1)


def _alpha_to_white_bgr(bg_or_bgra: np.ndarray) -> np.ndarray:
    if bg_or_bgra.ndim == 2:  # gray
        return cv2.cvtColor(bg_or_bgra, cv2.COLOR_GRAY2BGR)
    if bg_or_bgra.shape[2] == 3:
        return bg_or_bgra
    if bg_or_bgra.shape[2] == 4:
        bgr = bg_or_bgra[:, :, :3].astype(np.float32)
        a = bg_or_bgra[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (bgr * a + 255.0 * (1.0 - a)).astype(np.uint8)
        return bgr
    return bg_or_bgra[:, :, :3]


class PrefetchLoaderPrepared:
    """
    画像を並列ロードし、BGR で軽量デコード→縮小→tagger で最終整形して、
    (paths, np_batch, sizes) を供給するローダ。
      - paths: List[str]（バッチ内のファイルパス。順序は元の順）
      - np_batch: np.ndarray 形状 (B, H, W, 3), float32  ※tagger.prepare_batch_from_bgr() の結果
      - sizes: List[Tuple[int,int]] 元画像の (width, height)
    """

    def __init__(
        self,
        paths: List[str],
        *,
        tagger,  # WD14Tagger インスタンス（prepare_batch_from_bgr を呼ぶ）
        batch_size: int,
        prefetch_batches: int = 2,
        io_workers: int | None = None,
    ) -> None:
        self._paths = list(paths)
        self._B = int(batch_size)
        self._depth = max(1, int(prefetch_batches))
        cpu = os.cpu_count() or 4
        # 既定はやや強め（PNG 多めのとき効く）: 明示指定があればそちらを優先、env でも上書き可
        env_workers = os.getenv("KE_IO_WORKERS")
        if env_workers is not None:
            io_workers = int(env_workers)
        if io_workers is None:
            io_workers = min(max(4, cpu), 16)
        self._io_workers: int = max(1, int(io_workers))
        self._tagger = tagger

        # (paths, np_batch, sizes) or None(sentinal)
        self._q: "queue.Queue[tuple[list[str], np.ndarray, list[tuple[int,int]]] | None]" = queue.Queue(self._depth)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._producer, name="PL-Feeder", daemon=True)
        self._th.start()
        logger.info(
            "PrefetchLoaderPrepared: start (B=%d, depth=%d, io_workers=%d)", self._B, self._depth, self._io_workers
        )

    # --- 1 枚ロード（TurboJPEG/OpenCV 優先、失敗したら PIL） ---
    def _load_one(self, p: str) -> tuple[str, np.ndarray | None, tuple[int, int] | None]:
        ext = Path(p).suffix.lower()
        try:
            # --- JPEG: TurboJPEG 縮小デコード ---
            if ext in (".jpg", ".jpeg") and _TJ is not None:
                with open(p, "rb") as f:
                    buf = f.read()
                # ヘッダから元サイズ取得
                try:
                    w, h, _, _ = _TJ.decode_header(buf)
                except Exception:
                    # ヘッダ取れない超古い JPEG 等は一旦フルで読んで形状から決める
                    tmp = _TJ.decode(buf, pixel_format=TJPF.BGR)
                    h, w = tmp.shape[:2]
                scale = _choose_tj_scale(w, h)
                bgr = _TJ.decode(buf, pixel_format=TJPF.BGR, scaling_factor=scale)
                # ここで軽く TARGET 付近へ
                hh, ww = bgr.shape[:2]
                side = max(hh, ww)
                if side != _TARGET:
                    ratio = _TARGET / side
                    interp = cv2.INTER_AREA if side > _TARGET else cv2.INTER_CUBIC
                    bgr = cv2.resize(bgr, (max(1, int(ww * ratio)), max(1, int(hh * ratio))), interpolation=interp)
                return (p, bgr, (w, h))

            # --- PNG / WebP / その他: OpenCV ---
            data = np.fromfile(p, dtype=np.uint8)  # Windows で速い
            im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError("cv2.imdecode failed")
            bgr = _alpha_to_white_bgr(im)
            hh, ww = bgr.shape[:2]
            side = max(hh, ww)
            if side != _TARGET:
                ratio = _TARGET / side
                interp = cv2.INTER_AREA if side > _TARGET else cv2.INTER_CUBIC
                bgr = cv2.resize(bgr, (max(1, int(ww * ratio)), max(1, int(hh * ratio))), interpolation=interp)
            # 元サイズは im の生サイズ（アルファ有無に関係なし）
            H0, W0 = (im.shape[0], im.shape[1]) if im.ndim >= 2 else (hh, ww)
            return (p, bgr, (int(W0), int(H0)))

        except Exception as e:
            logger.warning("PrefetchLoaderPrepared: failed to load %s: %s (fallback PIL)", p, e)
            # --- フォールバック: PIL ---
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, "WHITE")
                    bg.paste(im, mask=im.split()[-1])
                    rgb = bg.convert("RGB")
                    # ここでは軽く縮小のみ（最終整形は tagger に任せる）
                    rgb.thumbnail((_TARGET, _TARGET))
                    bgr = np.asarray(rgb)[:, :, ::-1]  # RGB->BGR
                    return (p, bgr, (w, h))
            except Exception as e2:
                logger.warning("PrefetchLoaderPrepared: PIL fallback also failed %s: %s", p, e2)
                return (p, None, None)

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except Exception:
            return -1

    def _producer(self) -> None:
        try:
            N = len(self._paths)
            with ThreadPoolExecutor(max_workers=self._io_workers, thread_name_prefix="ke-io") as ex:
                for i in range(0, N, self._B):
                    if self._stop.is_set():
                        break

                    batch_paths = self._paths[i : i + self._B]
                    futs = [ex.submit(self._load_one, p) for p in batch_paths]

                    tmp: dict[str, tuple[np.ndarray | None, tuple[int, int] | None]] = {}
                    for fut in as_completed(futs):
                        p, arr, sz = fut.result()
                        tmp[p] = (arr, sz)

                    # 順序維持で集約
                    bgr_list: list[np.ndarray] = []
                    sizes: list[tuple[int, int]] = []
                    kept_paths: list[str] = []
                    for p in batch_paths:
                        arr, sz = tmp.get(p, (None, None))
                        if arr is None or sz is None:
                            continue
                        bgr_list.append(arr)
                        sizes.append(sz)
                        kept_paths.append(p)

                    if not bgr_list:
                        continue

                    # ここで最終整形（正方形 + ぴったり TARGET + float32）を tagger に任せる
                    np_batch = self._tagger.prepare_batch_from_bgr(bgr_list)

                    # キューへ（必要なら put 時間をログ）
                    t0 = time.perf_counter()
                    q_before = self._q.qsize()
                    self._q.put((kept_paths, np_batch, sizes))
                    wait_put_ms = (time.perf_counter() - t0) * 1000.0
                    if wait_put_ms > 1.0 or q_before == self._depth - 1:
                        logger.info("LOAD put wait=%.1fms q=%d/%d", wait_put_ms, q_before, self._depth)

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
        tag_config: TagJobConfig,
    ) -> None:
        super().__init__(list(paths))
        self._db_path = db_path
        self._tagger = tagger
        self._tag_config = tag_config
        self._conn = None

    def prepare(self) -> None:
        self._conn = get_conn(self._db_path)
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
        self._job_manager = job_manager
        self._settings = settings or PipelineSettings()
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
        self._settings = settings
        self._refresh_tag_config()

    def stop(self) -> None:
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
    hard_delete_missing: bool = False,
) -> dict[str, object]:
    """Tag unprocessed images within ``root``."""

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
    needs_tagging: bool
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


# ============================================
# 新規: 共通コンテキスト / 進捗エミッタ
# ============================================


@dataclass
class PipelineContext:
    db_path: str | Path
    settings: PipelineSettings
    thresholds: dict[TagCategory, float]
    max_tags_map: dict[TagCategory, int]
    tagger_sig: str
    progress_cb: Callable[[IndexProgress], None] | None = None
    is_cancelled: Callable[[], bool] | None = None


class ProgressEmitter:
    def __init__(self, cb: Callable[[IndexProgress], None] | None):
        self._cb = cb
        self._last: dict[IndexPhase, tuple[int, float]] = {}

    def emit(self, progress: IndexProgress, force: bool = False) -> None:
        if self._cb is None:
            return
        import time as _t

        now = _t.perf_counter()
        last = self._last.get(progress.phase)
        should = force or last is None
        if not should and last is not None:
            last_done, last_time = last
            if progress.total > 0:
                step = max(1, progress.total // 100)
                should = (progress.done >= progress.total) or ((progress.done - last_done) >= step)
            should = should or ((now - last_time) >= 0.1)
        if should:
            self._last[progress.phase] = (progress.done, now)
            try:
                self._cb(progress)
            except Exception:
                # Qt 側破棄などで死んだら停止
                logger.exception("Progress callback raised; disabling further updates.")
                self._cb = None

    def cancelled(self, fn: Callable[[], bool] | None) -> bool:
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:
            logger.exception("Cancellation callback failed")
            return False


# ============================================
# 新規: スキャン段階
# ============================================
class Scanner:
    def __init__(self, ctx: PipelineContext, emitter: ProgressEmitter):
        self.ctx = ctx
        self.emitter = emitter

    def scan(self) -> tuple[list[_FileRecord], dict[str, int]]:
        settings = self.ctx.settings
        roots = [Path(root).expanduser() for root in settings.roots if root]
        roots = [r for r in roots if r.exists()]
        excluded_paths = [Path(p).expanduser() for p in settings.excluded if p]
        allow_exts = {ext.lower() for ext in (settings.allow_exts or DEFAULT_EXTENSIONS)}

        stats = {"scanned": 0, "new_or_changed": 0}
        if not roots:
            logger.info("No valid roots configured; skipping scan.")
            return [], stats

        db_literal = str(self.ctx.db_path)
        conn = get_conn(
            db_literal
            if (db_literal.startswith("file:") or db_literal == ":memory:")
            else Path(self.ctx.db_path).expanduser(),
            allow_when_quiesced=True,
        )

        records: list[_FileRecord] = []
        try:
            logger.info("Scanning %d root(s) for eligible images", len(roots))
            for image_path in iter_images(roots, excluded=excluded_paths, extensions=allow_exts):
                if self.emitter.cancelled(self.ctx.is_cancelled):
                    break
                stats["scanned"] += 1
                self.emitter.emit(
                    IndexProgress(phase=IndexPhase.SCAN, done=stats["scanned"], total=-1, message=str(image_path))
                )

                try:
                    stat = image_path.stat()
                except OSError as exc:
                    logger.warning("Failed to stat %s: %s", image_path, exc)
                    continue

                row = get_file_by_path(conn, str(image_path))
                is_new = row is None
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

                tag_exists = (
                    conn.execute("SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1", (file_id,)).fetchone() is not None
                )
                stored_sig = str(row["tagger_sig"]) if (row is not None and row["tagger_sig"] is not None) else None
                stored_tagged_at = row["last_tagged_at"] if row is not None else None
                last_tagged_at = float(stored_tagged_at) if stored_tagged_at is not None else None
                needs_tagging = is_new or changed or (not tag_exists) or (stored_sig != self.ctx.tagger_sig)

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
                        needs_tagging=needs_tagging,
                        stored_tagger_sig=stored_sig,
                        current_tagger_sig=self.ctx.tagger_sig,
                        last_tagged_at=last_tagged_at,
                    )
                )
            conn.commit()

            stats["new_or_changed"] = sum(1 for r in records if r.is_new or r.changed)
            logger.info("Scan complete: %d file(s) seen, %d new or changed", stats["scanned"], stats["new_or_changed"])
            self.emitter.emit(
                IndexProgress(phase=IndexPhase.SCAN, done=stats["scanned"], total=stats["scanned"]), force=True
            )
            return records, stats
        finally:
            conn.close()


# ============================================
# 新規: タグ付け段階（quiesce + DBWriter + PrefetchPrepared）
# ============================================
class TaggingStage:
    def __init__(self, ctx: PipelineContext, emitter: ProgressEmitter):
        self.ctx = ctx
        self.emitter = emitter

    def _log_step(self, stage: str, done: int, total: int, last: int) -> int:
        if total <= 0:
            return last
        step = max(1, total // 5)
        if done >= total or done - last >= step:
            logger.info("%s progress: %d/%d", stage, done, total)
            return done
        return last

    def run(self, records: list[_FileRecord]) -> tuple[int, int, list[tuple[int, np.ndarray]]]:
        thresholds = self.ctx.thresholds
        max_tags_map = self.ctx.max_tags_map
        tagger_sig = self.ctx.tagger_sig
        settings = self.ctx.settings
        db_path = self.ctx.db_path

        tag_records = [r for r in records if r.needs_tagging and not r.load_failed]
        self.emitter.emit(IndexProgress(phase=IndexPhase.TAG, done=0, total=len(tag_records)), force=True)
        self.emitter.emit(IndexProgress(phase=IndexPhase.FTS, done=0, total=len(tag_records)), force=True)
        if not tag_records:
            return 0, 0, []

        # quiesce & DBWriter
        from db.connection import begin_quiesce, end_quiesce

        processed_tags = 0
        fts_processed = 0
        last_logged = 0
        quiesced = False
        dbw: DBWriter | None = None

        def _dbw_progress(kind: str, done: int, total: int) -> None:
            nonlocal fts_processed
            fts_processed = done
            self.emitter.emit(IndexProgress(phase=IndexPhase.FTS, done=done, total=total))
            try:
                logger.info("finalizing: %s %d/%d", kind.split(".")[0], done, total)
            except Exception:
                pass

        try:
            conn = get_conn(db_path, allow_when_quiesced=True)  # スキャナ用接続をすぐ閉じる
            try:
                conn.close()
            except Exception:
                pass
            begin_quiesce()
            quiesced = True

            tagger = _resolve_tagger(settings, None, thresholds=thresholds or None, max_tags=max_tags_map or None)
            logger.info("Tagging %d image(s)", len(tag_records))

            dbw = DBWriter(
                db_path,
                flush_chunk=getattr(settings, "db_flush_chunk", 1024),
                fts_topk=getattr(settings, "fts_topk", 128),
                queue_size=int(os.environ.get("KE_DB_QUEUE", "1024")),
                default_tagger_sig=tagger_sig,
                unsafe_fast=True,
                skip_fts=True,
                progress_cb=_dbw_progress,
            )
            dbw.start()
            time.sleep(0.2)
            dbw.raise_if_failed()

            # PrefetchPrepared + prepared 経路
            rec_by_path: dict[str, _FileRecord] = {str(r.path): r for r in tag_records}
            tag_paths: list[str] = list(rec_by_path.keys())
            tag_paths.sort(key=lambda p: (Path(p).parent, os.path.getsize(p)))
            current_batch = max(1, int(getattr(settings, "batch_size", 32) or 32))
            prefetch_depth = int(os.environ.get("KE_PREFETCH_DEPTH", "128") or "128") or 128
            io_workers = int(os.environ.get("KE_IO_WORKERS", "12") or "12") or None

            loader = PrefetchLoaderPrepared(
                tag_paths,
                batch_size=current_batch,
                prefetch_batches=prefetch_depth,
                io_workers=io_workers,
                tagger=tagger,
            )

            def _infer_with_retry_np(rgb_list: list[np.ndarray]) -> list[TagResult]:
                try:
                    np_batch = tagger.prepare_batch_from_rgb_np(rgb_list)
                    return tagger.infer_batch_prepared(
                        np_batch, thresholds=thresholds or None, max_tags=max_tags_map or None
                    )
                except Exception:
                    if len(rgb_list) <= 1:
                        raise
                    mid = len(rgb_list) // 2
                    return _infer_with_retry_np(rgb_list[:mid]) + _infer_with_retry_np(rgb_list[mid:])

            try:
                loader_iter = iter(loader)
                while True:
                    t_wait0 = time.perf_counter()
                    try:
                        batch = next(loader_iter)  # ← ここでローダが供給できるまでブロック
                    except StopIteration:
                        break

                    wait_batch_ms = (time.perf_counter() - t_wait0) * 1000.0
                    if self.emitter.cancelled(self.ctx.is_cancelled):
                        break
                    if not batch:
                        # 何も来なかったがループは前進。待ち時間だけログる
                        logger.info("PIPE wait_batch=%.2fms (empty batch)", wait_batch_ms)
                        continue
                    batch_paths, batch_np_rgb, sizes = batch

                    if wait_batch_ms > 1000:
                        from collections import Counter

                        exts = [os.path.splitext(p)[1].lower() for p in batch_paths]
                        ext_cnt = Counter(exts)
                        # 総画素数（参考指標）: そのバッチの decode 負荷の目安
                        total_px = sum((w * h) for (_, _), (w, h) in zip(sizes, sizes))  # sizes が (w,h) ならそのまま
                        logger.warning(
                            "PIPE slow_batch: wait=%.0fms, ext=%s, total_px=%.1f MP",
                            wait_batch_ms,
                            dict(ext_cnt),
                            total_px / 1e6,
                        )

                    batch_recs: list[_FileRecord] = []
                    rgb_list: list[np.ndarray] = []
                    wh_needed: list[tuple[int | None, int | None]] = []
                    for p, arr, (_w, _h) in zip(batch_paths, batch_np_rgb, sizes):
                        if arr is None:
                            continue
                        rec = rec_by_path.get(p)
                        if rec is None:
                            continue
                        batch_recs.append(rec)
                        rgb_list.append(arr)
                        need_wh = rec.is_new or rec.changed or rec.width is None or rec.height is None
                        if need_wh:
                            h, w = arr.shape[:2]
                            wh_needed.append((int(w), int(h)))
                        else:
                            wh_needed.append((None, None))
                    if not rgb_list:
                        continue

                    qsz = getattr(loader, "qsize", lambda: -1)()
                    logger.info("PIPE batch=%d wait_batch=%.2fms loader_qsize=%d", len(rgb_list), wait_batch_ms, qsz)

                    try:
                        results = _infer_with_retry_np(rgb_list)
                    except Exception:
                        logger.exception("Tagger failed for a prepared batch")
                        continue

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

                        prev_sig = rec.stored_tagger_sig
                        rec.needs_tagging = False
                        rec.tag_exists = True
                        rec.stored_tagger_sig = tagger_sig
                        rec.current_tagger_sig = tagger_sig
                        rec.last_tagged_at = now_ts
                        if (not rec.is_new) and (not rec.changed) and (prev_sig != tagger_sig):
                            # retagged は呼び出し側で集計したい場合は戻り値に含めても良い
                            pass

                    # --- DBWriter へ投入（満杯だとここでブロック）---
                    q_before = None
                    try:
                        q_before = dbw.qsize()  # あれば
                    except Exception:
                        pass
                    t_enqueue0 = time.perf_counter()
                    for db_item in per_file_items:
                        dbw.put(db_item)  # blocking put
                    enqueue_ms = (time.perf_counter() - t_enqueue0) * 1000.0
                    q_after = None
                    try:
                        q_after = dbw.qsize()
                    except Exception:
                        pass

                    logger.info(
                        "PIPE batch=%d wait_batch=%.2fms enqueue=%.2fms qsize=%s->%s",
                        len(per_file_items),
                        wait_batch_ms,
                        enqueue_ms,
                        q_before,
                        q_after,
                    )
                    processed_tags += len(per_file_items)
                    fts_processed += len(per_file_items)
                    last_logged = self._log_step("Tagging", processed_tags, len(tag_records), last_logged)
                    self.emitter.emit(IndexProgress(phase=IndexPhase.TAG, done=processed_tags, total=len(tag_records)))
                    self.emitter.emit(IndexProgress(phase=IndexPhase.FTS, done=fts_processed, total=len(tag_records)))

            finally:
                try:
                    loader.close()
                except Exception:
                    pass
                if dbw is not None:
                    try:
                        dbw.stop(flush=True, wait_forever=True)
                    finally:
                        dbw = None
                if quiesced:
                    try:
                        end_quiesce()
                    except Exception:
                        logger.exception("end_quiesce failed")
                    quiesced = False
                try:
                    _settle_after_quiesce(str(db_path))
                except Exception:
                    logger.warning("settle_after_quiesce failed; continuing")

            logger.info("Tagging complete: %d image(s) processed", processed_tags)
            self.emitter.emit(
                IndexProgress(phase=IndexPhase.TAG, done=processed_tags, total=len(tag_records)), force=True
            )
            self.emitter.emit(
                IndexProgress(phase=IndexPhase.FTS, done=fts_processed, total=len(tag_records)), force=True
            )

        except Exception as exc:
            logger.exception("Tagging stage failed: %s", exc)

        # Tagging では HNSW の追加対象は作らない（Embeddingで作る）
        return processed_tags, fts_processed, []


# ============================================
# 新規: 全体オーケストレータ
# ============================================
class IndexPipeline:
    def __init__(
        self,
        db_path: str | Path,
        settings: PipelineSettings | None = None,
        *,
        tagger_override: ITagger | None = None,  # （今は未使用だが将来拡張用）
        progress_cb: Callable[[IndexProgress], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        settings = settings or load_settings()
        bootstrap_if_needed(db_path)
        ensure_dirs()

        thresholds = _build_threshold_map(settings.tagger.thresholds)
        max_tags_map = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))
        tagger_sig = current_tagger_sig(settings, thresholds=thresholds, max_tags=max_tags_map)

        self.ctx = PipelineContext(
            db_path=str(db_path),
            settings=settings,
            thresholds=thresholds,
            max_tags_map=max_tags_map,
            tagger_sig=tagger_sig,
            progress_cb=progress_cb,
            is_cancelled=is_cancelled,
        )
        self.emitter = ProgressEmitter(progress_cb)

    def run(self) -> dict[str, object]:
        start = time.perf_counter()
        stats: dict[str, object] = {
            "scanned": 0,
            "new_or_changed": 0,
            "tagged": 0,
            "signatures": 0,
            "elapsed_sec": 0.0,
            "tagger_name": self.ctx.settings.tagger.name,
            "retagged": 0,
            "cancelled": False,
            "tagger_sig": self.ctx.tagger_sig,
        }
        self.emitter.emit(IndexProgress(phase=IndexPhase.SCAN, done=0, total=-1, message="start"), force=True)

        # 1) Scan
        records, s = Scanner(self.ctx, self.emitter).scan()
        stats["scanned"] = s["scanned"]
        stats["new_or_changed"] = s["new_or_changed"]

        # 2) Tag
        if not self.emitter.cancelled(self.ctx.is_cancelled):
            tagged, _fts, _ = TaggingStage(self.ctx, self.emitter).run(records)
            stats["tagged"] = tagged

        stats["elapsed_sec"] = time.perf_counter() - start
        self.emitter.emit(IndexProgress(phase=IndexPhase.DONE, done=1, total=1), force=True)
        logger.info(
            "Indexing complete: scanned=%d, new=%d, tagged=%d, (%.2fs)",
            stats["scanned"],
            stats["new_or_changed"],
            stats["tagged"],
            stats["elapsed_sec"],
        )
        return stats


# ============================================
# 置換: 既存 run_index_once を新クラスに委譲
# ============================================
def run_index_once(
    db_path: str | Path,
    settings: PipelineSettings | None = None,
    *,
    tagger_override: ITagger | None = None,
    progress_cb: Callable[[IndexProgress], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    return IndexPipeline(
        db_path=db_path,
        settings=settings,
        tagger_override=tagger_override,
        progress_cb=progress_cb,
        is_cancelled=is_cancelled,
    ).run()


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
    elif lowered == "pixai":
        from tagger.pixai_torch import PixAITagger

        if not settings.tagger.model_path:
            raise ValueError("PixAI: model_dir is required")
        model_dir = Path(settings.tagger.model_path)
        tagger_instance = PixAITagger(model_dir, default_thresholds=settings.tagger.thresholds)
        model_path_value = str(model_dir)
        tags_csv_value = None
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
