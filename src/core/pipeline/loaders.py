"""Image prefetch loaders used by the tagging pipeline."""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Literal, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.env import safe_int

logger = logging.getLogger(__name__)

_CV2_ERROR = getattr(cv2, "error", RuntimeError)
if not isinstance(_CV2_ERROR, type) or not issubclass(_CV2_ERROR, BaseException):
    _CV2_ERROR = RuntimeError
_DECODE_FALLBACK_ERRORS: tuple[type[BaseException], ...] = (OSError, ValueError, RuntimeError, MemoryError, _CV2_ERROR)

# JPEG decode is intentionally OpenCV-only. TurboJPEG was benchmarked but
# removed to avoid external DLL packaging; TensorRT made disk/decode the next
# separate optimization target rather than a dependency to ship by default.

_TARGET = 448
_LOAD_ROUTE_KEYS = ("opencv", "pil_fallback", "failed")
LoadRoute = Literal["opencv", "pil_fallback", "failed"]


@dataclass(slots=True)
class LoaderMetrics:
    """Diagnostic timings collected by :class:`PrefetchLoaderPrepared`."""

    submitted: int = 0
    loaded: int = 0
    failed: int = 0
    batches: int = 0
    route_counts: dict[str, int] = field(default_factory=lambda: {key: 0 for key in _LOAD_ROUTE_KEYS})
    route_seconds: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in _LOAD_ROUTE_KEYS})
    prepare_seconds: float = 0.0
    queue_put_wait_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable snapshot of the metrics."""

        return {
            "submitted": self.submitted,
            "loaded": self.loaded,
            "failed": self.failed,
            "batches": self.batches,
            "route_counts": dict(self.route_counts),
            "route_seconds": dict(self.route_seconds),
            "prepare_seconds": self.prepare_seconds,
            "queue_put_wait_seconds": self.queue_put_wait_seconds,
        }


def _alpha_to_white_bgr(bg_or_bgra: np.ndarray) -> np.ndarray:
    if bg_or_bgra.ndim == 2:
        return cv2.cvtColor(bg_or_bgra, cv2.COLOR_GRAY2BGR)
    if bg_or_bgra.shape[2] == 3:
        return bg_or_bgra
    if bg_or_bgra.shape[2] == 4:
        bgr = bg_or_bgra[:, :, :3].astype(np.float32)
        a = bg_or_bgra[:, :, 3:4].astype(np.float32) / 255.0
        bgr = (bgr * a + 255.0 * (1.0 - a)).astype(np.uint8)
        return bgr
    return bg_or_bgra[:, :, :3]


def _resize_to_target_side(image: np.ndarray, *, interpolation: int | None = None) -> np.ndarray:
    """Resize image so the longest side is near the tagger target size."""

    height, width = image.shape[:2]
    side = max(height, width)
    if side == _TARGET:
        return image
    ratio = _TARGET / max(1, side)
    interp = interpolation if interpolation is not None else (cv2.INTER_AREA if side > _TARGET else cv2.INTER_CUBIC)
    return cv2.resize(
        image,
        (max(1, int(width * ratio)), max(1, int(height * ratio))),
        interpolation=interp,
    )


def _alpha_to_white_then_resize_bgr(bg_or_bgra: np.ndarray) -> np.ndarray:
    """Composite alpha over white before resizing to avoid transparent-edge fringes."""

    if bg_or_bgra.ndim == 2:
        return _resize_to_target_side(cv2.cvtColor(bg_or_bgra, cv2.COLOR_GRAY2BGR))
    if bg_or_bgra.shape[2] == 3:
        return _resize_to_target_side(bg_or_bgra)
    if bg_or_bgra.shape[2] != 4:
        return _resize_to_target_side(bg_or_bgra[:, :, :3])

    alpha_u8 = bg_or_bgra[:, :, 3]
    if bool(np.all(alpha_u8 == 255)):
        return _resize_to_target_side(bg_or_bgra[:, :, :3])

    # Use uint16 arithmetic to avoid the full-size float32 allocation that can
    # fail on large transparent images, while preserving "white composite before
    # resize" semantics.
    bgr = bg_or_bgra[:, :, :3].astype(np.uint16, copy=False)
    alpha = alpha_u8[:, :, np.newaxis].astype(np.uint16, copy=False)
    white = np.uint16(255)
    composited = ((bgr * alpha) + (white * (white - alpha)) + np.uint16(127)) // white
    return _resize_to_target_side(composited.astype(np.uint8, copy=False))


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
        tagger: Any,  # WD14Tagger インスタンス（prepare_batch_from_bgr を呼ぶ）
        batch_size: int,
        prefetch_batches: int = 2,
        io_workers: int | None = None,
    ) -> None:
        self._paths = list(paths)
        self._B = int(batch_size)
        self._depth = max(1, int(prefetch_batches))
        cpu = os.cpu_count() or 4
        default_workers = min(max(4, cpu), 16)
        current_workers = default_workers if io_workers is None else safe_int(io_workers, default_workers, min_value=1)
        env_workers = os.getenv("KE_IO_WORKERS")
        if env_workers is not None:
            current_workers = safe_int(env_workers, current_workers, min_value=1)
        self._io_workers = current_workers
        self._tagger = tagger
        self._producer_error: BaseException | None = None
        self._metrics = LoaderMetrics()
        self._metrics_lock = threading.Lock()

        # (paths, np_batch, sizes) or None(sentinal)
        self._q: "queue.Queue[tuple[list[str], np.ndarray, list[tuple[int,int]]] | None]" = queue.Queue(self._depth)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._producer, name="PL-Feeder", daemon=True)
        self._th.start()
        logger.info(
            "PrefetchLoaderPrepared: start (B=%d, depth=%d, io_workers=%d)", self._B, self._depth, self._io_workers
        )

    # --- 1 枚ロード（OpenCV 優先、失敗したら PIL） ---
    def _record_route(self, route: LoadRoute, seconds: float, *, ok: bool) -> None:
        """Record one decode attempt in the loader metrics."""

        with self._metrics_lock:
            self._metrics.submitted += 1
            if ok:
                self._metrics.loaded += 1
            else:
                self._metrics.failed += 1
            self._metrics.route_counts[route] = self._metrics.route_counts.get(route, 0) + 1
            self._metrics.route_seconds[route] = self._metrics.route_seconds.get(route, 0.0) + float(seconds)

    def metrics_snapshot(self) -> LoaderMetrics:
        """Return a stable copy of diagnostic loader metrics."""

        with self._metrics_lock:
            return LoaderMetrics(
                submitted=self._metrics.submitted,
                loaded=self._metrics.loaded,
                failed=self._metrics.failed,
                batches=self._metrics.batches,
                route_counts=dict(self._metrics.route_counts),
                route_seconds=dict(self._metrics.route_seconds),
                prepare_seconds=self._metrics.prepare_seconds,
                queue_put_wait_seconds=self._metrics.queue_put_wait_seconds,
            )

    def _load_one(self, p: str) -> tuple[str, np.ndarray | None, tuple[int, int] | None, LoadRoute]:
        route: LoadRoute = "opencv"
        started = time.perf_counter()
        try:
            data = np.fromfile(p, dtype=np.uint8)  # Windows で速い
            im = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError("cv2.imdecode failed")
            H0, W0 = (im.shape[0], im.shape[1]) if im.ndim >= 2 else (0, 0)
            bgr = _alpha_to_white_then_resize_bgr(im)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # 元サイズは im の生サイズ（アルファ有無に関係なし）
            self._record_route(route, time.perf_counter() - started, ok=True)
            return (p, rgb, (int(W0), int(H0)), route)

        except _DECODE_FALLBACK_ERRORS as e:
            logger.warning("PrefetchLoaderPrepared: failed to load %s: %s (fallback PIL)", p, e)
            # --- フォールバック: PIL ---
            try:
                route = "pil_fallback"
                with Image.open(p) as pil_image:
                    w, h = pil_image.size
                    rgba = pil_image.convert("RGBA")
                    bg = Image.new("RGBA", rgba.size, "WHITE")
                    bg.paste(rgba, mask=rgba.split()[-1])
                    rgb_image = bg.convert("RGB")
                    rgb_image.thumbnail((_TARGET, _TARGET), Image.Resampling.LANCZOS)
                    # bgr = np.asarray(rgb)[:, :, ::-1]  # RGB->BGR
                    rgb_arr = np.asarray(rgb_image)  # ここはそのままRGB
                    self._record_route(route, time.perf_counter() - started, ok=True)
                    return (p, rgb_arr, (w, h), route)
            except _DECODE_FALLBACK_ERRORS as e2:
                logger.warning("PrefetchLoaderPrepared: PIL fallback also failed %s: %s", p, e2)
                self._record_route("failed", time.perf_counter() - started, ok=False)
                return (p, None, None, "failed")

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except Exception:
            # Failure policy: qsize is diagnostic-only and must never fail the
            # tagging pipeline.
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
                        result = fut.result()
                        p, arr, sz = result[:3]
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
                    prepare_started = time.perf_counter()
                    np_batch = self._tagger.prepare_batch_from_rgb_np(bgr_list)
                    prepare_seconds = time.perf_counter() - prepare_started

                    # キューへ（必要なら put 時間をログ）
                    t0 = time.perf_counter()
                    q_before = self._q.qsize()
                    self._q.put((kept_paths, np_batch, sizes))
                    wait_put_seconds = time.perf_counter() - t0
                    wait_put_ms = wait_put_seconds * 1000.0
                    with self._metrics_lock:
                        self._metrics.batches += 1
                        self._metrics.prepare_seconds += prepare_seconds
                        self._metrics.queue_put_wait_seconds += wait_put_seconds
                    if wait_put_ms > 1.0 or q_before == self._depth - 1:
                        logger.info("LOAD put wait=%.1fms q=%d/%d", wait_put_ms, q_before, self._depth)

                    if self._stop.is_set():
                        break
        except Exception as e:
            # Failure policy: producer errors are fatal. The iterator re-raises
            # them when it observes the sentinel.
            self._producer_error = e
            logger.error("PrefetchLoaderPrepared: producer failed: %s", e, exc_info=True)
        finally:
            # 終端シグナルは必ず入れる（ブロッキングでOK）
            logger.info("PrefetchLoaderPrepared: producer finished; enqueue sentinel")
            self._q.put(None)  # block until there is space

    def __iter__(self) -> Iterator[Tuple[list[str], np.ndarray, list[tuple[int, int]]]]:
        while True:
            try:
                item = self._q.get(timeout=5.0)
            except queue.Empty:
                if self._stop.is_set():
                    logger.info("PrefetchLoaderPrepared: stop signaled; iterator exits")
                    return
                continue
            if item is None:
                if self._producer_error is not None:
                    raise RuntimeError("PrefetchLoaderPrepared producer failed") from self._producer_error
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        # 先にセンチネルを入れて消費側を解除（あっても重複は害なし）
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        # 残骸は掃除（任意）
        try:
            while True:
                self._q.get_nowait()
        except Exception as exc:
            # Failure policy: queue draining during close is best-effort cleanup.
            # The iterator path above is responsible for fatal producer errors.
            if not isinstance(exc, queue.Empty):
                logger.debug("PrefetchLoaderPrepared: queue drain cleanup failed: %s", exc)
            pass
        if self._th.is_alive():
            self._th.join(timeout=2.0)
        logger.info("PrefetchLoaderPrepared: stop")


__all__ = ["LoaderMetrics", "PrefetchLoaderPrepared"]
