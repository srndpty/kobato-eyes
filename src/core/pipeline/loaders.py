from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


_USE_TJ = os.getenv("KE_USE_TURBOJPEG", "1") != "0"
_TJ = None
if _USE_TJ:
    try:
        from turbojpeg import TJPF, TurboJPEG  # type: ignore

        _TJ = TurboJPEG()
    except Exception as e:
        logger.info("TurboJPEG not available (%s); falling back to OpenCV/PIL", e)
        _TJ = None

_TARGET = 448


def _choose_tj_scale(w: int, h: int) -> tuple[int, int]:
    side = max(w, h)
    if side >= _TARGET * 8:
        return (1, 8)
    if side >= _TARGET * 4:
        return (1, 4)
    if side >= _TARGET * 2:
        return (1, 2)
    return (1, 1)


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
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return (p, rgb, (w, h))

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
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # 元サイズは im の生サイズ（アルファ有無に関係なし）
            H0, W0 = (im.shape[0], im.shape[1]) if im.ndim >= 2 else (hh, ww)
            return (p, rgb, (int(W0), int(H0)))

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
                    # bgr = np.asarray(rgb)[:, :, ::-1]  # RGB->BGR
                    rgb_arr = np.asarray(rgb)  # ここはそのままRGB
                    return (p, rgb_arr, (w, h))
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
                    np_batch = self._tagger.prepare_batch_from_rgb_np(bgr_list)

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
            while True:
                try:
                    self._q.put(None, timeout=1)
                    break
                except queue.Full:
                    try:
                        self._q.get(timeout=1)
                    except queue.Empty:
                        continue

    def __iter__(self) -> Iterator[Tuple[list[str], np.ndarray, list[tuple[int, int]]]]:
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item

    def close(self) -> None:
        self._stop.set()
        while True:
            try:
                self._q.put(None, timeout=1)
                break
            except queue.Full:
                try:
                    self._q.get(timeout=1)
                except queue.Empty:
                    continue
        if self._th.is_alive():
            self._th.join(timeout=2.0)
        logger.info("PrefetchLoaderPrepared: stop")
