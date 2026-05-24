"""Background workers for duplicate scanning and refinement."""

from __future__ import annotations

import logging
import os
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

from core.fastsig import fast_fill_missing_signatures
from dup.scanner import DuplicateFile, DuplicateScanConfig
from ui.dup_refine_parallel import refine_by_pixels_parallel, refine_by_tilehash_parallel
from ui.viewmodels import DupViewModel

logger = logging.getLogger(__name__)
DEBUG_THUMBS = False


def _as_int(value: object) -> int:
    """Convert database scalar values to int with a clear type boundary."""

    if isinstance(value, (str, bytes, bytearray)) or hasattr(value, "__int__"):
        return int(value)  # type: ignore[arg-type]
    raise TypeError(f"expected int-compatible value, got {type(value).__name__}")


class RefinePipelineSignals(QObject):
    """Signals emitted by duplicate refinement."""

    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    canceled = pyqtSignal()
    error = pyqtSignal(str)


class RefinePipelineRunnable(QRunnable):
    """Run tile-hash and optional pixel refinement in the background."""

    def __init__(self, clusters, tile_params, pixel_params) -> None:
        super().__init__()
        self.signals = RefinePipelineSignals()
        self._clusters = clusters
        self._tile_params = tile_params or {}
        self._pixel_params = pixel_params or {}
        self._cancelled = False

    def cancel(self) -> None:
        """Request cooperative cancellation."""

        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        """Run refinement."""

        try:

            def tick_tile(done, total, phase) -> None:
                stage = "TileHash 1/2" if phase == 1 else "TileHash 2/2"
                self.signals.progress.emit(done, total, stage)

            refined = refine_by_tilehash_parallel(
                self._clusters,
                tick=tick_tile,
                is_cancelled=self._is_cancelled,
                **self._tile_params,
            )
            if self._cancelled:
                self.signals.canceled.emit()
                return

            if self._pixel_params:

                def tick_px(done, total) -> None:
                    self.signals.progress.emit(done, total, "Pixel MAE")

                refined = refine_by_pixels_parallel(
                    refined,
                    tick=tick_px,
                    is_cancelled=self._is_cancelled,
                    **self._pixel_params,
                )
                if self._cancelled:
                    self.signals.canceled.emit()
                    return

            self.signals.finished.emit(refined)
        except Exception as exc:
            self.signals.error.emit(str(exc))


@dataclass(frozen=True)
class DuplicateScanRequest:
    """Parameters supplied to the duplicate scanning worker."""

    path_like: str | None
    hamming_threshold: int
    size_ratio: float | None


class DuplicateScanSignals(QObject):
    """Signals emitted by the scanning worker."""

    progress = pyqtSignal(int, int)
    progressState = pyqtSignal(int, int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)


class DuplicateScanRunnable(QRunnable):
    """Background runnable that loads file metadata and clusters duplicates."""

    def __init__(self, view_model: DupViewModel, db_path: Path, request: DuplicateScanRequest) -> None:
        super().__init__()
        self._view_model = view_model
        self._db_path = db_path
        self._request = request
        self.signals = DuplicateScanSignals()

    def _emit_progress(self, current: int, total: int, stage: str) -> None:
        """Emit both legacy and stage-aware progress signals."""

        self.signals.progressState.emit(current, total, stage)
        self.signals.progress.emit(current, total)

    def run(self) -> None:
        """Run duplicate scanning."""

        try:
            logger.info("scan: start")
            self._emit_progress(0, -1, "Loading files")
            with closing(self._view_model.open_connection(self._db_path)) as conn:
                rows = self._view_model.iter_files_for_dup(conn, self._request.path_like)
                logger.info("scan: rows loaded = %d", len(rows))
            self._emit_progress(len(rows), len(rows), "Loading files")

            missing: list[tuple[int, str]] = []
            for row in rows:
                file_id = _as_int(row["file_id"])
                path = str(row["path"])
                if row.get("phash_u64") is None:
                    missing.append((file_id, path))
            if missing:
                logger.info("sig: missing=%d -> computing in parallel ...", len(missing))

                def tick(done, total) -> None:
                    self._emit_progress(done, total, "Computing signatures")

                computed = fast_fill_missing_signatures(
                    str(self._db_path),
                    missing,
                    max_workers=int(os.environ.get("KE_SIG_WORKERS", "8")),
                    chunksize=int(os.environ.get("KE_SIG_CHUNK", "64")),
                    progress=tick,
                    apply_to_db=True,
                    unsafe_fast=True,
                )
                computed_by_id = {file_id: (phash, dhash) for (file_id, phash, dhash) in computed}
                patched = 0
                for row in rows:
                    pair = computed_by_id.get(_as_int(row["file_id"]))
                    if pair is not None:
                        row["phash_u64"] = int(pair[0])
                        patched += 1
                logger.info("sig: computed=%d, patched_rows=%d", len(computed), patched)

            total = len(rows)
            self._emit_progress(0, total, "Building groups")
            files: list[DuplicateFile] = []
            bad_rows = 0
            for index, row in enumerate(rows, start=1):
                try:
                    files.append(DuplicateFile.from_row(row))
                except ValueError:
                    bad_rows += 1
                    continue
                if index % 500 == 0:
                    self._emit_progress(index, total, "Building groups")
            self._emit_progress(total, total, "Building groups")

            logger.setLevel(logging.DEBUG)
            logger.info("scan: files built = %d (skipped rows=%d)", len(files), bad_rows)
            if files:
                logger.info("scan: sample phash head = %s", [hex(file.phash & ((1 << 64) - 1)) for file in files[:5]])
            config = DuplicateScanConfig(
                hamming_threshold=self._request.hamming_threshold,
                size_ratio=self._request.size_ratio,
            )

            logger.info("cluster: building ...")
            self._emit_progress(0, -1, "Clustering duplicates")
            started = time.perf_counter()
            clusters = self._view_model.build_clusters(config, files)
            logger.info("cluster: done; n=%d, %.2fs", len(clusters), time.perf_counter() - started)
            self._emit_progress(total, total, "Clustering duplicates")
            self.signals.finished.emit(clusters)
        except Exception as exc:
            logger.exception("scan worker crashed: %s", exc)
            try:
                self.signals.error.emit(str(exc))
            except RuntimeError:
                pass


class ThumbSignals(QObject):
    """Signals emitted by duplicate thumbnail jobs."""

    done = pyqtSignal(str, object)


class ThumbJob(QRunnable):
    """Generate or load one duplicate thumbnail in the background."""

    def __init__(
        self,
        view_model: DupViewModel,
        path: Path,
        size: tuple[int, int],
        cache_dir: Path,
        signals: ThumbSignals,
    ) -> None:
        super().__init__()
        self._view_model = view_model
        self._path = path
        self._size = size
        self._cache_dir = cache_dir
        self._signals = signals

    def run(self) -> None:
        """Run thumbnail generation and emit a QImage-compatible result."""

        qimg = None
        try:
            try:
                self._view_model.generate_thumbnail(self._path, self._cache_dir, size=self._size, format="WEBP")
            except Exception as exc:
                if DEBUG_THUMBS:
                    logger.info("thumb gen failed: %s %s", self._path, exc)

            with Image.open(self._path) as image:
                image.load()
                rgb_image = image.convert("RGB")
                rgb_image.thumbnail(self._size, Image.Resampling.LANCZOS)
                qimg = ImageQt(rgb_image).copy()
        except Exception as exc:
            if DEBUG_THUMBS:
                logger.info("thumb worker error: %s %s", self._path, exc)
            qimg = None
        finally:
            try:
                self._signals.done.emit(str(self._path), qimg)
                if DEBUG_THUMBS:
                    logger.info("thumb emitted: %s %s", self._path, bool(qimg))
            except Exception as exc:
                logger.info("thumb emit failed: %s", exc)


__all__ = [
    "DuplicateScanRequest",
    "DuplicateScanRunnable",
    "DuplicateScanSignals",
    "RefinePipelineRunnable",
    "RefinePipelineSignals",
    "ThumbJob",
    "ThumbSignals",
]
