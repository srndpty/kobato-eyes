from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from PIL import Image

from core.config import PipelineSettings
from tagger.base import TagCategory

logger = logging.getLogger(__name__)


class IndexPhase(Enum):
    SCAN = "scan"
    TAG = "tag"
    FTS = "fts"
    DONE = "done"


@dataclass
class IndexProgress:
    phase: IndexPhase
    done: int
    total: int
    message: str | None = None


@dataclass
class PipelineContext:
    db_path: str | Path
    settings: "PipelineSettings"
    thresholds: dict["TagCategory", float]
    max_tags_map: dict["TagCategory", int]
    tagger_sig: str
    progress_cb: Callable[[IndexProgress], None] | None = None
    is_cancelled: Callable[[], bool] | None = None


@dataclass
class _FileRecord:
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


class ProgressEmitter:
    def __init__(self, cb: Callable[[IndexProgress], None] | None):
        self._cb = cb
        self._last: dict[IndexPhase, tuple[int, float]] = {}

    def emit(self, progress: IndexProgress, force: bool = False) -> None:
        if self._cb is None:
            return
        now = time.perf_counter()
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
