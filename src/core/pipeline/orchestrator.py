from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from core.config import load_settings
from db.connection import bootstrap_if_needed
from tagger.base import ITagger
from utils.paths import ensure_dirs

from .scanner import Scanner
from .signature import current_tagger_sig
from .tagging import TaggingStage
from .types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter

logger = logging.getLogger(__name__)


class IndexPipeline:
    def __init__(
        self,
        db_path: str | Path,
        settings=None,
        *,
        tagger_override: ITagger | None = None,
        progress_cb: Callable[[IndexProgress], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> None:
        settings = settings or load_settings()
        bootstrap_if_needed(db_path)
        ensure_dirs()
        thresholds = {} if getattr(settings.tagger, "thresholds", None) is None else settings.tagger.thresholds
        # build maps via types helpers
        from .signature import _build_max_tags_map, _build_threshold_map

        thr_map = _build_threshold_map(thresholds)
        max_map = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))
        tagger_sig = current_tagger_sig(settings, thresholds=thr_map, max_tags=max_map)

        self.ctx = PipelineContext(
            db_path=str(db_path),
            settings=settings,
            thresholds=thr_map,
            max_tags_map=max_map,
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
        records, s = Scanner(self.ctx, self.emitter).scan()
        stats["scanned"] = s["scanned"]
        stats["new_or_changed"] = s["new_or_changed"]
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


def run_index_once(
    db_path: str | Path,
    settings=None,
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
