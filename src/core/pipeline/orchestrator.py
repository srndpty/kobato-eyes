"""Pipeline orchestrator executing scanning, tagging, and writing stages."""

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
from .signature import _build_max_tags_map, _build_threshold_map, current_tagger_sig  # 既存
from .stages.scan_stage import ScanStageResult
from .stages.tag_stage import TagStage, TagStageResult
from .stages.write_stage import WriteStage, WriteStageResult
from .types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter
from .utils import (
    detect_tagger_provider,
    is_wd14_defaults,
    overlay_defaults,
    provider_default_max_tags,
    provider_default_thresholds,
)

logger = logging.getLogger(__name__)


class IndexPipeline:
    """High-level orchestrator that executes the indexing stages sequentially."""

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
        provider = detect_tagger_provider(settings)
        thresholds_cfg = getattr(settings.tagger, "thresholds", None) or {}
        # max_tags_cfg = getattr(settings.tagger, "max_tags", None)
        thr_map = _build_threshold_map(thresholds_cfg)
        max_map = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))

        # ★ pixai のときの既定値適用ロジック
        if provider == "pixai":
            # 「未指定」または「WD14既定ぽい数値」が渡ってきたら、pixai既定に差し替え
            if not thr_map or is_wd14_defaults(thr_map):
                thr_map = provider_default_thresholds("pixai")
            else:
                # ユーザー指定の無いカテゴリだけ pixai 既定で補完
                thr_map = overlay_defaults(thr_map, provider_default_thresholds("pixai"))

            # max_tags が未指定なら pixai 既定
            if not max_map:
                max_map = provider_default_max_tags("pixai")
            else:
                max_map = overlay_defaults(max_map, provider_default_max_tags("pixai"))

        else:
            # wd14 側も「完全未指定」ならWD14既定に寄せると安定
            if not thr_map:
                thr_map = provider_default_thresholds("wd14")

        tagger_sig = current_tagger_sig(settings, thresholds=thr_map, max_tags=max_map)
        # （任意）有効値の可視化ログ
        eff_thr = {k.name.lower(): float(v) for k, v in thr_map.items()}
        eff_max = {k.name.lower(): int(v) for k, v in max_map.items()}
        logger.info("Effective provider=%s thresholds=%s max_tags=%s", provider, eff_thr, eff_max)

        self.ctx = PipelineContext(
            db_path=str(db_path),
            settings=settings,
            thresholds=thr_map,
            max_tags_map=max_map,
            tagger_sig=tagger_sig,
            tagger_override=tagger_override,
            progress_cb=progress_cb,
            is_cancelled=is_cancelled,
        )
        self.emitter = ProgressEmitter(progress_cb)
        self._stage_overrides: dict[str, object] = {}
        self._scan_result: ScanStageResult | None = None
        self._tag_result: TagStageResult | None = None
        self._write_result: WriteStageResult | None = None

    def set_stage_override(self, name: str, stage: object) -> None:
        """Override a stage with a test double."""

        self._stage_overrides[name] = stage

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
        records = self._run_scan()
        stats["scanned"] = self._scan_result.scanned if self._scan_result else 0
        stats["new_or_changed"] = self._scan_result.new_or_changed if self._scan_result else 0

        cancelled = self.emitter.cancelled(self.ctx.is_cancelled)
        if not cancelled and records:
            tag_result = self._run_tag(records)
            if tag_result is not None:
                stats["tagged"] = tag_result.tagged_count
                cancelled = self.emitter.cancelled(self.ctx.is_cancelled)
                if not cancelled:
                    write_result = self._run_write(tag_result)
                    if write_result is not None:
                        stats["signatures"] = write_result.written

        stats["elapsed_sec"] = time.perf_counter() - start
        stats["cancelled"] = self.emitter.cancelled(self.ctx.is_cancelled)
        self.emitter.emit(IndexProgress(phase=IndexPhase.DONE, done=1, total=1), force=True)
        sec = stats["elapsed_sec"]
        logger.info(
            "Indexing complete: scanned=%d, new=%d, tagged=%d, (%dm:%ds)",
            stats["scanned"],
            stats["new_or_changed"],
            stats["tagged"],
            sec // 60,
            sec % 60,
        )
        return stats

    # ---- Stage execution helpers -------------------------------------------------

    def _stage(self, name: str, factory: Callable[[], object]) -> object:
        override = self._stage_overrides.get(name)
        if override is not None:
            return override
        return factory()

    def _run_scan(self) -> list:
        stage_obj = self._stage("scan", lambda: Scanner(self.ctx, self.emitter))
        if hasattr(stage_obj, "run") and not isinstance(stage_obj, Scanner):
            result = stage_obj.run(self.ctx, self.emitter)  # type: ignore[call-arg]
            self._scan_result = result
            return getattr(result, "records", [])

        if isinstance(stage_obj, Scanner):
            records, stats = stage_obj.scan()
            self._scan_result = ScanStageResult(
                records=records,
                scanned=stats.get("scanned", len(records)),
                new_or_changed=stats.get("new_or_changed", 0),
            )
            return records
        return []

    def _run_tag(self, records: list) -> TagStageResult | None:
        stage_obj = self._stage("tag", lambda: TagStage())
        if hasattr(stage_obj, "run"):
            result = stage_obj.run(self.ctx, self.emitter, records)  # type: ignore[call-arg]
            self._tag_result = result
            return result
        return None

    def _run_write(self, tag_result: TagStageResult) -> WriteStageResult | None:
        stage_obj = self._stage("write", lambda: WriteStage())
        if hasattr(stage_obj, "run"):
            result = stage_obj.run(self.ctx, self.emitter, tag_result)  # type: ignore[call-arg]
            self._write_result = result
            return result
        return None


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
