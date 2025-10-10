"""Compatibility wrapper around the new scanning stage."""

from __future__ import annotations

from dataclasses import dataclass

from core.pipeline.types import PipelineContext, ProgressEmitter, _FileRecord

from .stages.scan_stage import ScanStage, ScanStageDeps, ScanStageResult


@dataclass(slots=True)
class ScanStats:
    """Simple structure mirroring the legacy scanner statistics."""

    scanned: int
    new_or_changed: int


class Scanner:
    """Backward-compatible wrapper delegating to :class:`ScanStage`."""

    def __init__(self, ctx: PipelineContext, emitter: ProgressEmitter, deps: ScanStageDeps | None = None) -> None:
        self._ctx = ctx
        self._emitter = emitter
        self._stage = ScanStage(deps=deps)

    def scan(self) -> tuple[list[_FileRecord], dict[str, int]]:
        result = self._stage.run(self._ctx, self._emitter)
        stats = {"scanned": result.scanned, "new_or_changed": result.new_or_changed}
        return result.records, stats


__all__ = ["Scanner", "ScanStats", "ScanStage", "ScanStageResult", "ScanStageDeps"]
