"""Compatibility wrappers for the refactored tagging and writing stages."""

from __future__ import annotations

from core.pipeline.types import PipelineContext, ProgressEmitter, _FileRecord

from .stages.tag_stage import TagStage, TagStageDeps, TagStageResult
from .stages.write_stage import WriteStage, WriteStageDeps, WriteStageResult


class TaggingStage:
    """Backward-compatible wrapper composing tagging and write stages."""

    def __init__(
        self,
        ctx: PipelineContext,
        emitter: ProgressEmitter,
        deps: TagStageDeps | None = None,
        writer_deps: WriteStageDeps | None = None,
    ) -> None:
        self._ctx = ctx
        self._emitter = emitter
        print(f"TaggingStage: ctx.thresholds={ctx.thresholds}, ctx.max_tags_map={ctx.max_tags_map}")
        self._tag_stage = TagStage(deps=deps)
        self._write_stage = WriteStage(deps=writer_deps)

    def run(self, records: list[_FileRecord]) -> tuple[int, int, list[tuple[int, object]]]:
        tag_result = self._tag_stage.run(self._ctx, self._emitter, records)
        write_result = self._write_stage.run(self._ctx, self._emitter, tag_result)
        return tag_result.tagged_count, write_result.fts_processed, []


__all__ = [
    "TaggingStage",
    "TagStage",
    "TagStageResult",
    "TagStageDeps",
    "WriteStage",
    "WriteStageResult",
    "WriteStageDeps",
]
