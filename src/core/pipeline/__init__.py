"""
Aggregate public API for core.pipeline (backward compatibility).
Re-exports are intentional; keep them listed in __all__ to satisfy linters.
"""

from .maintenance import _settle_after_quiesce, wait_for_unlock
from .manual_refresh import scan_and_tag
from .orchestrator import IndexPipeline, run_index_once
from .resolver import _resolve_tagger
from .retag import retag_all, retag_query
from .scanner import Scanner
from .signature import _build_max_tags_map, _build_threshold_map, current_tagger_sig
from .stages.scan_stage import ScanStage, ScanStageDeps, ScanStageResult
from .stages.tag_stage import TagStage, TagStageDeps, TagStageResult
from .stages.write_stage import WriteStage, WriteStageDeps, WriteStageResult
from .tagging import TaggingStage
from .testhooks import TaggingDeps
from .types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter, _FileRecord
from .watcher import ProcessingPipeline

__all__ = [
    # Orchestrator
    "IndexPipeline",
    "run_index_once",
    # Watcher
    "ProcessingPipeline",
    # Manual utilities
    "scan_and_tag",
    "retag_all",
    "retag_query",
    # Signature helpers
    "current_tagger_sig",
    "_build_threshold_map",
    "_build_max_tags_map",
    # Resolver
    "_resolve_tagger",
    # Maintenance helpers
    "wait_for_unlock",
    "_settle_after_quiesce",
    # Types
    "IndexPhase",
    "IndexProgress",
    "PipelineContext",
    "ProgressEmitter",
    "_FileRecord",
    # Stages (used by tests / advanced callers)
    "Scanner",
    "TaggingStage",
    "TaggingDeps",
    "ScanStage",
    "ScanStageDeps",
    "ScanStageResult",
    "TagStage",
    "TagStageDeps",
    "TagStageResult",
    "WriteStage",
    "WriteStageDeps",
    "WriteStageResult",
]
