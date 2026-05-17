# DB / pipeline exception audit

## Scope

- `src/db`
- `src/services/db_writing.py`
- `src/core/pipeline`
- `src/core/pipeline/stages`

## Findings

### Must propagate

- `DBWritingService._flush_standard`: batch write failures must rollback and propagate. Fixed by avoiding helper-level commits inside the service transaction.
- `DBWritingService._flush_into_temp_tables`: temp staging failures must rollback and propagate. Fixed by explicit rollback.
- `WriteStage.run`: writer start, writer flush, and FTS rebuild failures must return `success=False`. Covered by tests.
- `scan_and_tag`: manual refresh must propagate tagging-stage exceptions and treat `WriteStageResult.success=False` as a failed refresh. Covered by tests.
- `scan_and_tag`: cancelled missing-file cleanup must report only actually processed soft/hard delete counts. Covered by tests.
- `PrefetchLoaderPrepared`: producer-thread failures such as tagger batch preparation errors must be re-raised by the iterator instead of ending as a successful empty stream. Covered by tests.

### Best effort cleanup

- `DBWritingService.stop`: queueing stop/flush sentinels may fail during shutdown; the stored worker exception is still propagated by `raise_if_failed`.
- `DBWritingService._restore_normal_mode`: restore statements run during unsafe-fast cleanup and should not mask the worker failure.
- `WriteStage.run`: final writer cleanup and post-quiesce settle failures should not mask the primary write failure.
- SQLite checkpoint and optimize calls are maintenance-only and should not fail the user-facing operation.
- `_settle_after_quiesce`: lock wait, checkpoint, optimize, and sweep connection failures remain best effort and are covered by warning-log assertions.

### Environment fallback

- Runtime PRAGMAs such as `mmap_size`, WAL checkpoint, and unsafe-fast lock acquisition are environment dependent. These paths should log or fall back, not fail indexing unless the main write transaction fails.
- Optional image loader fallbacks in `core.pipeline.loaders` should continue to prefer available decoders and skip individual undecodable images without failing the whole batch.

## Next candidates

- Review `services.db_writing` shutdown paths for any remaining cleanup failures that should be logged more explicitly.
- Add UI-level assertions that manual refresh worker errors surface through `IndexRunnable.signals.error`.
