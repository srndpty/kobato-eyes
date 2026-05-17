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
- `IndexRunnable.run`: pre-run and runner exceptions must emit `signals.error` and must not emit `signals.finished`. Covered by tests.
- `PrefetchLoaderPrepared`: producer-thread failures such as tagger batch preparation errors must be re-raised by the iterator instead of ending as a successful empty stream. Covered by tests.
- `PrefetchLoaderPrepared`: unexpected IO worker failures must be re-raised through the iterator as producer failures. Covered by tests.

### Best effort cleanup

- `DBWritingService.stop`: queueing stop/flush sentinels may fail during shutdown; this is logged at warning level, and the stored worker exception is still propagated by `raise_if_failed`. Covered by tests.
- `DBWritingService._restore_normal_mode`: restore statements run during unsafe-fast cleanup and should not mask the worker failure. Transaction-close and checkpoint failures log at debug level; journal/locking/synchronous recovery failures log at warning level.
- `DBWritingService._emit_progress`: progress callback failures are user-extension boundary failures and are logged at warning level without failing the write. Covered by tests.
- `WriteStage.run`: final writer cleanup and post-quiesce settle failures should not mask the primary write failure.
- SQLite checkpoint and optimize calls are maintenance-only and should not fail the user-facing operation. DB writer checkpoint failures are logged at debug level. Covered by tests.
- `_settle_after_quiesce`: lock wait, checkpoint, optimize, and sweep connection failures remain best effort and are covered by warning-log assertions.

### Environment fallback

- Runtime PRAGMAs such as `mmap_size`, WAL checkpoint, and unsafe-fast lock acquisition are environment dependent. These paths should log or fall back, not fail indexing unless the main write transaction fails.
- Optional image loader fallbacks in `core.pipeline.loaders` should continue to prefer available decoders and skip individual undecodable images without failing the whole batch. Covered by tests.
- `ProcessingPipeline.enqueue_index`: path resolution failures fall back to absolute paths, and `finished`/`stop`/`shutdown` boundaries drain scheduled queue state. Covered by tests.

## Next candidates

- Add broader integration coverage for DB writer shutdown under real SQLite lock contention.
