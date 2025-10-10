from __future__ import annotations

import logging
from pathlib import Path

from core.pipeline.types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter, _FileRecord
from core.scanner import DEFAULT_EXTENSIONS, iter_images
from db.connection import get_conn
from db.repository import get_file_by_path, upsert_file
from utils.hash import compute_sha256

logger = logging.getLogger(__name__)


class Scanner:
    def __init__(self, ctx: PipelineContext, emitter: ProgressEmitter):
        self.ctx = ctx
        self.emitter = emitter

    def scan(self) -> tuple[list[_FileRecord], dict[str, int]]:
        settings = self.ctx.settings
        roots = [Path(root).expanduser() for root in settings.roots if root]
        roots = [r for r in roots if r.exists()]
        excluded_paths = [Path(p).expanduser() for p in settings.excluded if p]
        allow_exts = {ext.lower() for ext in (settings.allow_exts or DEFAULT_EXTENSIONS)}

        stats = {"scanned": 0, "new_or_changed": 0}
        if not roots:
            logger.info("No valid roots configured; skipping scan.")
            return [], stats

        db_literal = str(self.ctx.db_path)
        conn = get_conn(
            db_literal
            if (db_literal.startswith("file:") or db_literal == ":memory:")
            else Path(self.ctx.db_path).expanduser(),
            allow_when_quiesced=True,
        )

        records: list[_FileRecord] = []
        try:
            logger.info("Scanning %d root(s) for eligible images", len(roots))
            for image_path in iter_images(roots, excluded=excluded_paths, extensions=allow_exts):
                if self.emitter.cancelled(self.ctx.is_cancelled):
                    break
                stats["scanned"] += 1
                self.emitter.emit(
                    IndexProgress(phase=IndexPhase.SCAN, done=stats["scanned"], total=-1, message=str(image_path))
                )
                try:
                    stat = image_path.stat()
                except OSError as exc:
                    logger.warning("Failed to stat %s: %s", image_path, exc)
                    continue

                row = get_file_by_path(conn, str(image_path))
                is_new = row is None
                if row is not None:
                    size_changed = int(row["size"] or 0) != stat.st_size
                    mtime_changed = float(row["mtime"] or 0.0) != stat.st_mtime
                else:
                    size_changed = True
                    mtime_changed = True

                if is_new or size_changed or mtime_changed:
                    try:
                        sha = compute_sha256(image_path)
                    except OSError as exc:
                        logger.warning("Failed to hash %s: %s", image_path, exc)
                        continue
                    changed = True if is_new else (str(row["sha256"] or "") != sha)  # type: ignore[index]
                else:
                    sha = str(row["sha256"] or "")  # type: ignore[index]
                    changed = False

                indexed_at = None if changed else (row["indexed_at"] if row else None)
                file_id = upsert_file(
                    conn,
                    path=str(image_path),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    sha256=sha,
                    indexed_at=indexed_at,
                )

                tag_exists = (
                    conn.execute("SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1", (file_id,)).fetchone() is not None
                )
                stored_sig = str(row["tagger_sig"]) if (row is not None and row["tagger_sig"] is not None) else None
                stored_tagged_at = row["last_tagged_at"] if row is not None else None
                last_tagged_at = float(stored_tagged_at) if stored_tagged_at is not None else None
                needs_tagging = is_new or changed or (not tag_exists) or (stored_sig != self.ctx.tagger_sig)

                records.append(
                    _FileRecord(
                        file_id=file_id,
                        path=image_path,
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        sha=sha,
                        is_new=is_new,
                        changed=changed,
                        tag_exists=tag_exists,
                        needs_tagging=needs_tagging,
                        stored_tagger_sig=stored_sig,
                        current_tagger_sig=self.ctx.tagger_sig,
                        last_tagged_at=last_tagged_at,
                    )
                )
            conn.commit()

            stats["new_or_changed"] = sum(1 for r in records if r.is_new or r.changed)
            logger.info("Scan complete: %d file(s) seen, %d new or changed", stats["scanned"], stats["new_or_changed"])
            self.emitter.emit(
                IndexProgress(phase=IndexPhase.SCAN, done=stats["scanned"], total=stats["scanned"]), force=True
            )
            return records, stats
        finally:
            conn.close()


__all__ = ["Scanner"]
