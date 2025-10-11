"""Scanning stage for the indexing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from core.pipeline.types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter, _FileRecord
from core.scanner import DEFAULT_EXTENSIONS, iter_images
from utils.hash import compute_sha256

logger = logging.getLogger(__name__)


class ScanStageDeps(Protocol):
    """Protocol for database operations required by :class:`ScanStage`."""

    def get_connection(self, db_path: str) -> "DBConnection":
        """Return a database connection for the given path."""

    def fetch_file(self, conn: "DBConnection", path: str):
        """Fetch a file row from the database by path."""

    def upsert_file(
        self,
        conn: "DBConnection",
        *,
        path: str,
        size: int,
        mtime: float,
        sha256: str,
        indexed_at: float | None,
    ) -> int:
        """Insert or update a file row and return its identifier."""

    def has_tag(self, conn: "DBConnection", file_id: int) -> bool:
        """Return ``True`` if at least one tag exists for the file."""


class DBConnection(Protocol):
    """Subset of the SQLite connection interface used by the scan stage."""

    def execute(self, sql: str, params: tuple[object, ...] | list[object]) -> object: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


class _DefaultScanStageDeps:
    """Runtime implementation of :class:`ScanStageDeps`."""

    def get_connection(self, db_path: str) -> DBConnection:
        from db.connection import get_conn

        literal = str(db_path)
        path = db_path if literal.startswith("file:") or literal == ":memory:" else Path(db_path).expanduser()
        return get_conn(path, allow_when_quiesced=True)

    def fetch_file(self, conn: DBConnection, path: str):
        from db.repository import get_file_by_path

        return get_file_by_path(conn, path)

    def upsert_file(
        self,
        conn: DBConnection,
        *,
        path: str,
        size: int,
        mtime: float,
        sha256: str,
        indexed_at: float | None,
    ) -> int:
        from db.repository import upsert_file

        return upsert_file(
            conn,
            path=path,
            size=size,
            mtime=mtime,
            sha256=sha256,
            indexed_at=indexed_at,
        )

    def has_tag(self, conn: DBConnection, file_id: int) -> bool:
        cursor = conn.execute("SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1", (file_id,))
        return cursor.fetchone() is not None


@dataclass(slots=True)
class ScanStageResult:
    """Result produced by :class:`ScanStage`."""

    records: list[_FileRecord]
    scanned: int
    new_or_changed: int


class ScanStage:
    """Stage responsible for scanning filesystem roots and syncing the DB."""

    def __init__(self, deps: ScanStageDeps | None = None) -> None:
        self._deps = deps or _DefaultScanStageDeps()

    def run(self, ctx: PipelineContext, emitter: ProgressEmitter) -> ScanStageResult:
        """Execute the scan step and return discovered records."""

        settings = ctx.settings
        roots = [Path(root).expanduser() for root in settings.roots if root]
        roots = [root for root in roots if root.exists()]
        excluded_paths = [Path(p).expanduser() for p in settings.excluded if p]
        allow_exts = {ext.lower() for ext in (settings.allow_exts or DEFAULT_EXTENSIONS)}

        stats = {"scanned": 0, "new_or_changed": 0}
        if not roots:
            logger.info("No valid roots configured; skipping scan.")
            return ScanStageResult(records=[], scanned=0, new_or_changed=0)

        conn = self._deps.get_connection(str(ctx.db_path))

        records: list[_FileRecord] = []
        try:
            logger.info("Scanning %d root(s) for eligible images", len(roots))
            for image_path in iter_images(roots, excluded=excluded_paths, extensions=allow_exts):
                if emitter.cancelled(ctx.is_cancelled):
                    break

                stats["scanned"] += 1
                emitter.emit(
                    IndexProgress(
                        phase=IndexPhase.SCAN,
                        done=stats["scanned"],
                        total=-1,
                        message=str(image_path),
                    )
                )
                try:
                    stat = image_path.stat()
                except OSError as exc:
                    logger.warning("Failed to stat %s: %s", image_path, exc)
                    continue

                row = self._deps.fetch_file(conn, str(image_path))
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
                file_id = self._deps.upsert_file(
                    conn,
                    path=str(image_path),
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    sha256=sha,
                    indexed_at=indexed_at,
                )

                tag_exists = self._deps.has_tag(conn, file_id)
                stored_sig = str(row["tagger_sig"]) if (row is not None and row["tagger_sig"] is not None) else None
                stored_tagged_at = row["last_tagged_at"] if row is not None else None
                last_tagged_at = float(stored_tagged_at) if stored_tagged_at is not None else None
                needs_tagging = is_new or changed or (not tag_exists) or (stored_sig != ctx.tagger_sig)

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
                        current_tagger_sig=ctx.tagger_sig,
                        last_tagged_at=last_tagged_at,
                    )
                )
            conn.commit()

            stats["new_or_changed"] = sum(1 for r in records if r.is_new or r.changed)
            logger.info(
                "Scan complete: %d file(s) seen, %d new or changed",
                stats["scanned"],
                stats["new_or_changed"],
            )
            emitter.emit(
                IndexProgress(phase=IndexPhase.SCAN, done=stats["scanned"], total=stats["scanned"]),
                force=True,
            )
            return ScanStageResult(records=records, scanned=stats["scanned"], new_or_changed=stats["new_or_changed"])
        finally:
            conn.close()


__all__ = ["ScanStage", "ScanStageResult", "ScanStageDeps"]
