from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from core.config import load_settings
from db.connection import get_conn
from utils.hash import compute_sha256

from .orchestrator import IndexPipeline
from .signature import current_tagger_sig
from .stages.scan_stage import ScanStageResult
from .types import IndexPhase, IndexProgress, PipelineContext, ProgressEmitter, _FileRecord

logger = logging.getLogger(__name__)

# SQLite の SQLITE_MAX_VARIABLE_NUMBER は既定 999。
# 安全側に振って 900 でチャンクします（必要なら環境に合わせて調整可）。
_SQL_IN_CHUNK = 900


def _chunked(seq, n=_SQL_IN_CHUNK):
    """seq を n 件ずつに分割してジェネレートする軽量ヘルパー。"""
    seq = list(seq)
    for i in range(0, len(seq), n):
        chunk = seq[i : i + n]
        if chunk:
            yield chunk


@dataclass(slots=True)
class RetagResult:
    """Result of marking files for re-tagging."""

    affected: int
    file_ids: list[int]


def retag_query(
    db_path: str | Path,
    where_sql: str,
    params: Sequence[object] | None,
) -> RetagResult:
    """Reset tagger state for rows matching a predicate and return affected ids."""
    predicate = where_sql.strip() or "1=1"
    arguments: tuple[object, ...] = tuple(params or ())
    conn = get_conn(db_path, allow_when_quiesced=True)
    try:
        # 対象の id をまず全件確定
        cursor = conn.execute("SELECT f.id FROM files AS f WHERE " + predicate, arguments)
        file_ids = [int(row[0]) for row in cursor.fetchall()]

        affected = 0
        if file_ids:
            # 900件ずつ UPDATE ... WHERE id IN (...)
            for chunk in _chunked(file_ids):
                placeholders = ",".join("?" for _ in chunk)
                sql = (
                    "UPDATE files AS f "
                    "SET tagger_sig = NULL, last_tagged_at = NULL "
                    "WHERE f.id IN (" + placeholders + ")"
                )
                conn.execute(sql, tuple(chunk))
                # 直近のステートメントで変更された行数
                affected += int(conn.execute("SELECT changes()").fetchone()[0] or 0)

        conn.commit()
        logger.info("Flagged %d file(s) for re-tagging (predicate=%s)", affected, predicate)
        # UI 側で選択を再現する用途があるため、id リストはそのまま返す
        return RetagResult(affected=int(affected), file_ids=file_ids)
    finally:
        conn.close()


def retag_all(
    db_path: str | Path,
    *,
    force: bool = False,
    settings=None,
) -> RetagResult:
    """Mark rows whose signature matches the current tagger (or everyone when forced)."""
    effective_settings = settings or load_settings()
    signature = current_tagger_sig(effective_settings)
    if force:
        predicate = "1=1"
        params: tuple[object, ...] = ()
    else:
        predicate = "tagger_sig = ?"
        params = (signature,)
    return retag_query(db_path, predicate, params)


class _RetagScanStage:
    """Scan stage that yields records for explicit file identifiers."""

    def __init__(self, file_ids: Iterable[int]) -> None:
        self._file_ids = [int(fid) for fid in dict.fromkeys(file_ids) if int(fid) > 0]

    def run(self, ctx: PipelineContext, emitter: ProgressEmitter) -> ScanStageResult:
        total = len(self._file_ids)
        emitter.emit(IndexProgress(phase=IndexPhase.SCAN, done=0, total=total, message="Retag"), force=True)
        if not self._file_ids:
            return ScanStageResult(records=[], scanned=0, new_or_changed=0)

        conn = get_conn(ctx.db_path, allow_when_quiesced=True)
        try:
            records: list[_FileRecord] = []
            new_or_changed = 0
            scanned = 0

            # 900件ずつ SELECT ... WHERE id IN (...)
            for id_chunk in _chunked(self._file_ids):
                placeholders = ",".join("?" for _ in id_chunk)
                sql = (
                    "SELECT id, path, size, mtime, sha256, indexed_at, tagger_sig, last_tagged_at, width, height "
                    "FROM files WHERE id IN (" + placeholders + ") AND is_present = 1"
                )
                rows = conn.execute(sql, tuple(id_chunk)).fetchall()

                for row in rows:
                    file_id = int(row["id"])
                    path = Path(row["path"])

                    # 進捗表示（ファイルごと）
                    emitter.emit(
                        IndexProgress(
                            phase=IndexPhase.SCAN,
                            done=scanned,
                            total=total,
                            message=str(path),
                        ),
                        force=False,
                    )
                    scanned += 1

                    # ファイル現況チェック（サイズ/mtime/sha）
                    try:
                        stat = path.stat()
                    except OSError as exc:
                        logger.warning("Retag scan: failed to stat %s: %s", path, exc)
                        continue

                    stored_size = int(row["size"] or 0)
                    stored_mtime = float(row["mtime"] or 0.0)
                    size_changed = stored_size != stat.st_size
                    mtime_changed = stored_mtime != stat.st_mtime

                    if size_changed or mtime_changed:
                        try:
                            sha = compute_sha256(path)
                        except OSError as exc:
                            logger.warning("Retag scan: failed to hash %s: %s", path, exc)
                            continue
                        changed = str(row["sha256"] or "") != sha
                    else:
                        sha = str(row["sha256"] or "")
                        changed = False

                    if changed:
                        new_or_changed += 1
                        indexed_at = None
                    else:
                        indexed_at = row["indexed_at"]

                    conn.execute(
                        "UPDATE files SET size = ?, mtime = ?, sha256 = ?, indexed_at = ? WHERE id = ?",
                        (stat.st_size, stat.st_mtime, sha, indexed_at, file_id),
                    )

                    cursor = conn.execute("SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1", (file_id,))
                    try:
                        tag_exists = cursor.fetchone() is not None
                    finally:
                        cursor.close()

                    stored_sig = row["tagger_sig"]
                    stored_tagged = row["last_tagged_at"]

                    records.append(
                        _FileRecord(
                            file_id=file_id,
                            path=path,
                            size=stat.st_size,
                            mtime=stat.st_mtime,
                            sha=sha,
                            is_new=False,
                            changed=changed,
                            tag_exists=tag_exists,
                            needs_tagging=True,  # 明示リタグなので True 固定
                            stored_tagger_sig=str(stored_sig) if stored_sig is not None else None,
                            current_tagger_sig=ctx.tagger_sig,
                            last_tagged_at=float(stored_tagged) if stored_tagged is not None else None,
                            width=int(row["width"]) if row["width"] is not None else None,
                            height=int(row["height"]) if row["height"] is not None else None,
                        )
                    )

            conn.commit()
            emitter.emit(IndexProgress(phase=IndexPhase.SCAN, done=total, total=total), force=True)
            return ScanStageResult(records=records, scanned=len(records), new_or_changed=new_or_changed)
        finally:
            conn.close()


def run_retag_selection(
    db_path: str | Path,
    file_ids: Sequence[int],
    *,
    settings=None,
    progress_cb: Callable[[IndexProgress], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """Run the indexing pipeline restricted to the provided file identifiers."""

    pipeline = IndexPipeline(
        db_path=db_path,
        settings=settings,
        progress_cb=progress_cb,
        is_cancelled=is_cancelled,
    )
    pipeline.set_stage_override("scan", _RetagScanStage(file_ids))
    stats = pipeline.run()
    stats["retagged"] = stats.get("tagged", 0)
    return stats


__all__ = ["RetagResult", "retag_query", "retag_all", "run_retag_selection"]
