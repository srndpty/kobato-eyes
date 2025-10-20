from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from core.config import load_settings
from core.config.schema import PipelineSettings
from core.pipeline.types import PipelineContext, ProgressEmitter, _FileRecord
from core.scanner import DEFAULT_EXTENSIONS, iter_images
from db.connection import bootstrap_if_needed, get_conn
from db.repository import get_file_by_path, list_untagged_under_path, upsert_file
from tagger.base import ITagger
from utils.hash import compute_sha256
from utils.paths import get_db_path

from .resolver import _resolve_tagger
from .signature import current_tagger_sig
from .stages.tag_stage import TagStage
from .stages.write_stage import WriteStage
from .types import IndexPhase, IndexProgress

logger = logging.getLogger(__name__)


def scan_and_tag(
    root: Path,
    *,
    recursive: bool = True,
    batch_size: int = 8,
    hard_delete_missing: bool = False,
    settings: PipelineSettings | None = None,
    progress_cb: Callable[[IndexProgress], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, object]:
    """指定パス配下の未タグ画像をタグ付け（UIの手動更新用）。"""
    start_time = time.perf_counter()
    resolved_root = Path(root).expanduser()
    try:
        resolved_root = resolved_root.resolve(strict=False)
    except OSError:
        resolved_root = resolved_root.absolute()
    stats_out: dict[str, object] = {
        "queued": 0,
        "tagged": 0,
        "elapsed_sec": 0.0,
        "missing": 0,
        "soft_deleted": 0,
        "hard_deleted": 0,
    }

    root_exists = resolved_root.exists()
    if not root_exists:
        logger.info(
            "Manual tag refresh: path does not exist; will run DB cleanup only: resolved_root:%s, root:%s",
            resolved_root,
            root,
        )

    settings = settings or load_settings()
    if batch_size and getattr(settings, "batch_size", batch_size) != batch_size:
        try:
            override_batch = max(1, int(batch_size))
            settings = settings.model_copy(update={"batch_size": override_batch})
        except AttributeError:
            settings.batch_size = max(1, int(batch_size))
    allow_exts = {ext.lower() for ext in (settings.allow_exts or DEFAULT_EXTENSIONS)}
    # 非対応拡張子でも「掃除だけ」は実行したいので、ここでの早期 return はしない。
    skip_tagging_for_this_root = (
        root_exists and resolved_root.is_file() and resolved_root.suffix.lower() not in allow_exts
    )

    # thresholds = _build_threshold_map(settings.tagger.thresholds)
    # max_tags_map = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))
    db_path = get_db_path()
    bootstrap_if_needed(db_path)
    conn = get_conn(db_path, allow_when_quiesced=True)
    tagger: ITagger | None = None

    def _like_pattern_for_input(path: Path) -> str:
        """存在しないパスでも妥当な LIKE を作る（掃除用に使う）"""
        literal = str(path)
        # ディレクトリかどうかのヒント：
        # - 末尾が / or \ → ディレクトリ想定
        # - それ以外は、exists() が真なら実際の型を使う
        # - exists() が偽なら、拡張子がなければディレクトリ扱いに倒す
        looks_dir = literal.endswith(("/", "\\")) or (not root_exists and path.suffix == "")
        if (root_exists and path.is_dir()) or looks_dir:
            sep = "\\" if ("\\" in literal and "/" not in literal) else "/"
            if not literal.endswith(("/", "\\")):
                literal = literal + sep
            return f"{literal}%"
        return literal

    def _within_scope(candidate: Path) -> bool:
        """存在しない root のときも“スコープ内か”を判定できるように緩める。"""
        if root_exists and resolved_root.is_file():
            return candidate == resolved_root
        # root が存在しない場合は「ディレクトリ想定ならプレフィックス一致、ファイル想定なら完全一致」
        like_dir = (not root_exists and resolved_root.suffix == "") or (root_exists and resolved_root.is_dir())
        if like_dir:
            try:
                rel = candidate.relative_to(resolved_root)
                return recursive or len(rel.parts) <= 1
            except ValueError:
                return False
        else:
            return str(candidate) == str(resolved_root)

    cancelled = False

    def _cancelled() -> bool:
        nonlocal cancelled
        if cancelled:
            return True
        if is_cancelled is None:
            return False
        try:
            cancelled = bool(is_cancelled())
        except Exception:
            logger.exception("Refresh cancellation callback failed")
            cancelled = True
        return cancelled

    def _emit(progress: IndexProgress, *, force: bool = False) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(progress)
        except Exception:
            logger.exception("Refresh progress callback failed; disabling further updates")

    _emit(IndexProgress(phase=IndexPhase.SCAN, done=0, total=-1, message=str(resolved_root)), force=True)

    try:
        queued_paths: list[Path] = []
        seen: set[str] = set()
        fs_paths: set[str] = set()

        for _, stored_path in list_untagged_under_path(conn, _like_pattern_for_input(resolved_root)):
            if _cancelled():
                break
            path_obj = Path(stored_path)
            # ここでは「タグ付け候補の生成」。存在しないものはタグ付けできないので除外。
            if not path_obj.exists() or not _within_scope(path_obj):
                continue
            if path_obj.suffix.lower() not in allow_exts:
                continue
            literal = str(path_obj)
            fs_paths.add(literal)
            if literal in seen:
                continue
            queued_paths.append(path_obj)
            seen.add(literal)

        if root_exists and resolved_root.is_file():
            fs_iterable: Iterable[Path] = [resolved_root]
        elif root_exists:
            fs_iterable = iter_images([resolved_root], excluded=[], extensions=allow_exts)
        else:
            # ルートが存在しないなら、ファイルシステム側の走査は行わない（DB 掃除のみ）
            fs_iterable = []

        for candidate in fs_iterable:
            if _cancelled():
                break
            path_obj = Path(candidate)
            if not _within_scope(path_obj) or path_obj.suffix.lower() not in allow_exts:
                continue
            literal = str(path_obj)
            fs_paths.add(literal)
            if literal in seen:
                continue
            row = get_file_by_path(conn, literal)
            if row is None:
                queued_paths.append(path_obj)
                seen.add(literal)
                continue
            cursor = conn.execute("SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1", (row["id"],))
            try:
                if cursor.fetchone() is None:
                    queued_paths.append(path_obj)
                    seen.add(literal)
            finally:
                cursor.close()

        total = len(queued_paths)
        stats_out["queued"] = total
        _emit(IndexProgress(phase=IndexPhase.SCAN, done=total, total=total, message=str(resolved_root)), force=True)

        def _chunked(items: Sequence[int], size: int = 900) -> Iterator[list[int]]:
            for index in range(0, len(items), size):
                yield list(items[index : index + size])

        missing_ids: list[int] = []
        cursor = None
        try:
            if root_exists and resolved_root.is_file():
                cursor = conn.execute(
                    "SELECT id, path FROM files WHERE is_present = 1 AND path = ?", (str(resolved_root),)
                )
            else:
                cursor = conn.execute(
                    "SELECT id, path FROM files WHERE is_present = 1 AND path LIKE ?",
                    (_like_pattern_for_input(resolved_root),),
                )
            for row in cursor.fetchall():
                if _cancelled():
                    break
                path_text = str(row["path"])
                candidate = Path(path_text)
                if not _within_scope(candidate):
                    continue
                if path_text not in fs_paths:
                    missing_ids.append(int(row["id"]))
        finally:
            if cursor is not None:
                cursor.close()

        missing_count = len(missing_ids)
        stats_out["missing"] = missing_count
        soft_deleted = 0
        hard_deleted_count = 0
        removed = 0
        if missing_ids and not _cancelled():
            logger.info("Manual tag refresh: marking %d missing file(s) under %s", missing_count, resolved_root)
            if hard_delete_missing:
                with conn:
                    for chunk in _chunked(missing_ids):
                        if _cancelled():
                            break
                        placeholders = ", ".join("?" for _ in chunk)
                        conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({placeholders})", chunk)
                        conn.execute(f"DELETE FROM signatures WHERE file_id IN ({placeholders})", chunk)
                        from db.repository import fts_delete_rows as _fts_del

                        _fts_del(conn, chunk)
                        conn.execute(f"DELETE FROM files WHERE id IN ({placeholders})", chunk)
                        removed += len(chunk)
                        _emit(
                            IndexProgress(
                                phase=IndexPhase.FTS,
                                done=removed,
                                total=missing_count,
                                message="Hard delete missing",
                            )
                        )
                hard_deleted_count = missing_count
            else:
                with conn:
                    for chunk in _chunked(missing_ids):
                        if _cancelled():
                            break
                        placeholders = ", ".join("?" for _ in chunk)
                        conn.execute(
                            f"UPDATE files SET is_present = 0, deleted_at = CURRENT_TIMESTAMP WHERE id IN ({', '.join('?' for _ in chunk)})",
                            tuple(chunk),
                        )
                        from db.repository import fts_delete_rows as _fts_del

                        _fts_del(conn, chunk)
                        removed += len(chunk)
                        _emit(
                            IndexProgress(
                                phase=IndexPhase.FTS,
                                done=removed,
                                total=missing_count,
                                message="Mark missing",
                            )
                        )
                soft_deleted = missing_count
        stats_out["soft_deleted"] = soft_deleted
        stats_out["hard_deleted"] = hard_deleted_count

        if total == 0 or skip_tagging_for_this_root or _cancelled():
            elapsed = time.perf_counter() - start_time
            stats_out["elapsed_sec"] = elapsed
            stats_out["cancelled"] = cancelled
            if missing_count:
                logger.info(
                    "Manual tag refresh: removed %d missing file(s) (hard_delete=%s) in %.2fs",
                    missing_count,
                    hard_delete_missing,
                    elapsed,
                )
            else:
                logger.info("Manual tag refresh: no untagged files under %s (elapsed %.2fs)", resolved_root, elapsed)
            _emit(IndexProgress(phase=IndexPhase.DONE, done=1, total=1, message=str(resolved_root)), force=True)
            return stats_out

        logger.info(
            "Manual tag refresh: tagging %d file(s) under %s (recursive=%s)",
            total,
            resolved_root,
            recursive,
        )
        tagger_obj, th_fallback, max_tags_fallback = _resolve_tagger(
            settings,
            None,
            thresholds=None,
            max_tags=None,
        )
        tagger = tagger_obj
        effective_thresholds = th_fallback or None
        effective_max_tags = max_tags_fallback or None
        tagger_sig = current_tagger_sig(
            settings,
            thresholds=effective_thresholds,
            max_tags=effective_max_tags,
        )

        records: list[_FileRecord] = []
        _emit(IndexProgress(phase=IndexPhase.PREPARE, done=0, total=total, message="Preparing…"), force=True)

        for i, path_obj in enumerate(queued_paths, start=1):
            if _cancelled():
                break
            if i % 200 == 0:
                _emit(IndexProgress(phase=IndexPhase.PREPARE, done=i, total=total, message="Preparing…"))

            try:
                stat_result = path_obj.stat()
            except OSError as exc:
                logger.warning("Manual refresh: failed to stat %s: %s", path_obj, exc)
                continue

            row = get_file_by_path(conn, str(path_obj))
            is_new = row is None
            if row is not None:
                stored_size = int(row["size"] or 0)
                stored_mtime = float(row["mtime"] or 0.0)
            else:
                stored_size = 0
                stored_mtime = 0.0

            size_changed = is_new or stored_size != stat_result.st_size
            mtime_changed = is_new or stored_mtime != stat_result.st_mtime

            if is_new or size_changed or mtime_changed:
                try:
                    sha_hex = compute_sha256(path_obj)
                except OSError as exc:
                    logger.warning("Manual refresh: failed to hash %s: %s", path_obj, exc)
                    continue
                changed = True if is_new else (row is None or str(row["sha256"] or "") != sha_hex)
            else:
                sha_hex = str(row["sha256"] or "") if row is not None else ""
                changed = False

            indexed_at = None if changed else (row["indexed_at"] if row is not None else None)
            last_tagged_at = row["last_tagged_at"] if row is not None else None
            stored_sig = row["tagger_sig"] if row is not None else None
            width = row["width"] if row is not None else None
            height = row["height"] if row is not None else None

            file_id = upsert_file(
                conn,
                path=str(path_obj),
                size=stat_result.st_size,
                mtime=stat_result.st_mtime,
                sha256=sha_hex,
                width=width,
                height=height,
                indexed_at=indexed_at,
                tagger_sig=stored_sig,
                last_tagged_at=last_tagged_at,
                is_present=True,
                deleted_at=None,
            )

            cursor = conn.execute("SELECT 1 FROM file_tags WHERE file_id = ? LIMIT 1", (file_id,))
            try:
                has_tag = cursor.fetchone() is not None
            finally:
                cursor.close()

            records.append(
                _FileRecord(
                    file_id=file_id,
                    path=path_obj,
                    size=stat_result.st_size,
                    mtime=stat_result.st_mtime,
                    sha=sha_hex,
                    is_new=is_new,
                    changed=changed,
                    tag_exists=has_tag,
                    needs_tagging=True,
                    stored_tagger_sig=str(stored_sig) if stored_sig is not None else None,
                    current_tagger_sig=tagger_sig,
                    last_tagged_at=float(last_tagged_at) if last_tagged_at is not None else None,
                    width=int(width) if width is not None else None,
                    height=int(height) if height is not None else None,
                )
            )

        if not records or _cancelled():
            elapsed = time.perf_counter() - start_time
            stats_out["tagged"] = 0
            stats_out["elapsed_sec"] = elapsed
            stats_out["cancelled"] = cancelled
            logger.info("Manual tag refresh: no records to tag after preparation")
            _emit(IndexProgress(phase=IndexPhase.DONE, done=1, total=1, message=str(resolved_root)), force=True)
            return stats_out

        # ここで “準備フェーズ” の書き込みを確定し、接続を閉じる（EXCLUSIVE を通すため）
        try:
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass
        conn = None  # finally節で二重closeしないように

        ctx = PipelineContext(
            db_path=str(db_path),
            settings=settings,
            thresholds=effective_thresholds or {},
            max_tags_map=effective_max_tags or {},
            tagger_sig=tagger_sig,
            tagger_override=tagger_obj,
            progress_cb=progress_cb,
            is_cancelled=_cancelled,
        )
        emitter = ProgressEmitter(progress_cb)
        tag_stage = TagStage()
        write_stage = WriteStage()

        try:
            tag_result = tag_stage.run(ctx, emitter, records)
        except Exception:
            logger.exception("Manual tag refresh: tagging stage failed")
            tag_result = None

        tagged = tag_result.tagged_count if tag_result is not None else 0

        if tag_result is not None and not emitter.cancelled(ctx.is_cancelled):
            try:
                write_stage.run(ctx, emitter, tag_result)
            except Exception:
                logger.exception("Manual tag refresh: write stage failed")

        elapsed = time.perf_counter() - start_time
        stats_out["tagged"] = tagged
        stats_out["elapsed_sec"] = elapsed
        stats_out["cancelled"] = emitter.cancelled(ctx.is_cancelled)
        logger.info(
            "Manual tag refresh complete: tagged %d of %d file(s) in %.2fs (missing removed=%d, hard=%s)",
            tagged,
            total,
            elapsed,
            missing_count,
            hard_delete_missing,
        )
        _emit(IndexProgress(phase=IndexPhase.DONE, done=1, total=1, message=str(resolved_root)), force=True)
        return stats_out
    finally:
        if conn is not None:
            conn.close()
        if tagger is not None:
            closer = getattr(tagger, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.exception("Failed to close tagger after manual refresh")
        # --- ここから追加：積極的に GPU メモリを解放 ---
        try:
            import gc

            gc.collect()
        except Exception:
            pass


__all__ = ["scan_and_tag"]
