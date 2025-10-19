from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from core.config import load_settings
from core.scanner import DEFAULT_EXTENSIONS, iter_images
from core.tag_job import TagJobConfig, run_tag_job
from db.connection import bootstrap_if_needed, get_conn
from db.repository import get_file_by_path, list_untagged_under_path
from tagger.base import ITagger
from utils.paths import get_db_path

from .resolver import _resolve_tagger
from .signature import _build_max_tags_map, _build_threshold_map, current_tagger_sig

logger = logging.getLogger(__name__)


def scan_and_tag(
    root: Path,
    *,
    recursive: bool = True,
    batch_size: int = 8,
    hard_delete_missing: bool = False,
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

    settings = load_settings()
    allow_exts = {ext.lower() for ext in (settings.allow_exts or DEFAULT_EXTENSIONS)}
    # 非対応拡張子でも「掃除だけ」は実行したいので、ここでの早期 return はしない。
    skip_tagging_for_this_root = (
        root_exists and resolved_root.is_file() and resolved_root.suffix.lower() not in allow_exts
    )

    thresholds = _build_threshold_map(settings.tagger.thresholds)
    max_tags_map = _build_max_tags_map(getattr(settings.tagger, "max_tags", None))
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

    try:
        queued_paths: list[Path] = []
        seen: set[str] = set()
        fs_paths: set[str] = set()

        for _, stored_path in list_untagged_under_path(conn, _like_pattern_for_input(resolved_root)):
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
        if missing_ids:
            logger.info("Manual tag refresh: marking %d missing file(s) under %s", missing_count, resolved_root)
            if hard_delete_missing:
                with conn:
                    for chunk in _chunked(missing_ids):
                        placeholders = ", ".join("?" for _ in chunk)
                        conn.execute(f"DELETE FROM file_tags WHERE file_id IN ({placeholders})", chunk)
                        conn.execute(f"DELETE FROM signatures WHERE file_id IN ({placeholders})", chunk)
                        from db.repository import fts_delete_rows as _fts_del

                        _fts_del(conn, chunk)
                        conn.execute(f"DELETE FROM files WHERE id IN ({placeholders})", chunk)
                hard_deleted_count = missing_count
            else:
                with conn:
                    for chunk in _chunked(missing_ids):
                        placeholders = ", ".join("?" for _ in chunk)
                        conn.execute(
                            f"UPDATE files SET is_present = 0, deleted_at = CURRENT_TIMESTAMP WHERE id IN ({', '.join('?' for _ in chunk)})",
                            tuple(chunk),
                        )
                        from db.repository import fts_delete_rows as _fts_del

                        _fts_del(conn, chunk)
                soft_deleted = missing_count
        stats_out["soft_deleted"] = soft_deleted
        stats_out["hard_deleted"] = hard_deleted_count

        if total == 0 or skip_tagging_for_this_root:
            elapsed = time.perf_counter() - start_time
            stats_out["elapsed_sec"] = elapsed
            if missing_count:
                logger.info(
                    "Manual tag refresh: removed %d missing file(s) (hard_delete=%s) in %.2fs",
                    missing_count,
                    hard_delete_missing,
                    elapsed,
                )
            else:
                logger.info("Manual tag refresh: no untagged files under %s (elapsed %.2fs)", resolved_root, elapsed)
            return stats_out

        logger.info("Manual tag refresh: tagging %d file(s) under %s (recursive=%s)", total, resolved_root, recursive)
        tagger_obj, th_fallback, max_tags_fallback = _resolve_tagger(
            settings,
            None,
            thresholds=thresholds,
            max_tags=max_tags_map,
        )
        tagger = tagger_obj
        effective_thresholds = thresholds or th_fallback or None
        effective_max_tags = max_tags_map or max_tags_fallback or None
        tagger_sig = current_tagger_sig(
            settings,
            thresholds=effective_thresholds,
            max_tags=effective_max_tags,
        )
        config = TagJobConfig(
            thresholds=effective_thresholds,
            max_tags=effective_max_tags,
            tagger_sig=tagger_sig,
        )

        tagged = 0
        for index, path_obj in enumerate(queued_paths, start=1):
            logger.info("Manual tag refresh progress: %d/%d %s", index, total, path_obj)
            try:
                result = run_tag_job(tagger_obj, path_obj, conn, config=config)
            except Exception:
                logger.exception("Tagging failed during refresh for %s", path_obj)
                continue
            if result is None:
                continue
            tagged += 1
            if batch_size > 0 and tagged % max(batch_size, 1) == 0:
                logger.info("Manual tag refresh tagged %d/%d file(s)", tagged, total)

        elapsed = time.perf_counter() - start_time
        stats_out["tagged"] = tagged
        stats_out["elapsed_sec"] = elapsed
        logger.info(
            "Manual tag refresh complete: tagged %d of %d file(s) in %.2fs (missing removed=%d, hard=%s)",
            tagged,
            total,
            elapsed,
            missing_count,
            hard_delete_missing,
        )
        return stats_out
    finally:
        conn.close()
        if tagger is not None:
            closer = getattr(tagger, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    logger.exception("Failed to close tagger after manual refresh")


__all__ = ["scan_and_tag"]
