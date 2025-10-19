"""Tagging job orchestration for kobato-eyes."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from core.signature import ensure_signatures
from db.repository import replace_file_tags, update_fts, upsert_file, upsert_tags
from tagger.base import ITagger, MaxTagsMap, TagResult, ThresholdMap
from utils.hash import compute_sha256
from utils.image_io import safe_load_image


@dataclass(frozen=True)
class TagJobConfig:
    """Runtime configuration for a tagging job."""

    thresholds: ThresholdMap | None = None
    max_tags: MaxTagsMap | None = None
    tagger_sig: str | None = None
    tagged_at: float | None = None


@dataclass(frozen=True)
class TagJobOutput:
    """Result of processing a single image through the tagging pipeline."""

    file_id: int
    tag_result: TagResult


def run_tag_job(
    tagger: ITagger,
    image_path: str | Path,
    conn: sqlite3.Connection,
    *,
    config: TagJobConfig | None = None,
) -> TagJobOutput | None:
    """Execute a full tagging pass for ``image_path`` and persist into the database."""
    source = Path(image_path)
    if not source.exists() or not source.is_file():
        return None

    image = safe_load_image(source)
    if image is None:
        return None

    cfg = config or TagJobConfig()
    results = tagger.infer_batch(
        [image],
        thresholds=cfg.thresholds,
        max_tags=cfg.max_tags,
    )
    if not results:
        return None
    tag_result = results[0]

    stat = source.stat()
    sha256_hex = compute_sha256(source)
    tagged_at = cfg.tagged_at if cfg.tagged_at is not None else time.time()
    file_id = upsert_file(
        conn,
        path=str(source),
        size=stat.st_size,
        mtime=stat.st_mtime,
        sha256=sha256_hex,
        width=image.width,
        height=image.height,
        tagger_sig=cfg.tagger_sig,
        last_tagged_at=tagged_at,
    )
    # 画像は既に読んでいるので I/O 追加ゼロで署名保存できる
    ensure_signatures(conn, file_id, image=image, path=str(source))

    tag_defs = [{"name": prediction.name, "category": int(prediction.category)} for prediction in tag_result.tags]
    tag_id_map = upsert_tags(conn, tag_defs)

    tag_scores = [
        (tag_id_map[prediction.name], prediction.score)
        for prediction in tag_result.tags
        if prediction.name in tag_id_map
    ]
    replace_file_tags(conn, file_id, tag_scores)

    fts_text = " ".join(prediction.name for prediction in tag_result.tags) if tag_scores else None
    update_fts(conn, file_id, fts_text)

    return TagJobOutput(file_id=file_id, tag_result=tag_result)


__all__ = ["TagJobConfig", "TagJobOutput", "run_tag_job"]
