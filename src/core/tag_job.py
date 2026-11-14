"""Legacy single-image tagging adapter for kobato-eyes.

New indexing code should use :mod:`core.pipeline.stages` through
``core.pipeline.run_index_once``.  This module remains as a compact
compatibility API for tests and callers that still execute one image at a time.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from core.signature import ensure_signatures
from db.repository import replace_file_tags, update_fts, upsert_file, upsert_tags
from tagger.base import ITagger, MaxTagsMap, TagResult, ThresholdMap
from tagger.pixai_v1_identity import verify_loaded_identity
from utils.hash import compute_sha256
from utils.image_io import safe_load_image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TagJobConfig:
    """Runtime configuration for a tagging job."""

    thresholds: ThresholdMap | None = None
    max_tags: MaxTagsMap | None = None
    tagger_sig: str | None = None
    tagged_at: float | None = None
    # tagger_sig の計算に使ったモデル内容（v1 のみ）. 保持中 tagger との照合に使う.
    content_identity: Mapping[str, str] | None = None


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

    cfg = config or TagJobConfig()
    # モデル差し替え後の旧 tagger 結果を新署名で保存しないよう、推論前に照合する（全 tagger 共通）.
    verify_loaded_identity(tagger, cfg.content_identity, signed=cfg.tagger_sig is not None)
    image: Image.Image | None = None
    if getattr(tagger, "preserve_source_rgb", False):
        from tagger.pixai_v1_preprocessing import load_source_rgb

        # 画素上限をデコード前に効かせるため、原寸入力を最初に確定する. 署名用の縮小画像は後段で必要時のみ読む.
        try:
            rgb, (width, height) = load_source_rgb(source)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            logger.warning("Failed to load source image for tagging %s: %s", source, exc)
            return None
        results = tagger.infer_batch_prepared(
            tagger.prepare_batch_from_rgb_np([rgb]),
            thresholds=cfg.thresholds,
            max_tags=cfg.max_tags,
        )
        del rgb
    else:
        image = safe_load_image(source)
        if image is None:
            return None
        width, height = image.width, image.height
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
        width=width,
        height=height,
        tagger_sig=cfg.tagger_sig,
        last_tagged_at=tagged_at,
    )
    # 非 v1 は読み込み済み画像を再利用. v1 は未保存時のみ ensure_signatures が縮小読み込みする.
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
