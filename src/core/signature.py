# src/core/signature.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

from db.repository import upsert_signatures
from sig.phash import dhash, phash
from utils.image_io import safe_load_image

log = logging.getLogger(__name__)


def _u64(x: int) -> int:
    return int(x) & ((1 << 64) - 1)


def compute_signatures_from_image(im: Image.Image) -> tuple[int, int]:
    # 例外は呼び出し側で握る
    p = phash(im)
    d = dhash(im)
    return (_u64(p), _u64(d))


def ensure_signatures(
    conn,
    file_id: int,
    *,
    image: Optional[Image.Image] = None,
    path: Optional[str | Path] = None,
    force: bool = False,
) -> bool:
    """
    署名が未保存なら計算して upsert。戻り値: True=保存済み/維持、False=失敗。
    force=True なら既存あっても再計算（普段は False 推奨）。
    """
    try:
        if not force:
            row = conn.execute("SELECT 1 FROM signatures WHERE file_id=? LIMIT 1", (file_id,)).fetchone()
            if row is not None:
                return True

        if image is None:
            if path is None:
                return False
            im = safe_load_image(Path(path))
            if im is None:
                return False
            image = im

        p, d = compute_signatures_from_image(image)
        upsert_signatures(conn, file_id=file_id, phash_u64=p, dhash_u64=d)
        return True
    except Exception as e:
        log.debug("ensure_signatures failed for %s: %s", path or f"file_id={file_id}", e)
        return False
