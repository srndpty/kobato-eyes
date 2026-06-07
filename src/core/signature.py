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


def _to_signed64(x: int) -> int:
    # SQLite の INTEGER は符号付き64bit。符号なし(>=2^63)のまま渡すと
    # OverflowError になり upsert が無言で失敗するため、必ず符号付きに丸める。
    v = int(x) & ((1 << 64) - 1)
    return v - (1 << 64) if v >= (1 << 63) else v


def compute_signatures_from_image(im: Image.Image) -> tuple[int, int]:
    # 例外は呼び出し側で握る
    p = phash(im)
    d = dhash(im)
    return (_to_signed64(p), _to_signed64(d))


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
        log.warning("ensure_signatures failed for %s: %s", path or f"file_id={file_id}", e)
        return False
