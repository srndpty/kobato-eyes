from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from core.config import load_settings
from db.connection import get_conn

from .signature import current_tagger_sig

logger = logging.getLogger(__name__)


def retag_query(
    db_path: str | Path,
    where_sql: str,
    params: Sequence[object] | None,
) -> int:
    """WHERE 句で指定したレコードの tagger_sig / last_tagged_at をリセットして再タグ付け対象にする。"""
    predicate = where_sql.strip() or "1=1"
    arguments: tuple[object, ...] = tuple(params or ())
    conn = get_conn(db_path, allow_when_quiesced=True)
    try:
        sql = "UPDATE files AS f SET tagger_sig = NULL, last_tagged_at = NULL WHERE " + predicate
        conn.execute(sql, arguments)
        affected = conn.execute("SELECT changes()").fetchone()[0] or 0
        conn.commit()
        logger.info("Flagged %d file(s) for re-tagging (predicate=%s)", affected, predicate)
        return int(affected)
    finally:
        conn.close()


def retag_all(
    db_path: str | Path,
    *,
    force: bool = False,
    settings=None,
) -> int:
    """現在の tagger_sig と一致するものだけ(or 全件)を再タグ付け対象へ。"""
    effective_settings = settings or load_settings()
    signature = current_tagger_sig(effective_settings)
    if force:
        predicate = "1=1"
        params: tuple[object, ...] = ()
    else:
        predicate = "tagger_sig = ?"
        params = (signature,)
    return retag_query(db_path, predicate, params)


__all__ = ["retag_query", "retag_all"]
