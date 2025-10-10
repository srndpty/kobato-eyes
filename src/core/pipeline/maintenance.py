from __future__ import annotations

import logging
import sqlite3
import time

logger = logging.getLogger(__name__)


def wait_for_unlock(db_path: str, timeout: float = 15.0) -> bool:
    """短命接続で軽く叩き、ロックが抜けるのを待つ（最大 timeout 秒）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with sqlite3.connect(db_path, timeout=1.0) as c:
                c.execute("SELECT 1")
                return True
        except sqlite3.OperationalError as e:
            if "locked" not in str(e).lower():
                return True
        time.sleep(0.25)
    return False


def _settle_after_quiesce(db_path: str, progress_cb=None) -> None:
    logger.info("_settle_after_quiesce (best-effort)")
    ok = wait_for_unlock(db_path, timeout=15.0)
    if not ok:
        logger.warning("settle: DB still locked; proceeding best-effort")
    try:
        with sqlite3.connect(db_path, timeout=30.0) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA busy_timeout=30000")
            try:
                cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError as e:
                logger.warning("settle: wal_checkpoint failed: %s", e)
            try:
                cur.execute("PRAGMA optimize")
            except sqlite3.OperationalError as e:
                logger.warning("settle: optimize failed: %s", e)
            conn.commit()
    except Exception as e:
        logger.warning("settle: sweep connection failed: %s", e)
    time.sleep(0.2)


__all__ = ["wait_for_unlock", "_settle_after_quiesce"]
