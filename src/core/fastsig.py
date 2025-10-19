# src/core/fastsig.py
from __future__ import annotations

import multiprocessing as mp
import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from PIL import Image

from sig.phash import dhash, phash  # ← phashはcv2必須

U64MASK = (1 << 64) - 1


def _to_signed64(x: int) -> int:
    v = int(x) & U64MASK
    return v - (1 << 64) if v >= (1 << 63) else v


def _compute_worker(task: Tuple[int, str]) -> Tuple[int, int, int] | None:
    """サブプロセスで (file_id, path) → (file_id, phash, dhash)"""
    fid, p = task
    try:
        path = Path(p)
        if not path.exists() or not path.is_file():
            return None
        with Image.open(path) as im:
            # phash/dhash の結果を“必ず”符号付き64bitに丸める
            ph = _to_signed64(phash(im))
            dh = _to_signed64(dhash(im))
        return (int(fid), ph, dh)
    except Exception:
        return None  # 失敗は捨てる（速度最優先）


def _fast_pragmas(conn: sqlite3.Connection) -> None:
    """耐久性を犠牲に速度を最優先するPRAGMA（unsafe_fast）。"""
    conn.execute("PRAGMA journal_mode=WAL")  # 競合減＆速いことが多い
    conn.execute("PRAGMA synchronous=OFF")  # 電源断に弱いが爆速
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=30000000000")  # 環境に応じて調整


def bulk_upsert_signatures(conn: sqlite3.Connection, rows: Iterable[Tuple[int, int, int]]) -> int:
    """(file_id, phash, dhash) を executemany でまとめて upsert。"""
    sql = """
        INSERT INTO signatures (file_id, phash_u64, dhash_u64)
        VALUES (?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            phash_u64 = excluded.phash_u64,
            dhash_u64 = excluded.dhash_u64
    """
    rows = [(int(fid), _to_signed64(ph), _to_signed64(dh)) for (fid, ph, dh) in rows]
    if not rows:
        return 0
    with conn:
        cur = conn.executemany(sql, rows)
    return cur.rowcount or 0


def compute_signatures_mp(
    tasks: List[Tuple[int, str]],
    *,
    max_workers: Optional[int] = None,
    chunksize: int = 64,
    progress: Optional[callable] = None,  # progress(done:int, total:int)
) -> List[Tuple[int, int, int]]:
    """(file_id, path) を並列で処理して (file_id, ph, dh) を返す。"""
    if not tasks:
        return []
    total = len(tasks)
    done = 0
    results: List[Tuple[int, int, int]] = []

    # Windows/macOS を考慮して spawn を明示
    ctx = mp.get_context("spawn")
    workers = max_workers or max(1, (os.cpu_count() or 4) - 1)  # HDDなら4〜8, SSDなら8〜16が目安
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        # mapは戻り順が固定されて速い。進捗を出したい場合はas_completedでもOK
        for out in ex.map(_compute_worker, tasks, chunksize=chunksize):
            done += 1
            if out is not None:
                results.append(out)
            if progress and (done % 200 == 0 or done == total):
                try:
                    progress(done, total)
                except Exception:
                    pass
    return results


def fast_fill_missing_signatures(
    db_path: str,
    items: List[Tuple[int, str]],
    *,
    max_workers: Optional[int] = None,
    chunksize: int = 64,
    progress: Optional[callable] = None,
    apply_to_db: bool = True,
    unsafe_fast: bool = True,
) -> List[Tuple[int, int, int]]:
    """未計算の署名を並列計算→まとめてDB反映（戻り値: 成功した行の( fid, ph, dh )）。"""
    computed = compute_signatures_mp(items, max_workers=max_workers, chunksize=chunksize, progress=progress)
    if apply_to_db and computed:
        with sqlite3.connect(db_path) as conn:
            if unsafe_fast:
                _fast_pragmas(conn)
            bulk_upsert_signatures(conn, computed)
    return computed
