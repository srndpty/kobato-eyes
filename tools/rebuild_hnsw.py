# tools/rebuild_hnsw.py
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import hnswlib
import numpy as np

from db.connection import get_conn
from utils.paths import ensure_dirs


def iter_embeddings(conn: sqlite3.Connection, model: str, batch: int = 1000):
    off = 0
    while True:
        rows = conn.execute(
            "SELECT file_id, dim, vector FROM embeddings WHERE model=? LIMIT ? OFFSET ?",
            (model, batch, off),
        ).fetchall()
        if not rows:
            break
        yield rows
        off += len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--space", default="cosine")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--reserve", type=int, default=2048)
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")

    db = Path(args.db).expanduser()
    index_dir = Path(args.index_dir).expanduser()
    ensure_dirs()
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "hnsw_cosine.bin"

    conn = get_conn(db)
    try:
        first = conn.execute("SELECT dim FROM embeddings WHERE model=? LIMIT 1", (args.model,)).fetchone()
        if not first:
            print("No embeddings for model:", args.model)
            return 0
        dim = int(first["dim"])
        idx = hnswlib.Index(space=args.space, dim=dim)

        # だいたいの容量を見積もる（正確でなくてOK）
        total = conn.execute("SELECT COUNT(*) AS c FROM embeddings WHERE model=?", (args.model,)).fetchone()["c"]
        max_elements = max(total + 256, args.reserve)
        idx.init_index(max_elements=max_elements, ef_construction=args.efc, M=args.m)

        added = 0
        for rows in iter_embeddings(conn, args.model, batch=2000):
            vecs = []
            labels = []
            for r in rows:
                fid = int(r["file_id"])
                dim_row = int(r["dim"])
                if dim_row != dim:
                    continue
                v = np.frombuffer(r["vector"], dtype=np.float32)
                if v.ndim != 1 or v.shape[0] != dim:
                    continue
                # 念のため正規化
                n = np.linalg.norm(v)
                if n > 0:
                    v = (v / n).astype(np.float32, copy=False)
                vecs.append(v)
                labels.append(fid)
            if not vecs:
                continue
            mat = np.vstack(vecs).astype(np.float32, copy=False)
            lab = np.ascontiguousarray(labels, dtype=np.int64)
            # 容量拡張（余裕を見ながら）
            cur = idx.get_current_count()
            need = cur + mat.shape[0] + 256
            if need > max_elements:
                idx.resize_index(need)
                max_elements = need
            idx.add_items(mat, lab, num_threads=1)
            added += mat.shape[0]
            print(f"added={added}/{total}", flush=True)

        tmp = index_path.with_suffix(".bin.tmp")
        idx.save_index(str(tmp))
        os.replace(tmp, index_path)

        meta = index_path.with_suffix(index_path.suffix + ".meta.json")
        meta.write_text(
            f'{{"space":"{args.space}","dim":{dim},"max_elements":{max_elements}}}',
            encoding="utf-8",
        )
        print("done:", index_path, flush=True)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
