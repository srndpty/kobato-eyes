# debug_db_open.py
import os
import pathlib
import sqlite3
import sys
import time

db = pathlib.Path(os.environ["APPDATA"]) / "kobato-eyes" / "kobato-eyes.db"


def p(msg):
    print(time.strftime("%H:%M:%S"), msg)
    sys.stdout.flush()


p(f"DB: {db}")
t = time.perf_counter()

p("connect() ...")
conn = sqlite3.connect(str(db), timeout=3.0)
p(f"connect ok ({time.perf_counter()-t:.2f}s)")

conn.row_factory = sqlite3.Row


def q(sql):
    t0 = time.perf_counter()
    try:
        r = conn.execute(sql).fetchall()
        p(f"{sql} -> ok ({time.perf_counter()-t0:.2f}s)")
        return r
    except Exception as e:
        p(f"{sql} -> ERROR: {e}")
        raise


# 基礎情報
q("PRAGMA journal_mode")
q("PRAGMA page_size")
q("PRAGMA user_version")
# WAL サイズ確認
wal = str(db) + "-wal"
p(f"WAL exists: {os.path.exists(wal)} size={os.path.getsize(wal) if os.path.exists(wal) else 0} bytes")

# “重くなりがち”な箇所を個別計測
p("BEGIN IMMEDIATE for smoke test ...")
t0 = time.perf_counter()
try:
    conn.execute("BEGIN IMMEDIATE")
    conn.commit()
    p(f"BEGIN/COMMIT ok ({time.perf_counter()-t0:.2f}s)")
except Exception as e:
    p(f"BEGIN/COMMIT ERROR: {e}")

# optimize が止まるケースの確認
t0 = time.perf_counter()
try:
    conn.execute("PRAGMA optimize")
    p(f"PRAGMA optimize ok ({time.perf_counter()-t0:.2f}s)")
except Exception as e:
    p(f"PRAGMA optimize ERROR: {e}")

# WAL を明示的に切るテスト（時間がかかるなら WAL が犯人）
t0 = time.perf_counter()
try:
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    p(f"wal_checkpoint(TRUNCATE) ok ({time.perf_counter()-t0:.2f}s)")
except Exception as e:
    p(f"wal_checkpoint(TRUNCATE) ERROR: {e}")

conn.close()
p("done.")
