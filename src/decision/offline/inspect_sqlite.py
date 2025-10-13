from __future__ import annotations
import sqlite3, json, sys
from pathlib import Path

def connect(db_path: str):
    uri = f"file:{db_path}?immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def main(db_path: str):
    p = Path(db_path)
    if not p.exists():
        print(f"[ERR] DB not found: {p}"); return

    conn = connect(db_path)
    cur = conn.cursor()

    print("== Tables ==")
    tables = cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()
    for t in tables:
        print(f"- {t['name']}")
    if not tables:
        print("[WARN] No tables found (is this a valid SQLite file?)"); return

    print("\n== Columns per table ==")
    for t in tables:
        name = t["name"]
        cols = conn.execute(f"PRAGMA table_info({name});").fetchall()
        col_str = ", ".join([f"{c['name']}:{c['type']}" for c in cols])
        print(f"- {name}: {col_str}")

    print("\n== Row samples (first 2 rows per table) ==")
    for t in tables:
        name = t["name"]
        try:
            rows = cur.execute(f"SELECT rowid, * FROM {name} LIMIT 2;").fetchall()
        except Exception as e:
            print(f"- {name}: cannot SELECT * ( {e} )")
            continue
        print(f"- {name}:")
        for r in rows:
            d = {k: r[k] for k in r.keys()}
            # Try to detect a JSON-like column
            for k,v in d.items():
                if isinstance(v, (bytes, bytearray)):
                    try:
                        txt = v.decode("utf-8", "ignore")
                        d[k] = (txt[:200] + "...") if len(txt) > 200 else txt
                    except Exception:
                        d[k] = f"<{len(v)} bytes>"
                elif isinstance(v, str) and len(v) > 200:
                    d[k] = v[:200] + "..."
            print(f"  rowid={r['rowid']} keys={list(d.keys())}")
            print(f"  sample: {d}")
    conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.decision.offline.inspect_sqlite <path/to/db.sqlite>")
        sys.exit(1)
    main(sys.argv[1])
