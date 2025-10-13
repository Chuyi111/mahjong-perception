# src/decision/offline/index_sqlite_logs.py
from __future__ import annotations
import sqlite3, json, argparse
from pathlib import Path
from typing import Optional, List
import numpy as np

def connect(db_path: str):
    uri = f"file:{db_path}?immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    return conn

def pick_table(conn: sqlite3.Connection, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()
    names = [r["name"] for r in rows]
    if not names:
        raise SystemExit("[ERR] No tables in DB.")
    if len(names) == 1:
        return names[0]
    raise SystemExit(f"[ERR] Multiple tables found: {names}. Use --table <name>.")

def autodetect_json_col(conn: sqlite3.Connection, table: str, probe_rows: int = 50) -> str:
    cols = conn.execute(f"PRAGMA table_info({table});").fetchall()
    text_cols = [c["name"] for c in cols if "CHAR" in (c["type"] or "").upper() or "TEXT" in (c["type"] or "").upper()]
    if not text_cols:
        # fall back: try all columns
        text_cols = [c["name"] for c in cols]
    # Probe a few rows
    q = f"SELECT rowid, * FROM {table} LIMIT {probe_rows};"
    rows = conn.execute(q).fetchall()
    for c in text_cols:
        ok = 0
        for r in rows:
            v = r[c]
            if v is None:
                continue
            if isinstance(v, (bytes, bytearray)):
                try:
                    v = v.decode("utf-8", "ignore")
                except Exception:
                    continue
            if not isinstance(v, str):
                continue
            try:
                obj = json.loads(v)
                if isinstance(obj, dict) and "valid_actions" in obj:
                    ok += 1
            except Exception:
                continue
        if ok > 0:
            return c
    raise SystemExit(f"[ERR] Could not find a JSON column in {table} with key 'valid_actions'. Columns: {[c['name'] for c in cols]}")

def main():
    ap = argparse.ArgumentParser("Index discard/riichi rows (rowid) from SQLite logs")
    ap.add_argument("--db", required=True, help="Path to .sqlite")
    ap.add_argument("--table", default=None, help="Table name; if omitted and only one table exists, it will be used")
    ap.add_argument("--json_col", default=None, help="JSON column; auto-detected if omitted")
    ap.add_argument("--max_keep", type=int, default=200_000)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True, help="Output .npy path for rowids")
    args = ap.parse_args()

    conn = connect(args.db)

    table = pick_table(conn, args.table)
    if args.json_col:
        json_col = args.json_col
    else:
        json_col = autodetect_json_col(conn, table)
    print(f"[INFO] Using table='{table}', json_col='{json_col}'")

    total = conn.execute(f"SELECT COUNT(*) AS n FROM {table};").fetchone()["n"]
    print(f"[INFO] total rows: {total}")

    kept: List[int] = []
    i = 0
    for row in conn.execute(f"SELECT rowid, {json_col} FROM {table};"):
        i += 1
        if args.stride > 1 and (i % args.stride != 0):
            continue
        blob = row[json_col]
        if blob is None:
            continue
        if isinstance(blob, (bytes, bytearray)):
            try:
                blob = blob.decode("utf-8", "ignore")
            except Exception:
                continue
        if not isinstance(blob, str):
            continue
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        acts = obj.get("valid_actions", []) or []
        aidx = obj.get("action_idx", -1)
        if 0 <= aidx < len(acts):
            t = int(acts[aidx].get("type", -1))
            if t in (1, 7):  # discard or riichi-discard
                kept.append(int(row["rowid"]))
        if len(kept) and len(kept) % 100000 == 0:
            print(f"[PROG] kept {len(kept)}")
        if len(kept) >= args.max_keep:
            break

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    np.save(outp, np.asarray(kept, dtype=np.int64))
    print(f"[DONE] saved {len(kept)} rowids -> {outp}")

if __name__ == "__main__":
    main()
