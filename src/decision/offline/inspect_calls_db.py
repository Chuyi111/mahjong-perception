import argparse, sqlite3, gzip, json, random
from collections import Counter

CALL_TYPES = {2:"chi",3:"pon",4:"daiminkan",5:"shouminkan",6:"ankan"}

def parse_val(v):
    if v is None: return None
    if isinstance(v,(bytes,bytearray)):
        b = bytes(v)
        try:
            return json.loads(gzip.decompress(b))
        except Exception:
            try:
                return json.loads(b.decode("utf-8","ignore"))
            except Exception:
                return None
    if isinstance(v,str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return None

def find_json_col(conn, table):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    # try common names first
    for c in ("data","json","payload","blob","content"):
        if c in cols:
            try:
                r = conn.execute(f"SELECT {c} FROM {table} LIMIT 1").fetchone()
                if r and parse_val(r[0]) is not None:
                    return c
            except Exception:
                pass
    # fallback: probe all
    for c in cols:
        try:
            r = conn.execute(f"SELECT {c} FROM {table} LIMIT 1").fetchone()
            if r and parse_val(r[0]) is not None:
                return c
        except Exception:
            pass
    raise SystemExit(f"[ERR] no JSON column found in {table}")

def sample_rows(conn, table, json_col, n=2000):
    # prefer random-ish spread
    cnt = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if cnt == 0: return []
    ids = [i for i in range(1, cnt+1, max(1, cnt//n))][:n]
    rows = []
    for i in ids:
        r = conn.execute(f"SELECT {json_col} FROM {table} WHERE id=?", (i,)).fetchone()
        if not r: continue
        obj = parse_val(r[0])
        if obj: rows.append(obj)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--tables", default="Chi,Pon,DaiMinKan,ShouMinKan,AnKan,Skip,Discard")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?immutable=1", uri=True)
    conn.row_factory = sqlite3.Row

    for tbl in [t.strip() for t in args.tables.split(",") if t.strip()]:
        try:
            conn.execute(f"SELECT 1 FROM {tbl} LIMIT 1")
        except Exception:
            print(f"[WARN] table {tbl} not found")
            continue
        json_col = find_json_col(conn, tbl)
        rows = sample_rows(conn, tbl, json_col, n=args.n)

        has_calls = 0
        has_skip  = 0
        label_skip = 0
        size = 0
        type_counts = Counter()
        chosen_type_counts = Counter()

        for obj in rows:
            acts = obj.get("valid_actions") or []
            aidx = obj.get("action_idx", -1)
            if not acts or aidx < 0 or aidx >= len(acts): 
                continue
            size += 1
            types = [int(a.get("type",-1)) for a in acts]
            for t in types: type_counts[t]+=1

            # does this row offer any call candidates?
            if any(t in (2,3,4,5,6) for t in types):
                has_calls += 1
                # is skip (type 0) present among options?
                if 0 in types: has_skip += 1

            # what did expert choose?
            ct = int(acts[aidx].get("type",-1))
            chosen_type_counts[ct]+=1
            if ct == 0:
                label_skip += 1

        print(f"\n[TABLE {tbl}] rows={size}")
        print(f"  rows_with_any_call_option={has_calls}")
        print(f"  rows_with_call_and_skip_option={has_skip}")
        print(f"  chosen_type_counts={dict(chosen_type_counts)}")
        print(f"  option_type_counts (sample)={dict(type_counts)}")

    conn.close()

if __name__ == "__main__":
    main()
