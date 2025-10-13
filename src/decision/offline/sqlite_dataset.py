from __future__ import annotations
import sqlite3, gzip, json, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_builders import build_bc_features

# ----------------- helpers -----------------
def _connect(db_path: str) -> sqlite3.Connection:
    uri = f"file:{db_path}?immutable=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    return conn

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    r = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (name,)).fetchone()
    return r is not None

def _columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cols = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return [c["name"] for c in cols]

def _try_parse_json(val: Any) -> Optional[dict]:
    if val is None:
        return None
    # bytes → try gzip → json; then try utf-8 → json
    if isinstance(val, (bytes, bytearray)):
        b = bytes(val)
        # gzip?
        try:
            dec = gzip.decompress(b)
            try:
                return json.loads(dec)
            except Exception:
                pass
        except Exception:
            pass
        # not gzip; maybe plain JSON in bytes
        try:
            txt = b.decode("utf-8", "ignore")
            return json.loads(txt)
        except Exception:
            return None
    # str → json
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    return None

def _autodetect_json_col(conn: sqlite3.Connection, table: str, probe_rows: int = 50) -> str:
    cols = _columns(conn, table)
    # try common names first
    candidates = [c for c in cols if c.lower() in ("data","json","payload","blob","content")]
    for col in (candidates + cols):
        try:
            rows = conn.execute(f"SELECT {col} FROM {table} LIMIT {probe_rows};").fetchall()
        except Exception:
            continue
        ok = 0
        for r in rows:
            obj = _try_parse_json(r[col])
            if isinstance(obj, dict) and "valid_actions" in obj and "action_idx" in obj:
                ok += 1
        if ok > 0:
            return col
    raise SystemExit(f"[ERR] Could not auto-detect JSON column in table '{table}'. Columns: {cols}")

# We store (table, id) pairs
@dataclass(frozen=True)
class RowPtr:
    table: str
    id: int

# ----------------- dataset -----------------
class MahjongSQLiteDataset(Dataset):
    """
    Reads rows from SQLite tables (e.g., 'Discard', 'Riichi'), where a column holds
    gzipped or plain JSON snapshots. Auto-detects that JSON column if not provided.
    Converts to (feat, mask, label) for BC.
    """
    def __init__(
        self,
        db_path: str,
        tables: Sequence[str] = ("Discard",),
        sample_limit: Optional[int] = 200_000,
        sample_stride: int = 1,
        shuffle_index: bool = True,
        json_col: Optional[str] = None,       # if you know it, pass it; else we auto-detect
        id_col: str = "ID",                   # primary key column
    ):
        super().__init__()
        self.db_path = str(db_path)
        self.tables = list(tables)
        self.sample_limit = sample_limit
        self.sample_stride = max(1, int(sample_stride))
        self.shuffle_index = shuffle_index
        self.json_col = json_col
        self.id_col = id_col

        self._conn: Optional[sqlite3.Connection] = None

        # Build compact index of (table,id) & determine the JSON column if needed
        conn = _connect(self.db_path)
        ptrs: List[RowPtr] = []
        detected_json_col: Optional[str] = None

        for tbl in self.tables:
            if not _table_exists(conn, tbl):
                print(f"[WARN] table '{tbl}' not found, skipping.")
                continue
            cols = _columns(conn, tbl)
            if self.id_col not in cols:
                # try another common PK name
                if "rowid" in cols:
                    self.id_col = "rowid"
                else:
                    raise SystemExit(f"[ERR] PK column '{self.id_col}' not in {tbl} columns={cols}")

            col_for_json = self.json_col or detected_json_col
            if col_for_json is None:
                col_for_json = _autodetect_json_col(conn, tbl)
                detected_json_col = col_for_json  # reuse for other tables if structure matches
                print(f"[INFO] table '{tbl}' JSON column auto-detected as '{col_for_json}'")
            self.json_col = col_for_json  # store for fetching

            # index ids with stride
            cur = conn.execute(f"SELECT {self.id_col} AS id FROM {tbl} ORDER BY {self.id_col};")
            i = 0
            added = 0
            for (rid,) in cur:
                i += 1
                if self.sample_stride > 1 and (i % self.sample_stride != 0):
                    continue
                ptrs.append(RowPtr(tbl, int(rid)))
                added += 1
                if self.sample_limit and len(ptrs) >= self.sample_limit:
                    break
            print(f"[INFO] Indexed {added} ids from '{tbl}' (stride={self.sample_stride})")
            if self.sample_limit and len(ptrs) >= self.sample_limit:
                break

        conn.close()

        if not ptrs:
            raise SystemExit("[ERR] No rows indexed; check table names and schema.")

        if shuffle_index:
            random.shuffle(ptrs)

        self._index: List[RowPtr] = ptrs

    def __len__(self) -> int:
        return len(self._index)
    
    def __getstate__(self):
        """
        Ensure sqlite3.Connection is not pickled. Workers will lazily recreate
        their own connections via _get_conn().
        """
        state = self.__dict__.copy()
        state["_conn"] = None
        return state
    
    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _connect(self.db_path)
        return self._conn

    def _fetch_value(self, table: str, rid: int):
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT {self.json_col} AS j FROM {table} WHERE {self.id_col}=?;",
            (rid,)
        ).fetchone()
        if row is None:
            return None
        return row["j"]

    def __getitem__(self, idx: int):
        ptr = self._index[idx]
        val = self._fetch_value(ptr.table, ptr.id)
        if val is None:
            return None
        obj = _try_parse_json(val)
        if obj is None:
            return None

        feat, mask, label = build_bc_features(obj)
        if label < 0 or mask[label] < 0.5:
            return None

        X = torch.from_numpy(feat)
        M = torch.from_numpy(mask)
        y = torch.tensor(int(label)).long()
        return X, M, y
