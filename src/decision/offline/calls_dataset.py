from __future__ import annotations
import sqlite3, gzip, json, random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.decision.features import shanten_all  # you already have this
from .feature_builders import tiles_list_to_counts34, phys_to_34  # from your BC builder


# ----------------- SQLite helpers -----------------
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

def _parse_json_val(val: Any) -> Optional[dict]:
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        b = bytes(val)
        # try gzip
        try:
            dec = gzip.decompress(b)
            return json.loads(dec)
        except Exception:
            pass
        # try plain json
        try:
            return json.loads(b.decode("utf-8", "ignore"))
        except Exception:
            return None
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    return None


# ----------------- Option feature builder -----------------
CALL_TYPES = {2: "chi", 3: "pon", 4: "daiminkan", 5: "shouminkan", 6: "ankan"}
CALL_TYPE_KEYS = [2, 3, 4, 5, 6]  # order for one-hot

def _one_hot_calltype(t: int) -> np.ndarray:
    v = np.zeros(len(CALL_TYPE_KEYS) + 1, np.float32)  # +1 for "skip"
    if t == -1:  # skip
        v[0] = 1.0
        return v
    # map 2..6 to indices 1..5
    if t in CALL_TYPES:
        v[CALL_TYPE_KEYS.index(t) + 1] = 1.0
    return v

def _relative_who_onehot(src_wind: int, self_wind: int) -> np.ndarray:
    """
    relative seat of caller (source tile owner) w.r.t you:
    left=1, across=2, right=3. Use 4-length onehot with index 0 for 'n/a'.
    """
    v = np.zeros(4, np.float32)
    if src_wind < 0:
        v[0] = 1.0
        return v
    rel = (src_wind - self_wind) % 4
    if rel == 1: v[1] = 1.0  # left
    elif rel == 2: v[2] = 1.0  # across
    elif rel == 3: v[3] = 1.0  # right
    else:
        v[0] = 1.0  # your own (n/a)
    return v

def _apply_call_to_hand_counts(hand34: List[int], action: dict, player_wind: int) -> List[int]:
    """
    Remove tiles used from your hand to make the call.
    'action["tiles"]' are phys ids; 'action["who"]' mark owners per tile.
    We remove tiles for which who == player_wind (the ones you must provide).
    For 'ankan' (6), all four tiles come from your hand.
    """
    post = hand34[:]
    tiles = action.get("tiles", []) or []
    who = action.get("who", []) or []
    ttype = int(action.get("type", -1))
    for i, p in enumerate(tiles):
        if p < 0: 
            continue
        owner = who[i] if i < len(who) else -1
        t34 = phys_to_34(p)
        if t34 < 0 or t34 >= 34:
            continue
        # remove if it's yours, or for ankan remove all
        if ttype == 6 or owner == player_wind:
            if post[t34] > 0:
                post[t34] -= 1
    return post

def _option_features(sample: dict, action: Optional[dict]) -> Tuple[np.ndarray, dict]:
    """
    Build features for a single option:
      - action=None or type=-1 → SKIP (pre-call hand)
      - otherwise → POST-CALL hand (tiles removed accordingly)
    Returns:
      feat (float32, D): concatenated vector
      meta (dict): small info for debugging (type, rel_who, d_shanten)
    """
    # --- basics from sample ---
    player_wind = int(sample.get("player_wind", 0)) % 4

    hand_phys = sample["hand_tiles"]
    hand34 = tiles_list_to_counts34(hand_phys)

    # post-call hand for option (or pre if skip)
    if action is None:
        post_hand34 = hand34[:]
        call_type = -1
        src_who = -1
    else:
        post_hand34 = _apply_call_to_hand_counts(hand34, action, player_wind)
        call_type = int(action.get("type", -1))
        # pick a source who: first tile not yours
        who_list = action.get("who", []) or []
        src_who = next((w for w in who_list if w != player_wind and w >= 0), -1)

    # visible counts (same as BC features, quick rebuild)
    vis34 = hand34[:]
    for seat in ("0","1","2","3"):
        info = sample.get(seat, {}) or {}
        disc34 = tiles_list_to_counts34(info.get("discards", []) or [])
        for i in range(34): vis34[i] += disc34[i]
        melds = info.get("melds", []) or []
        for m in melds:
            for p in (m.get("tiles", []) or []):
                if p >= 0:
                    t = phys_to_34(p)
                    if 0 <= t < 34: vis34[t] += 1

    # dora onehot
    dora_inds = sample.get("dora_indicators", []) or []
    dora = np.zeros(34, np.float32)
    for p in dora_inds:
        if p < 0: continue
        t = phys_to_34(p)
        if 0 <= t < 34:
            # same as BC helper dora mapping (approx: next tile, wrap)
            if   0 <= t <= 8:  dora[(t - 0 + 1) % 9 + 0] = 1
            elif 9 <= t <= 17: dora[(t - 9 + 1) % 9 + 9] = 1
            elif 18<= t <= 26: dora[(t - 18 + 1)% 9 + 18] = 1
            else:              dora[27 + ((t - 27 + 1) % 7)] = 1

    round_wind = int(sample.get("round_wind", 0)) % 4
    num_honba = float(sample.get("num_honba", 0))
    num_riichi= float(sample.get("num_riichi", 0))
    remain_tiles = float(sample.get("remain_tiles", 70))

    riichi_flags = np.array([
        1.0 if (sample.get("0",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("1",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("2",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("3",{}).get("riichi", False)) else 0.0,
    ], dtype=np.float32)

    # shanten before/after
    sh_pre = shanten_all(hand34)
    sh_post = shanten_all(post_hand34)
    sh_vec = np.array([
        float(sh_pre["normal"]), float(sh_post["normal"]),
        float(sh_pre["chiitoi"]), float(sh_post["chiitoi"]),
        float(sh_pre["kokushi"]), float(sh_post["kokushi"]),
        float(sh_post["normal"] - sh_pre["normal"]),   # d_shanten normal
        float(sh_post["chiitoi"] - sh_pre["chiitoi"]),
        float(sh_post["kokushi"] - sh_pre["kokushi"]),
    ], dtype=np.float32)

    # winds one-hots
    round_oh = np.zeros(4, np.float32); round_oh[round_wind] = 1.0
    seat_oh  = np.zeros(4, np.float32); seat_oh[player_wind] = 1.0

    # option meta: call type + relative who
    call_oh = _one_hot_calltype(call_type)  # [skip, chi, pon, daiminkan, shouminkan, ankan]
    relwho_oh = _relative_who_onehot(src_who, player_wind)

    # pack features
    feat = np.concatenate([
        np.asarray(post_hand34, np.float32),  # IMPORTANT: post-call hand for the option
        np.asarray(vis34, np.float32),
        dora,
        round_oh, seat_oh,
        np.array([num_honba, num_riichi, remain_tiles/70.0], np.float32),
        sh_vec,
        riichi_flags,
        call_oh,
        relwho_oh,
    ], axis=0).astype(np.float32)

    meta = {
        "type": call_type,
        "src_who": src_who,
        "d_shanten_normal": float(sh_post["normal"] - sh_pre["normal"]),
    }
    return feat, meta


# ----------------- Dataset with grouped options -----------------
@dataclass(frozen=True)
class RowPtr:
    table: str
    id: int

class CallsDataset(Dataset):
    """
    Each __getitem__ returns a set of options for a single decision row:
      - X: (K, D) features for K options (skip + each call candidate)
      - y: int label in [0..K-1] indicating which option expert chose
    """
    def __init__(
        self,
        db_path: str,
        tables: Sequence[str] = ("Chi","Pon","DaiMinKan","ShouMinKan","AnKan","Skip"),
        sample_limit: Optional[int] = 200_000,
        sample_stride: int = 1,
        shuffle_index: bool = True,
        json_col: Optional[str] = None,
        id_col: str = "id",
        p_keep_call: float = 1.0,   # <— NEW: keep fraction of call-labeled rows
        seed: int = 42,
    ):
        super().__init__()
        self.db_path = db_path
        self.tables = list(tables)
        self.sample_limit = sample_limit
        self.sample_stride = max(1, int(sample_stride))
        self.shuffle_index = shuffle_index
        self.json_col = json_col
        self.id_col = id_col
        self.p_keep_call = float(p_keep_call)
        self._rng = random.Random(seed)

        self._conn: Optional[sqlite3.Connection] = None

# --- in CallsDataset.__init__ ---

        conn = _connect(self.db_path)
        ptrs: List[RowPtr] = []

        # autodetect json column once (per table) when needed
        def autodetect_json_col(table: str) -> str:
            cols = _columns(conn, table)
            candidates = [c for c in cols if c.lower() in ("data","json","payload","blob","content")]
            probe = candidates + cols
            for c in probe:
                try:
                    rows = conn.execute(f"SELECT {c} FROM {table} LIMIT 10;").fetchall()
                except Exception:
                    continue
                for r in rows:
                    if _parse_json_val(r[c]) is not None:
                        return c
            raise SystemExit(f"[ERR] cannot find JSON column in table {table}")

        # compute per-table quota so we don't starve 'Skip'
        tables_present = [t for t in self.tables if _table_exists(conn, t)]
        if not tables_present:
            conn.close()
            raise SystemExit("[ERR] no tables found")

        per_table_limit = None
        if self.sample_limit:
            per_table_limit = max(1, self.sample_limit // len(tables_present))

        detected = None
        total_added = 0
        for tbl in tables_present:
            cols = _columns(conn, tbl)
            if self.id_col not in cols and "rowid" in cols:
                self.id_col = "rowid"

            if self.json_col is None:
                self.json_col = detected or autodetect_json_col(tbl)
                detected = self.json_col

            # uniform sampling across the table (use stride)
            count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            if count == 0:
                print(f"[INFO] '{tbl}' empty; skip")
                continue

            want = count if per_table_limit is None else min(per_table_limit, count)
            stride = max(1, count // want)
            added = 0
            # take roughly uniformly spaced ids (not just the first N)
            for rid in range(1, count + 1, stride):
                ptrs.append(RowPtr(tbl, rid))
                added += 1
                total_added += 1
                if per_table_limit and added >= per_table_limit:
                    break

            print(f"[INFO] Indexed ~{added} rows from '{tbl}' (count={count}, stride={stride})")

        conn.close()
        if not ptrs:
            raise SystemExit("[ERR] No call/skip rows indexed")

        if shuffle_index:
            random.shuffle(ptrs)
        self._index = ptrs


    def __len__(self) -> int:
        return len(self._index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_conn"] = None
        return state

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _connect(self.db_path)
        return self._conn

    def _fetch_obj(self, table: str, rid: int) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(f"SELECT {self.json_col} AS j FROM {table} WHERE {self.id_col}=?;", (rid,)).fetchone()
        if row is None:
            return None
        return _parse_json_val(row["j"])

    def __getitem__(self, idx: int):
        ptr = self._index[idx]
        obj = self._fetch_obj(ptr.table, ptr.id)
        if obj is None:
            return None

        acts = obj.get("valid_actions", []) or []
        aidx = int(obj.get("action_idx", -1))
        if len(acts) == 0 or aidx < 0 or aidx >= len(acts):
            return None

        # require a real decision: skip present AND at least one call present
        types = [int(a.get("type", -1)) for a in acts]
        has_call = any(t in (2,3,4,5,6) for t in types)
        has_skip = 0 in types
        if not (has_call and has_skip):
            return None


        # build options: SKIP + all calls (2..6) present in valid_actions
        options: List[Tuple[Optional[dict], dict]] = []
        options.append((None, {"type": -1}))  # SKIP
        call_map: List[int] = [-1]  # maps option index -> original action index; 0 is skip
        for j, a in enumerate(acts):
            t = int(a.get("type", -1))
            if t in (2,3,4,5,6):
                options.append((a, {"type": t}))
                call_map.append(j)

        if len(options) <= 1:
            # No calls available => only skip, but these aren't interesting for training; drop
            return None

        # label is which option matches action_idx; if expert skipped (type 0), label=0
        chosen = acts[aidx]
        chosen_type = int(chosen.get("type", -1))
        if chosen_type in (2,3,4,5,6):
            # find which option index maps to this aidx
            try:
                label = call_map.index(aidx)
            except ValueError:
                # Inconsistent row; drop
                return None
        else:
            label = 0  # skip

        if label != 0 and self.p_keep_call < 1.0:
            if self._rng.random() > self.p_keep_call:
                return None

        # features per option
        feats: List[np.ndarray] = []
        for a, _meta in options:
            f, _ = _option_features(obj, a)
            feats.append(f)
        X = np.stack(feats, axis=0)  # (K, D)
        y = label
        return torch.from_numpy(X), torch.tensor(int(y), dtype=torch.long)
