from __future__ import annotations
import json, random, math
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import numpy as np

# --- import your shanten utilities for strong features ---
from decision.features import shanten_all
# We'll use your 34-tile convention from decision/state.py
# Mapping note: your logs use "physical tile ids" (0..135) with 4 copies per tile type.
# We map to 34-id via t34 = phys_id // 4

# --------- helpers: tile mapping & dora ----------
def phys_to_34(phys_id: int) -> int:
    if phys_id < 0:  # -1 padding in logs
        return -1
    return phys_id // 4

def dora_from_indicator_34(ind34: int) -> int:
    # 0..8 manzu, 9..17 pinzu, 18..26 souzu, 27..33 honors
    if 0 <= ind34 <= 8:
        base = 0
        return base + ((ind34 - base + 1) % 9)
    if 9 <= ind34 <= 17:
        base = 9
        return base + ((ind34 - base + 1) % 9)
    if 18 <= ind34 <= 26:
        base = 18
        return base + ((ind34 - base + 1) % 9)
    # honors wrap within 27..33
    return 27 + ((ind34 - 27 + 1) % 7)

def tiles_list_to_counts34(tiles_phys: List[int]) -> List[int]:
    c = [0]*34
    for p in tiles_phys:
        if p < 0: continue
        tid = phys_to_34(p)
        if 0 <= tid < 34:
            c[tid] += 1
    return c

def add_counts_from_melds(counts: List[int], melds: List[Dict]):
    # logs say "melds: same format as valid_actions"
    # we only need tiles[]. Add 0..3 valid tile phys ids.
    for m in melds:
        arr = m.get("tiles", [])
        for p in arr:
            if p < 0: continue
            tid = phys_to_34(p)
            if 0 <= tid < 34:
                counts[tid] += 1

# --------- features ----------
def build_features(sample: Dict) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
      feat: (D,) float32 feature vector
      mask: (34,) float32 1/0 for legal discards (by 34-id)
      label: int in [0,33] chosen discard (if sample is a discard), else -1 to skip
    """
    # Basic visibles
    hand_phys: List[int] = sample["hand_tiles"]
    hand34 = tiles_list_to_counts34(hand_phys)
    # Public info per player "0"/"1"/"2"/"3"
    vis34 = hand34[:]  # start with hand
    for seat in ("0","1","2","3"):
        info = sample.get(seat, {})
        # discards
        disc_phys = info.get("discards", []) or []
        disc34 = tiles_list_to_counts34(disc_phys)
        for i in range(34): vis34[i] += disc34[i]
        # melds
        melds = info.get("melds", []) or []
        add_counts_from_melds(vis34, melds)

    # Dora indicators (phys) -> dora tiles (34)
    dora_inds: List[int] = sample.get("dora_indicators", []) or []
    dora_tiles = [dora_from_indicator_34(phys_to_34(p)) for p in dora_inds if p >= 0]
    dora_onehot = np.zeros(34, dtype=np.float32)
    for d in dora_tiles:
        dora_onehot[d] = 1.0

    # Winds / counters
    round_wind = sample.get("round_wind", 0)  # 0..?
    num_honba = sample.get("num_honba", 0)
    num_riichi = sample.get("num_riichi", 0)
    player_wind = sample.get("player_wind", 0)  # 0:E,1:S,2:W,3:N per spec
    remain_tiles = sample.get("remain_tiles", 70)

    # Riichi statuses
    riichi_flags = []
    for seat in ("0","1","2","3"):
        riichi_flags.append(1.0 if (sample.get(seat,{}).get("riichi", False)) else 0.0)
    riichi_flags = np.array(riichi_flags, dtype=np.float32)

    # Shanten features (normal/chiitoi/kokushi)
    sh_dict = shanten_all(hand34)
    sh_norm = float(sh_dict["normal"])
    sh_chii = float(sh_dict["chiitoi"])
    sh_koku = float(sh_dict["kokushi"])

    # Compose features:
    #   [ hand34 (34), visible34 (34), dora_onehot (34),
    #     winds one-hot (round 4, seat 4), honba, riichi_sticks,
    #     remain_tiles_norm, shanten(3), riichi_flags(4) ]
    f = []
    f += hand34
    f += vis34
    f += dora_onehot.tolist()

    round_oh = np.zeros(4, dtype=np.float32)
    round_oh[min(3, round_wind % 4)] = 1.0
    seat_oh = np.zeros(4, dtype=np.float32)
    seat_oh[min(3, player_wind % 4)] = 1.0
    f += round_oh.tolist()
    f += seat_oh.tolist()

    f.append(float(num_honba))
    f.append(float(num_riichi))
    f.append(float(remain_tiles) / 70.0)  # normalized

    f += [sh_norm, sh_chii, sh_koku]
    f += riichi_flags.tolist()

    feat = np.asarray(f, dtype=np.float32)

    # Mask of legal discards (by 34-id)
    mask = np.zeros(34, dtype=np.float32)
    for i in range(34):
        if hand34[i] > 0:
            mask[i] = 1.0

    # Label (chosen discard tile 34-id) — we ONLY keep samples where chosen action is discard or riichi-discard
    acts = sample.get("valid_actions", [])
    aidx = sample.get("action_idx", -1)
    label = -1
    if 0 <= aidx < len(acts):
        chosen = acts[aidx]
        t = chosen.get("type", -1)
        if t in (1, 7):  # 1: discard, 7: riichi (tile to discard)
            tile_phys = chosen.get("tiles", [ -1 ])[0]
            label = phys_to_34(tile_phys)

    return feat, mask, label

def main():
    ap = argparse.ArgumentParser("Build BC dataset from Mahjong logs")
    ap.add_argument("--logs", required=True, help="Folder with *.json snapshots (recursively)")
    ap.add_argument("--out", required=True, help="Output npz path (e.g., data_bc/dataset_bc.npz)")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    root = Path(args.logs)
    files = sorted([p for p in root.rglob("*.json") if p.is_file()])
    X, M, Y = [], [], []

    total, kept = 0, 0
    for fp in files:
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] bad json: {fp} ({e})"); continue

        feat, mask, label = build_features(obj)
        total += 1
        if label < 0 or mask[label] < 0.5:
            # skip non-discard snapshots or corrupted ones
            continue
        X.append(feat); M.append(mask); Y.append(label); kept += 1

    if kept == 0:
        raise SystemExit("[ERR] No usable samples (no discard/riichi decisions found).")

    X = np.stack(X, axis=0)
    M = np.stack(M, axis=0)
    Y = np.asarray(Y, dtype=np.int64)

    # train/val split
    N = len(Y)
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1.0 - args.val_ratio))
    tr_idx, va_idx = idx[:split], idx[split:]

    np.savez_compressed(args.out,
                        X_train=X[tr_idx], mask_train=M[tr_idx], y_train=Y[tr_idx],
                        X_val=X[va_idx], mask_val=M[va_idx], y_val=Y[va_idx])
    print(f"[OK] Saved {args.out}  | total json={total}  kept={kept}  train={len(tr_idx)} val={len(va_idx)}")

if __name__ == "__main__":
    main()
