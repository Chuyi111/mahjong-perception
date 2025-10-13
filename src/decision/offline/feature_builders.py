# src/decision/offline/feature_builders.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from src.decision.features import shanten_all

def phys_to_34(phys_id: int) -> int:
    if phys_id < 0: return -1
    return phys_id // 4

def dora_from_indicator_34(ind34: int) -> int:
    if 0 <= ind34 <= 8:  base=0;  return base + ((ind34 - base + 1) % 9)
    if 9 <= ind34 <= 17: base=9;  return base + ((ind34 - base + 1) % 9)
    if 18 <= ind34 <= 26:base=18; return base + ((ind34 - base + 1) % 9)
    return 27 + ((ind34 - 27 + 1) % 7)

def tiles_list_to_counts34(tiles_phys: List[int]) -> List[int]:
    c = [0]*34
    for p in tiles_phys:
        if p < 0: continue
        t = phys_to_34(p)
        if 0 <= t < 34: c[t]+=1
    return c

def add_counts_from_melds(counts: List[int], melds: List[Dict]):
    for m in (melds or []):
        for p in (m.get("tiles", []) or []):
            if p < 0: continue
            t = phys_to_34(p)
            if 0 <= t < 34: counts[t]+=1

def build_bc_features(sample: Dict):
    """
    Returns:
      feat: (D,) float32
      mask: (34,) float32
      label: int in [0,33] or -1 if not a discard sample
    """
    hand_phys = sample["hand_tiles"]
    hand34 = tiles_list_to_counts34(hand_phys)
    vis34 = hand34[:]
    for seat in ("0","1","2","3"):
        info = sample.get(seat, {}) or {}
        disc34 = tiles_list_to_counts34(info.get("discards", []) or [])
        for i in range(34): vis34[i] += disc34[i]
        add_counts_from_melds(vis34, info.get("melds", []) or [])

    dora_inds = sample.get("dora_indicators", []) or []
    dora_tiles = [dora_from_indicator_34(phys_to_34(p)) for p in dora_inds if p>=0]
    dora_onehot = np.zeros(34, np.float32)
    for d in dora_tiles: dora_onehot[d]=1.0

    round_wind = int(sample.get("round_wind", 0)) % 4
    player_wind = int(sample.get("player_wind", 0)) % 4
    num_honba = float(sample.get("num_honba", 0))
    num_riichi = float(sample.get("num_riichi", 0))
    remain_tiles = float(sample.get("remain_tiles", 70))

    riichi_flags = np.array([
        1.0 if (sample.get("0",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("1",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("2",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("3",{}).get("riichi", False)) else 0.0,
    ], dtype=np.float32)

    sh = shanten_all(hand34)  # {'normal','chiitoi','kokushi'}
    sh_norm, sh_chii, sh_koku = float(sh["normal"]), float(sh["chiitoi"]), float(sh["kokushi"])

    round_oh = np.zeros(4, np.float32); round_oh[round_wind]=1.0
    seat_oh = np.zeros(4, np.float32);  seat_oh[player_wind]=1.0

    feat = np.concatenate([
        np.asarray(hand34, np.float32),
        np.asarray(vis34,  np.float32),
        dora_onehot,
        round_oh, seat_oh,
        np.array([num_honba, num_riichi, remain_tiles/70.0, sh_norm, sh_chii, sh_koku], np.float32),
        riichi_flags
    ], axis=0)

    mask = np.zeros(34, np.float32)
    for i in range(34):
        if hand34[i] > 0: mask[i]=1.0

    # Label if chosen action is discard (type 1) or riichi (type 7)
    label = -1
    acts = sample.get("valid_actions", []) or []
    aidx = int(sample.get("action_idx", -1))
    if 0 <= aidx < len(acts):
        chosen = acts[aidx]
        t = int(chosen.get("type", -1))
        if t in (1,7):
            tile_phys = (chosen.get("tiles", [-1]) or [-1])[0]
            label = phys_to_34(tile_phys)

    return feat.astype(np.float32), mask.astype(np.float32), int(label)
