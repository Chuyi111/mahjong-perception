from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

from .state import TileId, is_honor

# ---------------- Shanten (normal / chiitoi / kokushi) ---------------- #

def shanten_chiitoi(counts: List[int]) -> int:
    pairs = sum(1 for x in counts if x >= 2)
    uniques = sum(1 for x in counts if x > 0)
    need_pairs = max(0, 7 - pairs)
    need_uniques = max(0, 7 - uniques)
    return need_pairs + need_uniques - 1  # -1 per standard definition

def shanten_kokushi(counts: List[int]) -> int:
    terminals = {0,8,9,17,18,26,27,28,29,30,31,32,33}
    uniques = sum(1 for i in terminals if counts[i] > 0)
    has_pair = any(counts[i] >= 2 for i in terminals)
    return 13 - uniques - (1 if has_pair else 0)

# -------- Normal shanten: DFS meld search + greedy taatsu on remainder -------- #

def _greedy_taatsu(counts: List[int]) -> Tuple[int,int]:
    """Return (mentsu_completed, taatsu_pairs_runs) found greedily in residual counts (suits only)."""
    c = counts[:]  # work copy
    mentsu = 0
    # remove triplets
    for i in range(34):
        while c[i] >= 3:
            mentsu += 1
            c[i] -= 3
    taatsu = 0
    # suits: try sequences of length 2 (taatsu) and isolated pairs
    for base in (0, 9, 18):
        for i in range(base, base+9):
            # sequences taatsu: (i,i+1) and (i,i+2)
            if i % 9 <= 7:
                k = min(c[i], c[i+1])
                take = min(k, 1)
                taatsu += take
                c[i] -= take; c[i+1] -= take
            if i % 9 <= 6:
                k = min(c[i], c[i+2])
                take = min(k, 1)
                taatsu += take
                c[i] -= take; c[i+2] -= take
    # leftover pairs (including honors)
    for i in range(34):
        if c[i] >= 2:
            taatsu += 1
            c[i] -= 2
    return mentsu, taatsu

@lru_cache(maxsize=20000)
def _dfs_min_shanten(state_key: Tuple[int,...], pair_used: bool) -> int:
    """Search minimal normal shanten; state_key is counts tuple of length 34."""
    counts = list(state_key)
    best = 8  # upper bound
    # try removing any meld (triplet/sequence)
    changed = False
    # triplets
    for i in range(34):
        if counts[i] >= 3:
            changed = True
            counts[i] -= 3
            best = min(best, _dfs_min_shanten(tuple(counts), pair_used))
            counts[i] += 3
    # sequences
    for base in (0, 9, 18):
        for i in range(base, base+7):  # i..i+2
            a,b,c = counts[i], counts[i+1], counts[i+2]
            if a>0 and b>0 and c>0:
                changed = True
                counts[i]-=1; counts[i+1]-=1; counts[i+2]-=1
                best = min(best, _dfs_min_shanten(tuple(counts), pair_used))
                counts[i]+=1; counts[i+1]+=1; counts[i+2]+=1

    if changed:
        return best

    # no more melds removable: count taatsu + possible pair (if not used)
    mentsu, taatsu = _greedy_taatsu(counts)
    pair_bonus = 0
    if not pair_used:
        # try using one remaining pair as the pair
        has_pair = any(counts[i] >= 2 for i in range(34))
        if has_pair:
            pair_bonus = 1
    # limit taatsu so mentsu + taatsu <= 4
    taatsu = min(taatsu, 4 - mentsu)
    shanten = 8 - 2*mentsu - taatsu - pair_bonus
    return shanten

def shanten_normal(counts: List[int]) -> int:
    return _dfs_min_shanten(tuple(counts), False)

def shanten_all(counts: List[int]) -> Dict[str, int]:
    return {
        "normal": shanten_normal(counts),
        "chiitoi": shanten_chiitoi(counts),
        "kokushi": shanten_kokushi(counts),
    }

# ---------------- Ukeire (normal hand only, with visibility) ---------------- #

def ukeire_normal(counts_hand: List[int], counts_visible: List[int]) -> Dict[TileId, int]:
    """
    Returns a map tile_id -> remaining visible draws that reduce shanten (normal).
    visible = hand + discards + open meld tiles (+ anything else you know).
    """
    base_sh = shanten_normal(counts_hand)
    out: Dict[int, int] = {}
    for t in range(34):
        if counts_hand[t] >= 4:
            continue
        # simulate drawing t
        counts_hand[t] += 1
        new_sh = shanten_normal(counts_hand)
        counts_hand[t] -= 1
        if new_sh < base_sh:
            remain = max(0, 4 - counts_visible[t])  # unseen copies
            if remain > 0:
                out[t] = remain
    return out
