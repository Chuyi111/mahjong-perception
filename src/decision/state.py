from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# ---- 34-tile encoding ----
# 0..8:  1m..9m
# 9..17: 1p..9p
# 18..26:1s..9s
# 27:E, 28:S, 29:W, 30:N, 31:P(白), 32:F(發), 33:C(中)
# Red fives share the same ID (4, 13, 22) and are flagged separately.

TileId = int

HONORS = list(range(27, 34))
SUIT_START = {"m": 0, "p": 9, "s": 18}
RED_IDS = {  # id in 34-index that corresponds to a red five (same id as normal 5)
    "5mr": 4,   # 5m id = 4
    "5pr": 13,  # 5p id = 13
    "5sr": 22,  # 5s id = 22
}

def tile_str_to_id(name: str) -> Tuple[TileId, bool]:
    """'1m'..'9s', 'E','S','W','N','P','F','C', plus red '5mr','5pr','5sr'."""
    name = name.strip()
    if name in ("E","S","W","N","P","F","C"):
        return {"E":27,"S":28,"W":29,"N":30,"P":31,"F":32,"C":33}[name], False
    if name in RED_IDS:
        return RED_IDS[name], True
    # normal suited
    if len(name) == 2 and name[0].isdigit() and name[1] in "mps":
        n = int(name[0]); suit = name[1]
        assert 1 <= n <= 9
        return SUIT_START[suit] + (n - 1), False
    raise ValueError(f"Unknown tile name: {name}")

def tile_id_to_str(t: TileId, red: bool=False) -> str:
    if t >= 27:
        return {27:"E",28:"S",29:"W",30:"N",31:"P",32:"F",33:"C"}[t]
    suit = "m" if t < 9 else ("p" if t < 18 else "s")
    n = (t % 9) + 1
    if red and t in (4,13,22):
        return {4:"5mr",13:"5pr",22:"5sr"}[t]
    return f"{n}{suit}"

def is_honor(t: TileId) -> bool:
    return t >= 27

def same_suit(a: TileId, b: TileId) -> bool:
    if a >= 27 or b >= 27: return False
    return (a // 9) == (b // 9)

# ----------------- Structures -----------------
@dataclass(frozen=True)
class Meld:
    kind: str  # "chi" | "pon" | "kan_closed" | "kan_open" | "kan_added"
    tiles: List[TileId]  # 3 or 4 tile ids (34-index; reds share ids)
    from_seat: Optional[int] = None  # 0=self, 1=right, 2=opposite, 3=left

@dataclass(frozen=True)
class Discard:
    tile: TileId
    from_seat: int          # 0..3
    is_riichi: bool=False
    turn_index: int=0

@dataclass
class GameState:
    # visible info
    hand_tiles: List[Tuple[TileId, bool]]  # [(id, is_red)]
    melds_self: List[Meld] = field(default_factory=list)
    discards: Dict[int, List[Discard]] = field(default_factory=lambda: {0:[],1:[],2:[],3:[]})
    dora_indicators: List[TileId] = field(default_factory=list)

    round_wind: Optional[int] = None  # 27..30 for E/S/W/N if you like
    seat_wind: Optional[int] = None   # 27..30 for E/S/W/N if you like
    honba: int = 0
    riichi_sticks: int = 0
    turn_number: int = 0  # total discards so far

    # --------- helpers ---------
    def counts34_hand(self) -> List[int]:
        c = [0]*34
        for tid, _red in self.hand_tiles:
            c[tid] += 1
        return c

    def counts34_visible(self) -> List[int]:
        """Hand + all discards + open meld tiles (closed kans count 4 tiles visible in Soul UI as well)."""
        c = self.counts34_hand()
        for seat, ds in self.discards.items():
            for d in ds:
                c[d.tile] += 1
        for m in self.melds_self:
            for t in m.tiles:
                c[t] += 1
        # (If you later parse opponent melds, include them here too.)
        return c
