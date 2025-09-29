from pydantic import BaseModel
from typing import List, Dict, Optional

class GameState(BaseModel):
    variant: str = "riichi"
    round: Optional[str] = None
    seat: Optional[str] = None
    dora: List[str] = []
    hand: List[str] = []
    melds_self: List[dict] = []
    discards: Dict[str, List[str]] = {}
    turn: Optional[str] = None
    timestamp_ms: int
