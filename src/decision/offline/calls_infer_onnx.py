from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import onnxruntime as ort

# reuse your existing helpers
from src.decision.offline.feature_builders import tiles_list_to_counts34, phys_to_34
from src.decision.features import shanten_all

# ---- ONNX wrapper ----
class CallsPolicyONNX:
    def __init__(self, onnx_path: str):
        providers = ["CPUExecutionProvider"]
        try:
            from onnxruntime.capi._pybind_state import get_available_providers
            av = get_available_providers()
            if "CUDAExecutionProvider" in av:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name   # "feat"
        self.out_name = self.sess.get_outputs()[0].name # "score"

    def score_many(self, feats: np.ndarray) -> np.ndarray:
        """
        feats: (K, D) float32
        returns scores: (K,) float32 (higher=better)
        """
        if feats.ndim == 1:
            feats = feats[None, :]
        out = self.sess.run([self.out_name], {self.in_name: feats.astype(np.float32)})[0]
        return out.reshape(-1)

# ---- option building ----
# match training mapping
CALL_TYPES = {2: "chi", 3: "pon", 4: "daiminkan", 6: "ankan"}  # (shouminkan 5 omitted unless you track pons)

def _one_hot_calltype(call_type: int) -> np.ndarray:
    # order: skip, chi, pon, daiminkan, shouminkan, ankan
    keys = [-1, 2, 3, 4, 5, 6]
    v = np.zeros(len(keys), np.float32)
    if call_type in keys:
        v[keys.index(call_type)] = 1.0
    return v

def _relative_who_onehot(src_wind: int, self_wind: int) -> np.ndarray:
    # 0: n/a, 1:left, 2:across, 3:right
    v = np.zeros(4, np.float32)
    if src_wind < 0:
        v[0] = 1.0; return v
    rel = (src_wind - self_wind) % 4
    if   rel == 1: v[1] = 1.0
    elif rel == 2: v[2] = 1.0
    elif rel == 3: v[3] = 1.0
    else:          v[0] = 1.0
    return v

def _apply_call_to_hand_counts(hand34: List[int], action: dict, player_wind: int) -> List[int]:
    post = hand34[:]
    tiles = action.get("tiles", []) or []
    who   = action.get("who", []) or []
    ttype = int(action.get("type", -1))
    for i, p in enumerate(tiles):
        if p < 0: continue
        owner = who[i] if i < len(who) else -1
        t34 = phys_to_34(p)
        if 0 <= t34 < 34:
            if ttype == 6 or owner == player_wind:
                if post[t34] > 0:
                    post[t34] -= 1
    return post

def _visible_counts34(sample: dict, hand34: List[int]) -> List[int]:
    vis34 = hand34[:]
    for seat in ("0","1","2","3"):
        info = sample.get(seat, {}) or {}
        disc34 = tiles_list_to_counts34(info.get("discards", []) or [])
        for i in range(34): vis34[i] += disc34[i]
        for m in (info.get("melds", []) or []):
            for p in (m.get("tiles", []) or []):
                if p >= 0:
                    t = phys_to_34(p)
                    if 0 <= t < 34:
                        vis34[t] += 1
    return vis34

def _dora_onehot(sample: dict) -> np.ndarray:
    dora = np.zeros(34, np.float32)
    for p in (sample.get("dora_indicators", []) or []):
        if p < 0: continue
        t = phys_to_34(p)
        if 0 <= t < 34:
            if   0 <= t <= 8:  dora[(t - 0 + 1) % 9 + 0] = 1
            elif 9 <= t <= 17: dora[(t - 9 + 1) % 9 + 9] = 1
            elif 18<= t <= 26: dora[(t - 18 + 1)% 9 + 18] = 1
            else:              dora[27 + ((t - 27 + 1) % 7)] = 1
    return dora

def _option_features(sample: dict, action: Optional[dict]) -> Tuple[np.ndarray, dict]:
    """Build the same per-option feature vector used at training time."""
    player_wind = int(sample.get("player_wind", 0)) % 4
    round_wind  = int(sample.get("round_wind", 0)) % 4
    num_honba   = float(sample.get("num_honba", 0))
    num_riichi  = float(sample.get("num_riichi", 0))
    remain_tiles= float(sample.get("remain_tiles", 70))

    hand_phys = sample["hand_tiles"]
    hand34 = tiles_list_to_counts34(hand_phys)

    if action is None:
        post_hand34 = hand34[:]
        call_type = -1
        src_who = -1
    else:
        post_hand34 = _apply_call_to_hand_counts(hand34, action, player_wind)
        call_type = int(action.get("type", -1))
        who_list = action.get("who", []) or []
        src_who = next((w for w in who_list if w != player_wind and w >= 0), -1)

    vis34 = _visible_counts34(sample, hand34)
    dora = _dora_onehot(sample)

    riichi_flags = np.array([
        1.0 if (sample.get("0",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("1",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("2",{}).get("riichi", False)) else 0.0,
        1.0 if (sample.get("3",{}).get("riichi", False)) else 0.0,
    ], dtype=np.float32)

    sh_pre = shanten_all(hand34)
    sh_post= shanten_all(post_hand34)
    sh_vec = np.array([
        float(sh_pre["normal"]), float(sh_post["normal"]),
        float(sh_pre["chiitoi"]), float(sh_post["chiitoi"]),
        float(sh_pre["kokushi"]), float(sh_post["kokushi"]),
        float(sh_post["normal"] - sh_pre["normal"]),
        float(sh_post["chiitoi"] - sh_pre["chiitoi"]),
        float(sh_post["kokushi"] - sh_pre["kokushi"]),
    ], dtype=np.float32)

    round_oh = np.zeros(4, np.float32); round_oh[round_wind] = 1.0
    seat_oh  = np.zeros(4, np.float32); seat_oh[player_wind] = 1.0
    call_oh  = _one_hot_calltype(call_type)
    relwho_oh= _relative_who_onehot(src_who, player_wind)

    feat = np.concatenate([
        np.asarray(post_hand34, np.float32),   # IMPORTANT: post-call hand
        np.asarray(vis34, np.float32),
        dora,
        round_oh, seat_oh,
        np.array([num_honba, num_riichi, remain_tiles/70.0], np.float32),
        sh_vec,
        riichi_flags,
        call_oh,
        relwho_oh,
    ], axis=0).astype(np.float32)

    meta = {"type": call_type, "src_who": src_who,
            "d_shanten_normal": float(sh_post["normal"] - sh_pre["normal"])}

    return feat, meta

# ---- build options from live state ----
def build_call_options_from_live(
    counts34_hand: List[int],
    last_discard_tid: Optional[int],
    from_seat_wind: Optional[int],
    player_wind: int,
) -> List[dict]:
    """
    Build a list of action dicts (the same shape as logs' valid_actions entries) for the current moment.
    Includes: SKIP (represented as None), CHI (only if from left), PON, DAIMINKAN, ANKAN.
    We return only *call* actions; SKIP is handled by passing action=None to _option_features.
    """
    opts: List[dict] = []
    if last_discard_tid is None or from_seat_wind is None:
        # no opponent discard visible -> only self-kan and skip are sensible
        pass
    else:
        rel = (from_seat_wind - player_wind) % 4
        t = last_discard_tid

        # PON: need 2 in hand
        if counts34_hand[t] >= 2:
            opts.append({"type": 3, "tiles": [t*4, t*4+1, t*4+2], "who": [player_wind, player_wind, from_seat_wind]})

        # DAIMINKAN: need 3 in hand
        if counts34_hand[t] >= 3:
            opts.append({"type": 4, "tiles": [t*4, t*4+1, t*4+2, t*4+3], "who": [player_wind, player_wind, player_wind, from_seat_wind]})

        # CHI: only allowed if from left player (rel == 1) and tile is suit (0..26)
        if rel == 1 and 0 <= t <= 26:
            base = t % 9  # 0..8
            suit0 = (t // 9) * 9
            # left chi: (t-2, t-1, t)
            if base >= 2:
                need = [t-2, t-1]
                if all(0 <= k <= 26 and counts34_hand[k] >= 1 for k in need):
                    opts.append({"type": 2, "tiles": [need[0]*4, need[1]*4, t*4], "who": [player_wind, player_wind, from_seat_wind]})
            # mid chi: (t-1, t, t+1)
            if 1 <= base <= 7:
                need = [t-1, t+1]
                if all(0 <= k <= 26 and counts34_hand[k] >= 1 for k in need):
                    opts.append({"type": 2, "tiles": [need[0]*4, t*4, need[1]*4], "who": [player_wind, from_seat_wind, player_wind]})
            # right chi: (t, t+1, t+2)
            if base <= 6:
                need = [t+1, t+2]
                if all(0 <= k <= 26 and counts34_hand[k] >= 1 for k in need):
                    opts.append({"type": 2, "tiles": [t*4, need[0]*4, need[1]*4], "who": [from_seat_wind, player_wind, player_wind]})

    # ANKAN (self-only): any tile with count==4
    for tid, c in enumerate(counts34_hand):
        if c == 4:
            opts.append({"type": 6, "tiles": [tid*4, tid*4+1, tid*4+2, tid*4+3], "who": [player_wind, player_wind, player_wind, player_wind]})

    return opts

def build_runtime_sample(
    counts34_hand: List[int],
    discards_by_seat: Dict[int, List[int]],
    player_wind: int = 0,
    round_wind: int = 0,
    remain_tiles: int = 70,
    dora_indicators_phys: Optional[List[int]] = None,
) -> dict:
    """Create a minimal 'log-style' dict so we can reuse _option_features exactly."""
    # synthesize phys ids for hand
    hand_phys = []
    for tid, cnt in enumerate(counts34_hand):
        for k in range(cnt):
            hand_phys.append(tid*4 + k)

    sample = {
        "round_wind": round_wind,
        "num_honba": 0,
        "num_riichi": 0,
        "dora_indicators": dora_indicators_phys or [],
        "player_wind": player_wind,
        "position": player_wind,
        "hand_tiles": hand_phys,
        "valid_actions": [],
        "action_idx": -1,
        "remain_tiles": remain_tiles,
        "0": {"points": 25000, "melds": [], "discards": discards_by_seat.get(0, []), "tsumo_giri": [], "riichi": False},
        "1": {"points": 25000, "melds": [], "discards": discards_by_seat.get(1, []), "tsumo_giri": [], "riichi": False},
        "2": {"points": 25000, "melds": [], "discards": discards_by_seat.get(2, []), "tsumo_giri": [], "riichi": False},
        "3": {"points": 25000, "melds": [], "discards": discards_by_seat.get(3, []), "tsumo_giri": [], "riichi": False},
    }
    return sample

def score_call_set(
    calls_scorer: CallsPolicyONNX,
    runtime_sample: dict,
    options: List[Optional[dict]],
    *,
    temperature: float = 1.5,   # >1 flattens (less overconfident)
    skip_bias: float = 0.10,    # add to skip score as a prior
) -> Tuple[int, List[Tuple[int, float]], List[dict]]:
    """
    Returns:
      best_idx: argmax after softmax( scores / temperature ) with skip bias
      ranked:   [(idx, prob)] in descending order
      metas:    per-option meta dicts (contains d_shanten_normal, type, etc.)
    """
    feats, metas = [], []
    for a in options:
        f, m = _option_features(runtime_sample, a)
        feats.append(f); metas.append(m)

    feats = np.stack(feats, axis=0)
    scores = calls_scorer.score_many(feats)  # (K,)

    # apply skip bias to option 0 (skip)
    scores = scores.copy()
    if len(scores) > 0:
        scores[0] += skip_bias

    # temperature softmax
    s = scores / max(1e-6, float(temperature))
    m = s.max()
    probs = np.exp(s - m); Z = probs.sum()
    probs = probs / (Z if Z > 0 else 1.0)

    ranked = sorted([(i, float(probs[i])) for i in range(len(options))], key=lambda x: -x[1])
    best_idx = int(np.argmax(probs))
    return best_idx, ranked, metas

