# src/main.py
from __future__ import annotations
import time
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- window & capture utilities you already have ---
from src.perception.utils.window_finder import (
    set_process_dpi_aware,
    find_window_by_titles,
    set_window_topmost,
)
from src.perception.capture.screen_capture import ScreenCapturer, annotate_hud
from src.perception.utils.capture_exclude import set_window_exclude_from_capture

# --- detectors & recognizer ---
from src.perception.detectors.yolo_seg_onnx import YoloSegOnnxDetector
from src.perception.tiles.cnn_recognizer import CNNRecognizer

# --- decision core (new) ---
from src.decision.state import GameState, tile_str_to_id, tile_id_to_str
from src.decision.features import shanten_all, shanten_normal, ukeire_normal

from src.decision.offline.bc_infer_onnx import BCPolicyONNX, policy_topk
from src.decision.offline.feature_builders import build_bc_features

from src.decision.offline.calls_infer_onnx import (
    CallsPolicyONNX,
    build_call_options_from_live,
    build_runtime_sample,
    score_call_set,
)

# ===================== Config =====================
# YOLOv8-seg ONNX (ROI-aware)
YOLO_SEG_ONNX = "models/detectors/tiles_roi_yolov8n_seg.onnx"
YOLO_SEG_CLASSES = [
    "tile_discards_bottom",
    "tile_discards_left",
    "tile_discards_right",
    "tile_discards_top",
    "tile_hand_self",
    "tile_melds_left",
    "tile_melds_right",
    "tile_melds_self",
    "tile_melds_top",
]

TILE_CNN_WEIGHTS = "models/tiles/cnn_best.pt"
BC_ONNX = "runs/bc_policy/bc_policy.onnx"
CALLS_ONNX = "runs/calls_policy/calls_policy.onnx"  # adjust to your path

FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.8

# Detector thresholds (adjustable with hotkeys)
CONF_THR = 0.35
IOU_THR = 0.50

# Window title candidates for Mahjong Soul
TITLE_HINTS = ["雀魂麻將", "雀魂", "Mahjong Soul"]

CNN_ONNX = "models/tile_cnn/tile_cnn.onnx"
CNN_CLASSES = "models/tile_cnn/classes.json"
CNN_IMG_SIZE = 128

class ManualTableState:
    def __init__(self):
        self.round_wind = 0   # 0:E,1:S,2:W,3:N
        self.self_wind  = 0   # 0:E,1:S,2:W,3:N (your seat)
        self.honba = 0
        self.riichi_sticks = 0
        self.remain_tiles = 70
        self.riichi_flags = [False, False, False, False]  # seat 0..3
        self.dora_phys = []   # list of phys ids (0..135). We'll store tid*4.

    def winds_str(self):
        W = ["E","S","W","N"]
        return f"round={W[self.round_wind]} self={W[self.self_wind]}"

    def dora_str(self):
        # show effective dora tiles, not the indicator
        if not self.dora_phys: return "[]"
        # convert indicator phys -> indicator tid -> effective dora tid label
        labs = []
        for pid in self.dora_phys:
            tid = pid // 4
            # effective dora mapping (same as feature builder)
            if   0 <= tid <= 8:  d = (tid - 0 + 1) % 9 + 0
            elif 9 <= tid <= 17: d = (tid - 9 + 1) % 9 + 9
            elif 18<= tid <= 26: d = (tid - 18 + 1) % 9 + 18
            else:                d = 27 + ((tid - 27 + 1) % 7)
            labs.append(tile_id_to_str(d))
        return "[" + " ".join(labs) + "]"

# ===================== Small helpers =====================
def add_dora_indicator_by_label(state, s: str):
    """Add ONE dora *indicator* by human-friendly label.
       Honors: 'e','s','w','n','p','f','c'  (case-insensitive)
       Suits : '1m'..'9m', '1p'..'9p', '1s'..'9s' (digits+letter, case-insensitive)
    """
    if not s:
        return
    s = s.strip()
    # Honors
    if len(s) == 1:
        ch = s.upper()
        honor_map = {"E":27, "S":28, "W":29, "N":30, "P":31, "F":32, "C":33}
        tid = honor_map.get(ch, None)
        if tid is None:
            return
        state.dora_phys.append(tid * 4)
        return
    # Suits: e.g. '5m'
    if len(s) == 2 and s[0].isdigit():
        num = int(s[0])
        suit = s[1].lower()
        if not (1 <= num <= 9): return
        if suit == "m": base = 0
        elif suit == "p": base = 9
        elif suit == "s": base = 18
        else: return
        tid = base + (num - 1)
        state.dora_phys.append(tid * 4)

def remove_last_dora_indicator(state: ManualTableState):
    if state.dora_phys:
        state.dora_phys.pop()

def get_cv_window_hwnd(title: str) -> Optional[int]:
    try:
        import win32gui
        hwnd = win32gui.FindWindow(None, title)
        return int(hwnd) if hwnd else None
    except Exception:
        return None

def put_label(img: np.ndarray, text: str, org: Tuple[int, int], color=(235, 235, 235)):
    cv2.putText(img, text, org, FONT, HUD_SCALE, color, 2, cv2.LINE_AA)

    
def draw_panel(img: np.ndarray, lines: List[str], topleft=(10, 26), color=(235,235,235)):
    x, y = topleft
    for i, ln in enumerate(lines):
        yy = y + i * int(22 * HUD_SCALE)
        put_label(img, ln, (x, yy), color=color)

def crops_from_masks(frame_bgr: np.ndarray, dets: List[dict], cls_filter: str) -> List[np.ndarray]:
    crops = []
    for d in dets:
        name = d.get("cls_name") or d.get("class_name") or ""
        if name != cls_filter:
            continue
        x1, y1, x2, y2 = map(int, d["xyxy"])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame_bgr.shape[1], x2); y2 = min(frame_bgr.shape[0], y2)
        if x2 - x1 >= 6 and y2 - y1 >= 6:
            crops.append(frame_bgr[y1:y2, x1:x2].copy())
    return crops

def labels_to_counts34(labels: List[str]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    labels: list like ["5m","5mr","E",...,"unknown"]
    returns:
      counts34: length-34 counts
      hand_list: list[(tid, red_flag)] for legal tiles only (unknown skipped)
    """
    counts = [0] * 34
    hand_list: List[Tuple[int, int]] = []
    if not labels:
        return counts, hand_list
    for (lab, ang) in labels:
        if not lab or lab == "unknown":
            continue
        red = 0
        if lab.endswith("r"):
            red = 1
            lab = lab[:-1]
        if lab in ["E","S","W","N","P","F","C"]:
            tid = 27 + ["E","S","W","N","P","F","C"].index(lab)
        else:
            n = int(lab[0]); suit = lab[1]
            if suit == "m":
                tid = n - 1
            elif suit == "p":
                tid = 9 + (n - 1)
            else:
                tid = 18 + (n - 1)
        counts[tid] += 1
        hand_list.append((tid, red))
    return counts, hand_list

def synth_phys_from_counts(counts34: List[int]) -> List[int]:
    """Make fake 0..135 ids so build_bc_features() matches training (uses //4 internally)."""
    out = []
    for tid, cnt in enumerate(counts34):
        for k in range(cnt):
            out.append(tid * 4 + k)
    return out

def label_to_tid(lbl: str) -> Optional[int]:
    if not lbl or lbl == "unknown":
        return None
    if lbl in ["E","S","W","N","P","F","C"]:
        return 27 + ["E","S","W","N","P","F","C"].index(lbl)
    n = int(lbl[0]); suit = lbl[1]
    if suit == "m": return n - 1
    if suit == "p": return 9 + (n - 1)
    return 18 + (n - 1)

def last_discard_tid(labels: List[str]) -> Optional[int]:
    # heuristic: assume the rightmost/bottom-most crop is the newest – YOLO order may be arbitrary,
    # but your crops list is built in the order of dets; often newest is appended last.
    for (lbl, ang) in reversed(labels):
        tid = label_to_tid(lbl)
        if tid is not None:
            return tid
    return None

def to_phys_list(pairs: List[Tuple[str,int]]) -> List[int]:
    out = []
    for (lbl, ang) in pairs:
        tid = label_to_tid(lbl)
        if tid is not None:
            out.append(tid * 4)
    return out

def sorted_dets_for_roi(dets: List[dict], roi_name: str) -> List[dict]:
    # filter
    filt = [d for d in dets if (d.get("cls_name") or d.get("class_name") or "") == roi_name]
    if not filt:
        return []
    # sort key
    if roi_name in ("tile_discards_bottom", "tile_discards_top"):
        # left->right
        return sorted(filt, key=lambda d: (d["xyxy"][0] + d["xyxy"][2]) * 0.5)
    else:  # left/right rivers: top->bottom
        return sorted(filt, key=lambda d: (d["xyxy"][1] + d["xyxy"][3]) * 0.5)

def crops_from_dets(frame_bgr: np.ndarray, det_list: List[dict]) -> List[np.ndarray]:
    crops = []
    H, W = frame_bgr.shape[:2]
    for d in det_list:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2 - x1 >= 6 and y2 - y1 >= 6:
            crops.append(frame_bgr[y1:y2, x1:x2].copy())
    return crops

def sorted_dets_for_roi(dets: List[dict], roi_name: str) -> List[dict]:
    """Stable geometric ordering per river so 'last' ~ newest."""
    filt = [d for d in dets if (d.get("cls_name") or d.get("class_name") or "") == roi_name]
    if not filt:
        return []
    # bottom/top: left->right; left/right: top->bottom
    if roi_name in ("tile_discards_bottom", "tile_discards_top"):
        return sorted(filt, key=lambda d: (d["xyxy"][0] + d["xyxy"][2]) * 0.5)
    else:
        return sorted(filt, key=lambda d: (d["xyxy"][1] + d["xyxy"][3]) * 0.5)

def dets_to_crops(frame_bgr: np.ndarray, det_list: List[dict]) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    crops, boxes = [], []
    H, W = frame_bgr.shape[:2]
    for d in det_list:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        if x2 - x1 >= 6 and y2 - y1 >= 6:
            crops.append(frame_bgr[y1:y2, x1:x2].copy())
            boxes.append((x1, y1, x2, y2))
    return crops, boxes

# ===================== Main =====================
def main():
    set_process_dpi_aware()

    # Find Mahjong Soul window (client area)
    found = None
    for hint in TITLE_HINTS:
        found = find_window_by_titles(hint)
        if found:
            break
    if not found:
        print("[ERR] Mahjong Soul window not found. Open the table and run again.")
        return

    hwnd_game, rect = found
    l, t, r, b = rect
    W, H = r - l, b - t
    print(f"[INFO] Capturing client area {rect} size={W}x{H}")

    # Keep game window on top while preview is open
    set_window_topmost(hwnd_game, True)
    atexit.register(lambda: set_window_topmost(hwnd_game, False))

    # Screen capturer for that rect
    cap = ScreenCapturer(rect)

    # Create preview window and exclude it from capture (avoid mirror)
    win_title = "Mahjong Perception + Decision"
    cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_title, 50, 50)
    hwnd_prev = get_cv_window_hwnd(win_title)
    if hwnd_prev:
        set_window_exclude_from_capture(hwnd_prev, True)

    # Models
    yolo = YoloSegOnnxDetector(
        onnx_path=YOLO_SEG_ONNX,
        class_names=YOLO_SEG_CLASSES,
        is_xywh=True,
        input_size=768
    )
    recognizer = CNNRecognizer(
        onnx_path=CNN_ONNX,
        classes_path=CNN_CLASSES,
        img_size=CNN_IMG_SIZE,
    )
    
    manual = ManualTableState()


    calls_policy = None
    if Path(CALLS_ONNX).exists():
        calls_policy = CallsPolicyONNX(CALLS_ONNX)
        print(f"[INFO] Loaded Calls policy: {CALLS_ONNX}")
    else:
        print(f"[WARN] Calls policy not found at {CALLS_ONNX}; skipping call suggestions.")


    bc_policy = None
    if Path(BC_ONNX).exists():
        bc_policy = BCPolicyONNX(BC_ONNX)
        print(f"[INFO] Loaded BC policy: {BC_ONNX}")
    else:
        print(f"[WARN] BC policy not found at {BC_ONNX}; skipping BC suggestions.")

    last_ts = time.time()
    frames = 0

    while True:
        frame_bgr = cap.grab()
        if frame_bgr is None:
            continue

        # Detect tile ROIs with YOLO-seg
        dets = yolo.infer(frame_bgr)  # list of dict: {"xyxy":[x1,y1,x2,y2], "cls_name":str, "score":float, "mask":...}

        # Crop key ROIs
        dets_hand        = sorted_dets_for_roi(dets, "tile_hand_self")
        dets_melds_self  = sorted_dets_for_roi(dets, "tile_melds_self")

        crops_hand,       boxes_hand       = dets_to_crops(frame_bgr, dets_hand)
        crops_melds_self, boxes_melds_self = dets_to_crops(frame_bgr, dets_melds_self)

        labels_hand        = recognizer.classify_many_tta(crops_hand,        score_thresh=0.60, angles=[0,90,180,270])
        labels_melds_self  = recognizer.classify_many_tta(crops_melds_self,  score_thresh=0.60, angles=[0,90,180,270])


        # also crop & classify for left/right/top (add these if not already)
        dets_disc_bottom = sorted_dets_for_roi(dets, "tile_discards_bottom")
        dets_disc_left   = sorted_dets_for_roi(dets, "tile_discards_left")
        dets_disc_right  = sorted_dets_for_roi(dets, "tile_discards_right")
        dets_disc_top    = sorted_dets_for_roi(dets, "tile_discards_top")

        crops_disc_bottom, boxes_disc_bottom = dets_to_crops(frame_bgr, dets_disc_bottom)
        crops_disc_left,   boxes_disc_left   = dets_to_crops(frame_bgr, dets_disc_left)
        crops_disc_right,  boxes_disc_right  = dets_to_crops(frame_bgr, dets_disc_right)
        crops_disc_top,    boxes_disc_top    = dets_to_crops(frame_bgr, dets_disc_top)

        labels_disc_bottom = recognizer.classify_many_tta(crops_disc_bottom, score_thresh=0.60, angles=[0,90,180,270])
        labels_disc_left   = recognizer.classify_many_tta(crops_disc_left,   score_thresh=0.60, angles=[0,90,180,270])
        labels_disc_right  = recognizer.classify_many_tta(crops_disc_right,  score_thresh=0.60, angles=[0,90,180,270])
        labels_disc_top    = recognizer.classify_many_tta(crops_disc_top,    score_thresh=0.60, angles=[0,90,180,270])


        # Convert hand labels -> counts + list
        counts34_hand, hand_list = labels_to_counts34(labels_hand)

        discards_by_seat_phys = {
            0: to_phys_list(labels_disc_bottom),
            1: to_phys_list(labels_disc_left),
            2: to_phys_list(labels_disc_top),
            3: to_phys_list(labels_disc_right),
        }

        approx_remain = max(1, 70 - (
            len(labels_disc_bottom) + len(labels_disc_left) +
            len(labels_disc_right) + len(labels_disc_top)
        ))
        runtime_sample = build_runtime_sample(
            counts34_hand=counts34_hand,
            discards_by_seat=discards_by_seat_phys,
            player_wind=manual.self_wind,
            round_wind=manual.round_wind,
            remain_tiles=manual.remain_tiles,
            dora_indicators_phys=manual.dora_phys,  # phys list
        )

        # -------- BC Top-3 suggestions (NEW) --------
        bc_top3_display: List[str] = []
        if bc_policy is not None and len(hand_list) > 0:
            # Build minimal log-like sample for the feature builder (to match training layout exactly)
            hand_phys = synth_phys_from_counts(counts34_hand)
            discards_by_seat: Dict[int, List[int]] = {0: [], 1: [], 2: [], 3: []}
            for (lab, ang) in labels_disc_bottom:
                if not lab or lab == "unknown":
                    continue
                if lab in ["E","S","W","N","P","F","C"]:
                    tid = 27 + ["E","S","W","N","P","F","C"].index(lab)
                else:
                    n = int(lab[0]); suit = lab[1]
                    if suit == "m": tid = n - 1
                    elif suit == "p": tid = 9 + (n - 1)
                    else: tid = 18 + (n - 1)
                discards_by_seat[0].append(tid * 4)

            sample_for_bc = {
                "round_wind": manual.round_wind,
                "num_honba": manual.honba,
                "num_riichi": manual.riichi_sticks,
                "dora_indicators": manual.dora_phys,     # phys list
                "player_wind": manual.self_wind,
                "position": manual.self_wind,
                "hand_tiles": hand_phys,
                "valid_actions": [],
                "action_idx": -1,
                "remain_tiles": manual.remain_tiles,
                "0": {"points": 25000, "melds": [], "discards": discards_by_seat[0], "tsumo_giri": [], "riichi": manual.riichi_flags[0]},
                "1": {"points": 25000, "melds": [], "discards": discards_by_seat[1], "tsumo_giri": [], "riichi": manual.riichi_flags[1]},
                "2": {"points": 25000, "melds": [], "discards": discards_by_seat[2], "tsumo_giri": [], "riichi": manual.riichi_flags[2]},
                "3": {"points": 25000, "melds": [], "discards": discards_by_seat[3], "tsumo_giri": [], "riichi": manual.riichi_flags[3]},
            }

            feat_vec, mask_vec, _ = build_bc_features(sample_for_bc)  # (D,), (34,)
            # Replace mask with exact legal mask from current hand
            mask_vec = np.zeros(34, np.float32)
            for tid, _r in hand_list:
                mask_vec[tid] = 1.0

            logits = bc_policy(feat_vec)  # (34,)
            top3 = policy_topk(logits, mask_vec, k=3)  # [(tid, prob), ...]
            bc_top3_display = [f"{tile_id_to_str(t)}({p:.2f})" for (t, p) in top3]

        call_suggestion_text = "—"

        debug_opts: List[str] = []

        if calls_policy is not None:
            # choose a seat with a visible last discard (priority: right → across → left, tweak as you like)
            seat_last = {
                1: last_discard_tid(labels_disc_left),
                2: last_discard_tid(labels_disc_top),
                3: last_discard_tid(labels_disc_right),
            }
            chosen_seat, last_tid = None, None
            for seat in (1, 2, 3):  # prefer left -> top -> right (tweak order if you like)
                if seat_last[seat] is not None:
                    chosen_seat, last_tid = seat, seat_last[seat]
                    break

            # options list = [None (skip)] + call dicts
            options = [None]
            if last_tid is not None and chosen_seat is not None:
                options += build_call_options_from_live(
                    counts34_hand=counts34_hand,
                    last_discard_tid=last_tid,
                    from_seat_wind=chosen_seat,
                    player_wind=0,
                )
            else:
                options += build_call_options_from_live(
                    counts34_hand=counts34_hand,
                    last_discard_tid=None,
                    from_seat_wind=None,
                    player_wind=0,
                )

            for a in options:
                if a is None: debug_opts.append("skip")
                else:
                    t = int(a.get("type", -1))
                    debug_opts.append({2:"chi",3:"pon",4:"dkan",6:"akan"}.get(t, f"type{t}"))

            if len(options) > 1:
                best_idx, ranked, metas = score_call_set(
                    calls_policy, runtime_sample, options,
                    temperature=1.6,     # tune 1.3–2.0 as needed
                    skip_bias=0.10       # tune 0.05–0.20 as needed
                )

                # Extract skip prob and best call candidate
                prob_skip = next((p for i,p in ranked if i == 0), 0.0)
                best_call_idx, best_call_prob = next(((i,p) for i,p in ranked if i != 0), (0,0.0))
                best_call_meta = metas[best_call_idx] if best_call_idx < len(metas) else {"d_shanten_normal": 999}

                # Guards
                MIN_MARGIN = 0.15
                MIN_PROB   = 0.55
                improves_shanten = float(best_call_meta.get("d_shanten_normal", 1.0)) <= 0.0
                strong_margin    = (best_call_prob - prob_skip) >= MIN_MARGIN
                strong_prob      = best_call_prob >= MIN_PROB

                if best_call_idx != 0 and improves_shanten and strong_margin and strong_prob:
                    tname = {2:"Chi",3:"Pon",4:"Daiminkan",6:"Ankan"}.get(int(options[best_call_idx]["type"]), "?")
                    call_suggestion_text = f"{tname} ({best_call_prob:.2f}) Δshan={best_call_meta['d_shanten_normal']:+.0f}"
                else:
                    call_suggestion_text = f"Skip ({prob_skip:.2f})"

            if calls_policy is None:
                call_suggestion_text = "calls onnx not loaded"
            elif last_tid is None and not any(c == 4 for c in counts34_hand):
                call_suggestion_text = "no opp discard & no ankan"
            elif len(options) <= 1:
                call_suggestion_text = "no legal calls -> skip"

#            ranked_labels = []
#            for idx, prob in ranked[:3]:
#                if options[idx] is None:
#                    ranked_labels.append(f"skip({prob:.2f})")
#                else:
#                    t = int(options[idx]["type"])
#                    ranked_labels.append(f"{ {2:'chi',3:'pon',4:'dkan',6:'akan'}[t] }({prob:.2f})")


        # -------- Visualization --------
        vis = frame_bgr.copy()

        def draw_labeled_boxes(boxes: List[Tuple[int,int,int,int]], labels: List[Tuple[str,int]], color=(80,200,240)):
            for (x1,y1,x2,y2), (lab, ang) in zip(boxes, labels):
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                txt = lab if lab else "unknown"
                put_label(vis, txt, (x1+2, y1+18), color)

        # draw boxes (optional)
        C_HAND  = (0, 50, 40)
        C_MELD  = (180, 180, 60)
        C_RIVER = (235, 0, 235)

        draw_labeled_boxes(boxes_hand,       labels_hand,       C_HAND)
        draw_labeled_boxes(boxes_melds_self, labels_melds_self, C_MELD)
        draw_labeled_boxes(boxes_disc_bottom,labels_disc_bottom,C_RIVER)
        draw_labeled_boxes(boxes_disc_left,  labels_disc_left,  C_RIVER)
        draw_labeled_boxes(boxes_disc_right, labels_disc_right, C_RIVER)
        draw_labeled_boxes(boxes_disc_top,   labels_disc_top,   C_RIVER)

        # HUD text
        hud_lines = []
        if labels_hand:
            hud_lines.append(" ".join([lab if lab is not None else "unknown" for (lab, ang) in labels_hand]))
            hud_lines.append("")
        if labels_melds_self:
            hud_lines.append(" ".join([lab if lab is not None else "unknown" for (lab, ang) in labels_melds_self]))
            hud_lines.append("")
        if labels_disc_bottom:
            hud_lines.append(" ".join([lab if lab is not None else "unknown" for (lab, ang) in labels_disc_bottom]))
            hud_lines.append("")
        if bc_top3_display:
            hud_lines.append("BC top3: " + "  ".join(bc_top3_display))
        else:
            hud_lines.append("BC top3: (policy not loaded or no hand)")
        
        hud_lines.append("")
        hud_lines.append(f"CALL: {call_suggestion_text}")

        hud_lines.append("")
        hud_lines.append("CALL opts: " + (" ".join(debug_opts) if debug_opts else "—"))

        hud_lines.append("")
        hud_lines.append(f"{manual.winds_str()}  honba={manual.honba}  riichi_sticks={manual.riichi_sticks}  remain={manual.remain_tiles}")
        hud_lines.append("")
        hud_lines.append(f"riichi flags: {['✓' if f else '-' for f in manual.riichi_flags]}")
        hud_lines.append("")
        hud_lines.append(f"dora: {manual.dora_str()}")

        draw_panel(vis, hud_lines, (10, 26), (0,0,235))

        # fps
        frames += 1
        if frames % 20 == 0:
            now = time.time()
            fps = 20.0 / (now - last_ts)
            last_ts = now
            put_label(vis, f"FPS: {fps:.1f}", (10, vis.shape[0] - 10), (255, 190, 60))

        cv2.imshow("Mahjong Perception + BC Top-3", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        
        if key == ord('R'):  # cycle ROUND wind: E->S->W->N
            manual.round_wind = (manual.round_wind + 1) % 4
        if key == ord('r'):  # cycle SELF wind
            manual.self_wind = (manual.self_wind + 1) % 4

        # ---- Hotkeys: Honba / Riichi sticks / Remain tiles ----



        if key == ord('+') or key == ord('='): manual.remain_tiles = min(70, manual.remain_tiles + 1)
        if key == ord('-') or key == ord('_'): manual.remain_tiles = max(1,  manual.remain_tiles - 1)

        if key == ord('h'): manual.honba = min(10, manual.honba + 1)
        if key == ord('H'): manual.honba = max(0,  manual.honba - 1)

        if key == ord('t'): manual.riichi_sticks = min(10, manual.riichi_sticks + 1)
        if key == ord('T'): manual.riichi_sticks = max(0,  manual.riichi_sticks - 1)

        # Toggle riichi flags for seats 0..3
        if key == ord('z'): manual.riichi_flags[0] = not manual.riichi_flags[0]
        if key == ord('x'): manual.riichi_flags[1] = not manual.riichi_flags[1]
        if key == ord('c'): manual.riichi_flags[2] = not manual.riichi_flags[2]
        if key == ord('v'): manual.riichi_flags[3] = not manual.riichi_flags[3]

        # ---- Hotkeys: Dora indicators (type codes, press Enter to commit, Backspace to undo) ----
        # Simple mode: press suit key 'm','p','s' to set suit context, then digits '1'..'9' to add a dora indicator.
        # Honors: press one of 'E','S','W','N','P','F','C' to add directly.
        # Backspace removes last. '0' clears all.
        if key in (ord('m'), ord('p'), ord('s')):
            suit_ctx = chr(key)  # store in outer scope: define suit_ctx='m' before loop
        elif key in (ord('1'),ord('2'),ord('3'),ord('4'),ord('5'),ord('6'),ord('7'),ord('8'),ord('9')):
            if 'suit_ctx' in locals():
                add_dora_indicator_by_label(manual, f"{chr(key)}{suit_ctx}".replace("m","m").replace("p","p").replace("s","s"))  # e.g., '5m'
        elif key in (ord('E'),ord('S'),ord('W'),ord('N'),ord('P'),ord('F'),ord('C')):
            add_dora_indicator_by_label(manual, chr(key))
        elif key == 8:  # Backspace
            remove_last_dora_indicator(manual)
        elif key == ord('0'):
            manual.dora_phys.clear()

        

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

