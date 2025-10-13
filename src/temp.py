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


# ===================== Config =====================
# YOLOv8-seg ONNX (ROI-aware)
DET_ONNX = "models/detectors/tiles_roi_yolov8n_seg.onnx"
# Your training class order (exactly as in data.yaml)
ROI_CLASSES = [
    "tile_discards_bottom",  # 0
    "tile_discards_left",    # 1
    "tile_discards_right",   # 2
    "tile_discards_top",     # 3
    "tile_hand_self",        # 4
    "tile_melds_left",       # 5
    "tile_melds_right",      # 6
    "tile_melds_self",       # 7
    "tile_melds_top",        # 8
]

# Tile classifier (CNN)
CNN_ONNX = "models/tile_cnn/tile_cnn.onnx"
CNN_CLASSES = "models/tile_cnn/classes.json"
CNN_IMG_SIZE = 128

# Detector thresholds (adjustable with hotkeys)
CONF_THR = 0.35
IOU_THR = 0.50

# Window title candidates for Mahjong Soul
TITLE_HINTS = ["雀魂麻將", "雀魂", "Mahjong Soul"]


BC_ONNX = "runs/bc_policy/bc_policy.onnx"   # adjust to your actual path


# ===================== Small helpers =====================
def get_cv_window_hwnd(title: str) -> Optional[int]:
    try:
        import win32gui
        hwnd = win32gui.FindWindow(None, title)
        return int(hwnd) if hwnd else None
    except Exception:
        return None

def put_label(img, text, xy, color=(255, 255, 255)):
    x, y = xy
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1, cv2.LINE_AA)

def crop_square(img: np.ndarray, xyxy, pad: int = 3) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1] - 1, x2 + pad)
    y2 = min(img.shape[0] - 1, y2 + pad)
    return img[y1:y2 + 1, x1:x2 + 1]

def label_to_id_red(label: str) -> Optional[Tuple[int, bool]]:
    """Convert CNN label (e.g., '5mr','7p','E') -> (tile_id, is_red). Return None if unknown."""
    if not label or label == "unknown":
        return None
    try:
        tid, red = tile_str_to_id(label)
        # If no explicit red flag in label but it's a '5m/5p/5s' red visually, your CNN should have '5mr/5pr/5sr'
        return (tid, red)
    except Exception:
        return None


# ===================== Heuristic chooser =====================
def choose_discard_simple(hand_ids: List[Tuple[int, bool]], visible_cnts: List[int]) -> Tuple[Optional[int], Dict]:
    """
    Baseline: try discarding one copy of each tile (by tile id, ignoring red flags for decision),
    pick action that yields:
      1) minimal shanten (normal), then
      2) maximal total ukeire (sum of counts), then
      3) tie-breaker: prefer discarding isolated/honor singles (simple heuristic).
    Returns (tile_id_to_discard, debug_info)
    """
    # Build counts from hand
    counts = [0] * 34
    for tid, _red in hand_ids:
        counts[tid] += 1

    best_tid = None
    best_key = None
    debug_rows = []

    base_sh = shanten_normal(counts)
    # try discarding each distinct tile id in hand
    for tid in sorted({t for (t, _r) in hand_ids}):
        if counts[tid] == 0:
            continue
        counts[tid] -= 1
        sh = shanten_normal(counts)
        # compute ukeire after discard (for next draw)
        uke = ukeire_normal(counts, visible_cnts)
        uke_sum = int(sum(uke.values()))
        # small tie-breakers
        is_honor_single = (tid >= 27 and counts[tid] == 0)
        iso_bonus = 1 if is_honor_single else 0
        key = (sh, -uke_sum, -iso_bonus)  # minimize sh, maximize uke_sum (neg), prefer discarding honor single (neg)
        debug_rows.append((tid, sh, uke_sum, is_honor_single))
        if best_key is None or key < best_key:
            best_key = key
            best_tid = tid
        counts[tid] += 1

    dbg = {
        "base_shanten": base_sh,
        "candidates": [
            {"discard": tile_id_to_str(t), "sh": sh, "uke_sum": us, "honor_single": hs}
            for (t, sh, us, hs) in sorted(debug_rows, key=lambda x: (x[1], -x[2], -int(x[3])))
        ],
    }
    return best_tid, dbg


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
    detector = YoloSegOnnxDetector(
        onnx_path=DET_ONNX,
        input_size=768,
        is_xywh=True,              # set to None for auto-detect the first time if unsure
        class_names=ROI_CLASSES,
    )
    recognizer = CNNRecognizer(
        onnx_path=CNN_ONNX,
        classes_path=CNN_CLASSES,
        img_size=CNN_IMG_SIZE,
    )

    # Flags / knobs
    show_preview = True
    topmost = True
    conf_thr = CONF_THR
    iou_thr = IOU_THR

    # FPS HUD
    last, frames, fps = time.time(), 0, 0.0

    bc_policy = None
    if Path(BC_ONNX).exists():
        bc_policy = BCPolicyONNX(BC_ONNX)
        print(f"[INFO] Loaded BC policy: {BC_ONNX}")
    else:
        print(f"[WARN] BC policy not found at {BC_ONNX}; skipping BC suggestions.")

    while True:
        frame = cap.grab()
        if frame is None or frame.size == 0:
            raw = cv2.waitKey(1)
            if raw != -1 and (raw & 0xFF) in (27, ord('q')):
                break
            continue

        vis = frame.copy()

        # -------- DETECT (global) --------
        dets = detector.infer(frame, conf_thr=conf_thr, iou_thr=iou_thr, max_det=350)

        # Bucket by ROI class name
        buckets: Dict[str, List[dict]] = {name: [] for name in ROI_CLASSES}
        for d in dets:
            cname = d["cls_name"] if d["cls_name"] in buckets else ROI_CLASSES[d["cls_id"]]
            buckets[cname].append(d)

        # -------- CLASSIFY each ROI bucket --------
        roi_labels: Dict[str, List[Tuple[dict, Optional[str]]]] = {name: [] for name in ROI_CLASSES}
        for roi_name, items in buckets.items():
            if not items:
                continue
            # Sort bottom-row (hand_self / discards_bottom) left->right for nicer visuals
            if roi_name in ("tile_hand_self", "tile_discards_bottom", "tile_melds_self"):
                items = sorted(items, key=lambda d: (d["xyxy"][0] + d["xyxy"][2]) * 0.5)
            crops = [crop_square(frame, d["xyxy"], pad=4) for d in items]
            # rotation-robust classification
            outs = recognizer.classify_many_tta(crops, score_thresh=0.60, angles=[0, 90, 180, 270]) if crops else []
            labels = [lab if lab is not None else "unknown" for (lab, ang) in outs]
            for d, lab in zip(items, labels):
                roi_labels[roi_name].append((d, lab))

        # -------- Build GameState from recognized results --------
        # Hand tiles
        hand_list: List[Tuple[int, bool]] = []
        for d, lab in roi_labels["tile_hand_self"]:
            idr = label_to_id_red(lab)
            if idr is not None:
                hand_list.append(idr)

        # Basic visible counts (hand + our own melds + discards we currently parse: bottom only here)
        discards_map = {0: [], 1: [], 2: [], 3: []}
        # bottom discards (self’s river). If your perspective is different, remap seat indices later.
        for d, lab in roi_labels["tile_discards_bottom"]:
            idr = label_to_id_red(lab)
            if idr is not None:
                tid, _ = idr
                discards_map[0].append(
                    # minimal fields for now; you can fill riichi flags and turn indexes later
                    # Discard(tile=tid, from_seat=0, is_riichi=False, turn_index=len(discards_map[0]))
                    # Keeping it as simple tuple to avoid importing Discard dataclass; counts34_visible() sums fine without it.
                    # If you want Discard objects, import and construct them.
                    # For visibility counts we won’t need the extra metadata yet.
                    # To keep GameState API simple, we still pass the list but won't iterate objects here.
                    # We'll approximate counts by reading back from roi_labels below.
                    # (counts34_visible() in GameState currently assumes Discard objects; we’ll emulate below.)
                    tid  # we’ll handle counts manually
                )
        # Construct GameState (counts are derived; melds left empty for now)
        gs = GameState(
            hand_tiles=hand_list,
            melds_self=[],
            discards={0: [], 1: [], 2: [], 3: []},  # we will compute visible counts manually this frame
            dora_indicators=[],
        )

        # Visible counts = hand + detected discards (bottom) + open self melds (none yet)
        c_hand = gs.counts34_hand()
        c_vis = c_hand[:]  # start with hand
        for tid in discards_map[0]:
            c_vis[tid] += 1
        # (If/when you parse melds (self/opponents), add their tiles into c_vis here.)

        # -------- Decision: shanten + ukeire + discard suggestion --------
        rec_tile_id: Optional[int] = None
        reason = {}
        if len(hand_list) >= 10:  # avoid running with empty/partial hand
            # Compute shanten for display
            sh_all = shanten_all(c_hand)
            # Suggest discard
            rec_tile_id, dbg = choose_discard_simple(hand_list, c_vis)
            # Prepare HUD reason
            reason = {
                "sh_normal": sh_all["normal"],
                "sh_chiitoi": sh_all["chiitoi"],
                "sh_kokushi": sh_all["kokushi"],
                "rec_discard": tile_id_to_str(rec_tile_id) if rec_tile_id is not None else "-",
                "alts": [f'{row["discard"]}(sh{row["sh"]},U{row["uke_sum"]})' for row in dbg["candidates"][:3]],
            }
        # -------- Build BC features from live state & get top-3 (optional) --------
        bc_top3_display = []
        if bc_policy is not None and len(hand_list) > 0:
            # Build a minimal "log-style" sample so we can reuse build_bc_features()
            # NOTE: phys tile ids in logs are 0..135; our runtime only has 34-ids.
            # It's OK to synthesize phys ids as (tid * 4 + copy_idx), because build_bc_features() converts back via //4.
            def synth_phys_from_counts(counts34):
                out = []
                for tid, cnt in enumerate(counts34):
                    for k in range(cnt):
                        out.append(tid * 4 + k)
                return out

            # hand phys
            c_hand_now = [0]*34
            for tid, _red in hand_list:
                c_hand_now[tid] += 1
            hand_phys = synth_phys_from_counts(c_hand_now)

            # discards (we currently parse only bottom/self; extend as you add more)
            discards_by_seat = {0:[],1:[],2:[],3:[]}
            for tid in discards_map[0]:
                discards_by_seat[0].append(tid*4)  # one phys per visible copy is fine

            # minimal sample dict (fields used by build_bc_features)
            sample_for_bc = {
                "hand_tiles": hand_phys,
                "dora_indicators": [],  # fill if you detect them
                "valid_actions": [],    # not used to make features
                "action_idx": -1,       # unused
                "round_wind": 0,        # TODO: fill when you parse round/seat winds
                "num_honba": 0,
                "num_riichi": 0,
                "player_wind": 0,
                "remain_tiles": 70,     # rough; improves slightly if you track it
                "0": {"points": 25000, "melds": [], "discards": discards_by_seat[0], "tsumo_giri": [], "riichi": False},
                "1": {"points": 25000, "melds": [], "discards": discards_by_seat[1], "tsumo_giri": [], "riichi": False},
                "2": {"points": 25000, "melds": [], "discards": discards_by_seat[2], "tsumo_giri": [], "riichi": False},
                "3": {"points": 25000, "melds": [], "discards": discards_by_seat[3], "tsumo_giri": [], "riichi": False},
            }

            feat_vec, mask_vec, _label_dummy = build_bc_features(sample_for_bc)  # same layout as training
            # Replace mask with exact legal mask from our live hand (more reliable)
            mask_vec = np.zeros(34, np.float32)
            for tid, _r in hand_list:
                mask_vec[tid] = 1.0

            # ONNX inference
            logits = bc_policy(feat_vec)  # (34,)
            top3 = policy_topk(logits, mask_vec, k=3)  # [(tid, prob), ...]

            # For HUD
            bc_top3_display = [f"{tile_id_to_str(tid)}({prob:.2f})" for (tid, prob) in top3]

        # -------- Visualization --------
        # Draw boxes + labels
        for roi_name, items in roi_labels.items():
            for d, lab in items:
                x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
                roi_idx = ROI_CLASSES.index(roi_name)
                color = ((37 * (roi_idx + 3)) % 255, (97 * (roi_idx + 5)) % 255, (173 * (roi_idx + 7)) % 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                text = f"{lab} @{roi_name.split('_',1)[1]}"
                put_label(vis, text, (x1 + 4, max(16, y1 - 6)), color=(255, 255, 255) if lab != "unknown" else (180, 180, 180))

        # Highlight recommended discard on hand
        if rec_tile_id is not None and roi_labels["tile_hand_self"]:
            # choose the leftmost instance of that tile in hand boxes (best-effort)
            hand_boxes = [(d["xyxy"], lab) for (d, lab) in roi_labels["tile_hand_self"]]
            candidates = []
            for (xyxy, lab) in hand_boxes:
                idr = label_to_id_red(lab)
                if idr and idr[0] == rec_tile_id:
                    candidates.append(xyxy)
            if candidates:
                # pick leftmost
                xyxy = sorted(candidates, key=lambda b: (b[0] + b[2]) * 0.5)[0]
                x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
                put_label(vis, f"RECOMMEND: {tile_id_to_str(rec_tile_id)}", (x1, y2 + 18), color=(0, 255, 255))

        # -------- HUD --------
        frames += 1
        now = time.time()
        if now - last >= 1.0:
            fps = frames / (now - last)
            frames, last = 0, now

        hud = {
            "FPS": f"{fps:.1f}",
            "TopMost": topmost,
            "Det": Path(DET_ONNX).name,
            "Conf": f"{conf_thr:.2f}",
            "IoU": f"{iou_thr:.2f}",
            "CNN": Path(CNN_ONNX).name,
        }
        # add decision info if available
        if reason:
            hud.update({
                "Sh(n/c/k)": f'{reason["sh_normal"]}/{reason["sh_chiitoi"]}/{reason["sh_kokushi"]}',
                "Suggest": reason["rec_discard"],
                "Alt": " ".join(reason["alts"]),
            })
        if bc_top3_display:
            hud["BC top3"] = " ".join(bc_top3_display)
        vis = annotate_hud(vis, hud)

        # -------- Show --------
        if show_preview:
            cv2.imshow(win_title, vis)

        # -------- Keys --------
        raw = cv2.waitKey(1)
        if raw != -1:
            key = raw & 0xFF
            if key in (27, ord('q')):  # ESC/Q
                break
            elif key == ord('v'):
                show_preview = not show_preview
            elif key == ord('t'):
                topmost = not topmost
                set_window_topmost(hwnd_game, topmost)
            elif key == ord('s'):
                ts = int(time.time() * 1000)
                out = Path("data") / f"frame_{ts}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), vis)
                print(f"[INFO] Saved frame -> {out}")
            # Threshold tuning
            elif key == ord('+') or key == ord('='):
                conf_thr = min(0.90, conf_thr + 0.02)
                print(f"[DET] conf_thr -> {conf_thr:.2f}")
            elif key == ord('-') or key == ord('_'):
                conf_thr = max(0.05, conf_thr - 0.02)
                print(f"[DET] conf_thr -> {conf_thr:.2f}")
            elif key == ord(']'):
                iou_thr = min(0.90, iou_thr + 0.02)
                print(f"[DET] iou_thr -> {iou_thr:.2f}")
            elif key == ord('['):
                iou_thr = max(0.10, iou_thr - 0.02)
                print(f"[DET] iou_thr -> {iou_thr:.2f}")

    set_window_topmost(hwnd_game, False)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
