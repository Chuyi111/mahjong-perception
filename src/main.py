# src/main.py
import time
import atexit
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from perception.utils.window_finder import (
    set_process_dpi_aware,
    find_window_by_titles,
    set_window_topmost,
)
from perception.capture.screen_capture import ScreenCapturer, annotate_hud
from perception.utils.capture_exclude import set_window_exclude_from_capture

# NEW: YOLOv8-seg ONNX detector wrapper (ROI-aware)
from perception.detectors.yolo_seg_onnx import YoloSegOnnxDetector
# EXISTING: your tile CNN recognizer
from perception.tiles.cnn_recognizer import CNNRecognizer


# -------------------- Config --------------------
# Path to your exported YOLOv8-seg ONNX model (ROI-aware detector)
DET_ONNX = "models/detectors/tiles_roi_yolov8n_seg.onnx"

# Class order MUST match your training YAML 'names'
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

# Tile classifier (your CNN)
CNN_ONNX = "models/tile_cnn/tile_cnn.onnx"
CNN_CLASSES = "models/tile_cnn/classes.json"
CNN_IMG_SIZE = 128

# Detector thresholds (hotkeys to change at runtime)
CONF_THR = 0.35
IOU_THR = 0.50

# Window title candidates
TITLE_HINTS = ["雀魂麻將", "雀魂", "Mahjong Soul"]


# -------------------- Helpers --------------------
def get_cv_window_hwnd(title: str) -> Optional[int]:
    try:
        import win32gui
        hwnd = win32gui.FindWindow(None, title)
        return int(hwnd) if hwnd else None
    except Exception:
        return None

def put_label(img, text, xy, color=(255, 255, 255)):
    x, y = xy
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

def crop_square(img: np.ndarray, xyxy, pad: int = 3) -> np.ndarray:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img.shape[1] - 1, x2 + pad)
    y2 = min(img.shape[0] - 1, y2 + pad)
    return img[y1:y2 + 1, x1:x2 + 1]


# -------------------- Main --------------------
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
    win_title = "Mahjong Perception (Detector Mode)"
    cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_title, 50, 50)
    hwnd_prev = get_cv_window_hwnd(win_title)
    if hwnd_prev:
        set_window_exclude_from_capture(hwnd_prev, True)

    # Instantiate models
    detector = YoloSegOnnxDetector(
        onnx_path=DET_ONNX,
        input_size=768,       # match your export
        is_xywh=True,         # set False if your ONNX outputs xyxy already
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

    while True:
        frame = cap.grab()
        if frame is None or frame.size == 0:
            raw = cv2.waitKey(1)
            if raw != -1 and (raw & 0xFF) in (27, ord('q')):
                break
            continue

        vis = frame.copy()

        # -------- DETECT (global) --------
        dets = detector.infer(frame, conf_thr=conf_thr, iou_thr=iou_thr, max_det=300)

        # Bucket by ROI class
        buckets: Dict[str, List[dict]] = {name: [] for name in ROI_CLASSES}
        for d in dets:
            name = d["cls_name"] if d["cls_name"] in buckets else ROI_CLASSES[d["cls_id"]]
            buckets[name].append(d)

        # -------- CLASSIFY each crop --------
        for roi_name, items in buckets.items():
            if not items:
                continue
            crops = [crop_square(frame, d["xyxy"], pad=4) for d in items]
            outs = recognizer.classify_many(crops, score_thresh=0.60) if crops else []
            labels = [c if c is not None else "unknown" for c in outs]

            # Draw
            for d, lab in zip(items, labels):
                x1, y1, x2, y2 = [int(round(v)) for v in d["xyxy"]]
                # Box color per ROI (simple hash)
                roi_idx = ROI_CLASSES.index(roi_name)
                color = ((37 * (roi_idx + 3)) % 255, (97 * (roi_idx + 5)) % 255, (173 * (roi_idx + 7)) % 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                text = f"{lab} @{roi_name.split('_',1)[1]}"
                put_label(vis, text, (x1 + 4, y1 - 6), color=(255, 255, 255) if lab != "unknown" else (180, 180, 180))

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
        vis = annotate_hud(vis, hud)

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
