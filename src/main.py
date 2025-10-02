# src/main.py
import time
import atexit
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import win32gui

from perception.utils.window_finder import (
    set_process_dpi_aware,
    find_window_by_titles,
    set_window_topmost,
)
from perception.capture.screen_capture import ScreenCapturer, annotate_hud
from perception.config.io import load_config
from perception.config.model import Rect
from perception.preprocess.roi import preprocess_rois
from perception.preprocess.orient import rotate_region_by_name
from perception.preprocess.deskew import mild_deskew_by_region
from perception.tiles.tilers import tile_hand_self, tile_discards, tile_melds
from perception.tiles.recognizer import TemplateRecognizer
from perception.utils.capture_exclude import set_window_exclude_from_capture


# -------------------- helpers --------------------
def scale_rect(rect: Rect, sx: float, sy: float) -> Rect:
    l, t, w, h = rect
    return (int(round(l * sx)), int(round(t * sy)), int(round(w * sx)), int(round(h * sy)))

def maybe_scale_rois(rois: Dict[str, Optional[Rect]], base_w: int, base_h: int,
                     cur_w: int, cur_h: int) -> Dict[str, Optional[Rect]]:
    if base_w <= 0 or base_h <= 0 or (base_w == cur_w and base_h == cur_h):
        return rois
    sx, sy = cur_w / base_w, cur_h / base_h
    out: Dict[str, Optional[Rect]] = {}
    for k, v in rois.items():
        out[k] = None if v is None else scale_rect(tuple(v), sx, sy)
    return out

def ensure_dir(p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def get_cv_window_hwnd(title: str) -> Optional[int]:
    hwnd = win32gui.FindWindow(None, title)
    return int(hwnd) if hwnd else None

def put_label(img, text, xy, color=(255, 255, 255)):
    x, y = xy
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


# -------------------- main --------------------
def main():
    set_process_dpi_aware()

    # Load config (ROIs + calibration baseline size)
    cfg_path = "configs/mahjongsoul.json"
    cfg = load_config(cfg_path)

    # Find the Mahjong Soul window (localized title first)
    found = (
        find_window_by_titles(cfg.capture.window_title_hint)
        or find_window_by_titles("雀魂麻將")
        or find_window_by_titles("雀魂")
        or find_window_by_titles("Mahjong Soul")
    )
    if not found:
        print("Mahjong Soul window not found. Open it (not minimized) and try again.")
        return

    hwnd_game, rect = found
    l, t, r, b = rect
    cur_w, cur_h = r - l, b - t
    print(f"[INFO] Capturing client area at {rect} (size {cur_w}x{cur_h})")

    # Keep the game on top to avoid occlusion by other windows
    set_window_topmost(hwnd_game, True)
    atexit.register(lambda: set_window_topmost(hwnd_game, False))

    # Prepare capture
    cap = ScreenCapturer(rect)

    # Prepare ROIs (scaled to current size if needed)
    rois_dict: Dict[str, Optional[Rect]] = cfg.rois.model_dump()
    rois_dict = maybe_scale_rois(rois_dict, cfg.capture.baseline_width, cfg.capture.baseline_height, cur_w, cur_h)

    # Initialize recognizer (loads templates under assets/templates/**)
    recognizer = TemplateRecognizer(template_root="assets/templates")

    # Open preview and exclude it from capture to avoid mirror recursion
    win_title = "Mahjong Perception - Capture"
    cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_title, 50, 50)

    hwnd_preview = get_cv_window_hwnd(win_title)
    if hwnd_preview:
        ok = set_window_exclude_from_capture(hwnd_preview, True)
        print(f"[INFO] Exclude preview from capture: {ok}")

    # Flags
    show_preview = True
    show_boxes = True
    topmost = True

    # FPS
    last, frames, fps = time.time(), 0, 0.0

    while True:
        frame = cap.grab()  # BGR in client coordinates
        if frame is None or frame.size == 0:
            raw = cv2.waitKey(1)
            if raw != -1 and (raw & 0xFF) in (27, ord('q')):
                break
            continue

        # Start building visualization early so it's always defined
        vis = frame.copy()

        # Crop + normalize bands for each named ROI and rotate to self orientation
        crops = preprocess_rois(frame, rois_dict)
        for k, v in list(crops.items()):
            if v is None:
                continue
            crops[k] = rotate_region_by_name(v, k)

        # Draw ROI boxes if enabled
        if show_boxes:
            for name, rxywh in rois_dict.items():
                if rxywh is None:
                    continue
                l0, t0, w0, h0 = rxywh
                cv2.rectangle(vis, (l0, t0), (l0 + w0, t0 + h0), (0, 255, 255), 2)
                cv2.putText(vis, name, (l0 + 4, t0 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, name, (l0 + 4, t0 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # ---------------- Recognition with "unknown" fallback ----------------
        # Collect labels to draw: (text, (x,y), color)
        labels_to_draw: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []

        # Hand (no deskew)
        if crops.get("hand_self") is not None:
            tiles = tile_hand_self(crops["hand_self"])
            preds = recognizer.classify_many(tiles, score_thresh=0.40)  # -> list[Optional[str]]
            band = rois_dict.get("hand_self")
            if band:
                l0, t0, w0, h0 = band
                step = w0 // max(1, len(preds))
                for i, cls in enumerate(preds):
                    label = cls if cls is not None else "unknown"
                    color = (180, 180, 180) if cls is None else (255, 255, 255)
                    x = l0 + i * step + 6
                    y = t0 + h0 - 8
                    labels_to_draw.append((label, (x, y), color))

        # Discards (deskew per region)
        for key in ("discards_top", "discards_right", "discards_bottom", "discards_left"):
            if crops.get(key) is None:
                continue
            tiles = tile_discards(crops[key])
            tiles = [mild_deskew_by_region(t, key) for t in tiles]
            preds = recognizer.classify_many(tiles, score_thresh=0.40)
            band = rois_dict.get(key)
            if band:
                l0, t0, w0, h0 = band
                cols, rows = 6, 3
                cw = w0 // cols
                ch = h0 // rows
                for idx, cls in enumerate(preds):
                    label = cls if cls is not None else "unknown"
                    color = (180, 180, 180) if cls is None else (255, 255, 255)
                    cx = idx % cols
                    cy = idx // cols
                    x = l0 + cx * cw + 4
                    y = t0 + (cy + 1) * ch - 6
                    labels_to_draw.append((label, (x, y), color))

        # Melds (deskew per region; coarse slots)
        for key in ("melds_self", "melds_top", "melds_right", "melds_left"):
            if crops.get(key) is None:
                continue
            tiles = tile_melds(crops[key])
            tiles = [mild_deskew_by_region(t, key) for t in tiles]
            preds = recognizer.classify_many(tiles, score_thresh=0.4)
            band = rois_dict.get(key)
            if band:
                l0, t0, w0, h0 = band
                n = max(1, len(preds))
                step = w0 // n
                for i, cls in enumerate(preds):
                    label = cls if cls is not None else "unknown"
                    color = (180, 180, 180) if cls is None else (255, 255, 255)
                    x = l0 + i * step + 4
                    y = t0 + h0 // 2
                    labels_to_draw.append((label, (x, y), color))

        # Draw labels on vis
        for text, (x, y), color in labels_to_draw:
            put_label(vis, text, (x, y), color=color)

        # FPS / HUD
        frames += 1
        now = time.time()
        if now - last >= 1.0:
            fps = frames / (now - last)
            frames, last = 0, now

        hud = {
            "FPS": f"{fps:.1f}",
            "Size": f"{cur_w}x{cur_h}",
            "TopMost": topmost,
            "ROIs": sum(1 for v in rois_dict.values() if v is not None),
        }
        vis = annotate_hud(vis, hud)

        if show_preview:
            cv2.imshow(win_title, vis)

        # ---- input handling (safe) ----
        raw = cv2.waitKey(1)
        if raw != -1:
            key = raw & 0xFF
            if key in (27, ord('q')):  # ESC or Q
                break
            elif key == ord('s'):
                ts = int(now * 1000)
                # Save current preprocessed ROI crops for debugging
                for k, img in crops.items():
                    if img is None:
                        continue
                    ensure_dir(f"data/{k}_{ts}.png")
                    cv2.imwrite(f"data/{k}_{ts}.png", img)
                print(f"[INFO] Saved ROI crops at ts {ts}")
            elif key == ord('v'):
                show_preview = not show_preview
            elif key == ord('b'):
                show_boxes = not show_boxes
            elif key == ord('t'):
                topmost = not topmost
                set_window_topmost(hwnd_game, topmost)
            elif key == ord('r'):
                # Reload config on the fly (useful if you re-calibrated)
                cfg_new = load_config(cfg_path)
                rd = cfg_new.rois.model_dump()
                rois_dict = maybe_scale_rois(rd, cfg_new.capture.baseline_width, cfg_new.capture.baseline_height, cur_w, cur_h)
                print("[INFO] Reloaded ROIs from config.")

    # cleanup
    set_window_topmost(hwnd_game, False)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
