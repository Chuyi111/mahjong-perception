from __future__ import annotations
import json, atexit, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import win32gui

from perception.utils.window_finder import set_process_dpi_aware, find_window_by_titles, set_window_topmost
from perception.capture.screen_capture import ScreenCapturer
from perception.config.io import load_config

ROI_QUADS_PATH = Path("configs/roi_quads.json")

ROI_ORDER = [
    "hand_self",
    "melds_self",
    "discards_bottom",
    "discards_top",
    "discards_left",
    "discards_right",
    "melds_top",
    "melds_left",
    "melds_right",
]

def load_quads() -> Dict[str, dict]:
    if ROI_QUADS_PATH.exists():
        return json.loads(ROI_QUADS_PATH.read_text(encoding="utf-8"))
    return {}

def save_quads(q: Dict[str, dict]) -> None:
    ROI_QUADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROI_QUADS_PATH.write_text(json.dumps(q, indent=2, ensure_ascii=False), encoding="utf-8")

def main():
    set_process_dpi_aware()
    cfg = load_config("configs/mahjongsoul.json")

    found = (find_window_by_titles(getattr(cfg.capture, "window_title_hint", None))
             or find_window_by_titles("雀魂麻將")
             or find_window_by_titles("雀魂")
             or find_window_by_titles("Mahjong Soul"))
    if not found:
        print("Mahjong Soul window not found.")
        return

    hwnd, rect = found
    l,t,r,b = rect
    cap = ScreenCapturer(rect)
    set_window_topmost(hwnd, True)
    atexit.register(lambda: set_window_topmost(hwnd, False))

    win = "Calibrate ROI Quads (1-9 select ROI; click TL->TR->BR->BL; Enter save; C clear; Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win, 50, 50)

    roi_quads = load_quads()
    curr_idx = 0
    curr_name = ROI_ORDER[curr_idx]
    pts: List[Tuple[int,int]] = []

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < 4:
                pts.append((x, y))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        frame = cap.grab()
        vis = frame.copy()

        # draw existing quads
        for k, v in roi_quads.items():
            q = v.get("quad")
            if q and len(q)==4:
                q = [(int(px),int(py)) for px,py in q]
                cv2.polylines(vis, [np.array(q, dtype=np.int32)], True, (0,255,255), 2)
                cx = sum(p[0] for p in q)//4; cy = sum(p[1] for p in q)//4
                cv2.putText(vis, k, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, k, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        # draw current points
        for i, (px,py) in enumerate(pts):
            cv2.circle(vis, (px,py), 6, (0,165,255), -1)
            cv2.putText(vis, str(i+1), (px+6,py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, str(i+1), (px+6,py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)

        msg = f"[{curr_idx+1}] {curr_name} | 1-9 select ROI | Click TL->TR->BR->BL | Enter=save | C=clear | Q=quit"
        cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(win, vis)
        raw = cv2.waitKey(1)
        if raw == -1: continue
        key = raw & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('c'):
            pts = []
        elif key == 13:  # Enter -> save
            if len(pts) == 4:
                # choose output size: bbox of the selected quad
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                out_w = max(32, int(max(xs) - min(xs)))
                out_h = max(32, int(max(ys) - min(ys)))
                roi_quads[curr_name] = {"quad": pts, "size": [out_w, out_h]}
                save_quads(roi_quads)
                print(f"[saved] {curr_name} -> quad={pts}, size={(out_w,out_h)}")
                pts = []
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(ROI_ORDER):
                curr_idx = idx
                curr_name = ROI_ORDER[curr_idx]
                pts = []
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np
    main()
