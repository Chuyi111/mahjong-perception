#import sys, pathlib
#sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import time
from typing import Dict, Tuple, Optional, List
import cv2
import numpy as np
import win32gui

from perception.utils.window_finder import (
    set_process_dpi_aware,
    find_window_by_titles,
    get_client_rect_in_screen,
)

from perception.utils.capture_exclude import set_window_exclude_from_capture
from perception.capture.screen_capture import ScreenCapturer, annotate_hud
from perception.config.model import AppConfig, ROISet, Rect
from perception.config.io import load_config, save_config

ROI_NAMES = [
    "hand_self",
    "discards_top", "discards_right", "discards_bottom", "discards_left",
    "melds_self", "melds_top", "melds_right", "melds_left",
    "dora", "winds_round", "points_board",
    "anchor_tl", "anchor_br",
]


COLORS = {}
def color_for(name: str) -> Tuple[int,int,int]:
    if name not in COLORS:
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        COLORS[name] = tuple(int(x) for x in rng.integers(80, 255, size=3))  # BGR
    return COLORS[name]  # type: ignore

class DragState:
    def __init__(self):
        self.active_name: Optional[str] = None
        self.dragging: bool = False
        self.start: Tuple[int,int] = (0,0)
        self.current: Tuple[int,int] = (0,0)

def rect_from_points(p0: Tuple[int,int], p1: Tuple[int,int]) -> Rect:
    l = min(p0[0], p1[0]); t = min(p0[1], p1[1])
    r = max(p0[0], p1[0]); b = max(p0[1], p1[1])
    return (l, t, r - l, b - t)

def draw_rois(vis: np.ndarray, rois: Dict[str, Optional[Rect]], alpha: float = 0.25) -> np.ndarray:
    overlay = vis.copy()
    for name, rect in rois.items():
        if rect is None: continue
        l, t, w, h = rect; r, b = l + w, t + h
        c = color_for(name)
        cv2.rectangle(overlay, (l, t), (r, b), c, thickness=-1)
        cv2.rectangle(overlay, (l, t), (r, b), c, thickness=2)
        cv2.putText(overlay, name, (l+6, t+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, name, (l+6, t+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

def main():
    set_process_dpi_aware()

    cfg_path = "configs/mahjongsoul.json"
    cfg = load_config(cfg_path)

    found = (
        find_window_by_titles(cfg.capture.window_title_hint)
        or find_window_by_titles("雀魂麻將")
        or find_window_by_titles("雀魂")
        or find_window_by_titles("Mahjong Soul")
    )
    if not found:
        print("Mahjong Soul window not found (not minimized). Open it and try again.")
        return
    hwnd, rect = found
    l, t, r, b = rect
    client_w, client_h = r - l, b - t

    # save baseline client size for later scaling
    cfg.capture.baseline_width = client_w
    cfg.capture.baseline_height = client_h

    cap = ScreenCapturer(rect)
    window = "ROI Calibrator (drag to set; keys: 1-0 to select; Enter save; Del clear; A/Z opacity; H help)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    hwnd_preview = win32gui.FindWindow(None, window)  # 'window' is your calibrator title
    if hwnd_preview:
        ok = set_window_exclude_from_capture(hwnd_preview, True)
        print(f"[Calibrator] Exclude preview from capture: {ok}")
    cv2.resizeWindow(window, min(1280, client_w), min(800, client_h))

    rois: Dict[str, Optional[Rect]] = cfg.rois.model_dump()
    drag = DragState()
    opacity = 0.28
    pause = False
    frozen = None

    def on_mouse(event, x, y, flags, param):
        nonlocal drag, rois
        if event == cv2.EVENT_LBUTTONDOWN and drag.active_name:
            drag.dragging = True
            drag.start = (x, y); drag.current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag.dragging:
            drag.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag.dragging and drag.active_name:
            rect_xywh = rect_from_points(drag.start, drag.current)
            rois[drag.active_name] = rect_xywh
            drag.dragging = False

    cv2.setMouseCallback(window, on_mouse)

    def set_active_by_index(idx: int):
        if 0 <= idx < len(ROI_NAMES):
            drag.active_name = ROI_NAMES[idx]
            print(f"Active ROI: {drag.active_name}")

    def select_next():
        if drag.active_name is None:
            set_active_by_index(0); return
        i = ROI_NAMES.index(drag.active_name)
        set_active_by_index((i + 1) % len(ROI_NAMES))

    def select_prev():
        if drag.active_name is None:
            set_active_by_index(0); return
        i = ROI_NAMES.index(drag.active_name)
        set_active_by_index((i - 1) % len(ROI_NAMES))

    set_active_by_index(0)

    show_help = True
    last, frames, fps = time.time(), 0, 0.0

    help_lines = [
        "Controls:",
        "  1..9,0 = select first 10 ROIs",
        "  [ / ]  = prev / next ROI",
        "  Drag LMB = draw/adjust ROI",
        "  Backspace/Delete = clear active ROI",
        "  A / Z = opacity up / down",
        "  Space = pause, F = freeze/unfreeze current frame",
        "  S = save snapshot to data/",
        "  Enter = SAVE to configs/mahjongsoul.json",
        "  H = toggle help, Q or ESC = quit",
    ]


    while True:
        if not pause:
            frame = cap.grab()
        else:
            frame = cap.grab() if frozen is None else frozen.copy()

        frames += 1
        now = time.time()
        if now - last >= 1.0:
            fps = frames / (now - last)
            frames, last = 0, now

        vis = frame.copy()

        # If currently dragging, draw the transient rect
        if drag.dragging and drag.active_name:
            l0, t0 = drag.start; l1, t1 = drag.current
            tmp = rect_from_points((l0,t0), (l1,t1))
            tmp_map = {drag.active_name: tmp}
            vis = draw_rois(vis, tmp_map, alpha=opacity)

        # Draw existing ROIs
        vis = draw_rois(vis, rois, alpha=opacity)

        hud = {
            "FPS": f"{fps:.1f}",
            "Active": drag.active_name or "(none)",
            "Opacity": f"{opacity:.2f}",
            "Paused": pause,
            "Frozen": frozen is not None,
        }
        vis = annotate_hud(vis, hud)

        if show_help:
            y = 28 * 6
            for line in help_lines:
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                y += 24

        cv2.setWindowTitle(window, f"ROI Calibrator – selecting: {drag.active_name}  |  size: {client_w}x{client_h}")
        cv2.imshow(window, vis)

        raw = cv2.waitKey(1)
        if raw != -1:
            key = raw & 0xFF
            if key in (27, ):  # ESC
                break
            elif key == ord('q'):  # optional quit
                break
            elif key == 13:  # Enter = save
                cfg.rois = ROISet(**rois)
                save_config(cfg, cfg_path)
                print(f"Saved {cfg_path}")
            elif key == ord('h'):
                show_help = not show_help
            elif key == ord(' '):
                pause = not pause
                if not pause: frozen = None
            elif key == ord('f'):
                frozen = frame.copy() if frozen is None else None
            elif key == ord('s'):
                ts = int(now * 1000)
                path = f"data/calib_{ts}.png"
                cv2.imwrite(path, frame)
                print(f"Saved {path}")
            elif key == ord('a'):
                opacity = min(0.9, opacity + 0.05)
            elif key == ord('z'):
                opacity = max(0.05, opacity - 0.05)
            elif key == ord(']'):  # next ROI
                select_next()
            elif key == ord('['):  # previous ROI
                select_prev()
            elif ord('1') <= key <= ord('9'):
                set_active_by_index(key - ord('1'))          # 1..9 -> 0..8
            elif key == ord('0'):
                set_active_by_index(9)                        # 0 -> 9


    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
