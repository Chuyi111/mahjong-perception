# Click-to-crop individual tiles and save directly into assets/templates/<class>/.
# Hotkeys:
#   Mouse:  Drag LMB to select a tile region; release to lock crop
#   Rotate: [ = 90° CCW, ] = 90° CW, \ = 180°
#   Suits : m / p / s  (man/pin/sou). Then press 1..9 to choose rank. Toggle red-5 with r (only when rank==5).
#   Honors: E S W N P F C  (East, South, West, North, White, Green, Red) — saves immediately.
#   Save  : Enter (saves current crop with current suit+rank [+r]).
#   Misc  : H toggle help, Q/ESC quit, T toggle TopMost for game, C clear current crop (and selection)
#
# Output path example:
#   assets/templates/5mr/1727740000000.png
from __future__ import annotations
import time, atexit
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import win32gui

from perception.utils.window_finder import (
    set_process_dpi_aware,
    find_window_by_titles,
    set_window_topmost,
)
from perception.capture.screen_capture import ScreenCapturer, annotate_hud
from perception.tiles.normalize import normalize_tile
from perception.utils.capture_exclude import set_window_exclude_from_capture

# --------- small helpers ----------
def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def get_cv_window_hwnd(title: str) -> Optional[int]:
    hwnd = win32gui.FindWindow(None, title)
    return int(hwnd) if hwnd else None

def rotate_img(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if mode == "180":
        return cv2.rotate(img, cv2.ROTATE_180)
    return img

# --------- main tool ----------
class Drag:
    def __init__(self):
        self.dragging = False
        self.p0: Tuple[int,int] = (0,0)
        self.p1: Tuple[int,int] = (0,0)

    def rect(self) -> Optional[Tuple[int,int,int,int]]:
        if not self.dragging and self.p0 != self.p1:
            x0,y0 = self.p0; x1,y1 = self.p1
            l, t = min(x0,x1), min(y0,y1)
            r, b = max(x0,x1), max(y0,y1)
            if r-l > 4 and b-t > 4:
                return (l, t, r-l, b-t)
        return None

def main():
    set_process_dpi_aware()

    # 1) Find game window
    found = (find_window_by_titles("雀魂麻將")
             or find_window_by_titles("雀魂")
             or find_window_by_titles("Mahjong Soul"))
    if not found:
        print("Mahjong Soul window not found. Open it (not minimized).")
        return
    hwnd_game, rect = found
    l, t, r, b = rect
    print(f"[INFO] Capturing client area: {rect} (size {r-l}x{b-t})")

    # 2) Keep game on top to avoid occlusion; restore on exit
    set_window_topmost(hwnd_game, True)
    atexit.register(lambda: set_window_topmost(hwnd_game, False))

    # 3) Setup capture + windows
    cap = ScreenCapturer(rect)

    win_main = "Tile Capture (drag to crop | H help)"
    win_crop = "Current Tile"
    cv2.namedWindow(win_main, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(win_crop, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_main, 50, 50)
    cv2.moveWindow(win_crop, 50, 520)

    # Exclude these preview windows from being captured (prevents mirroring)
    for title in (win_main, win_crop):
        hwnd_prev = get_cv_window_hwnd(title)
        if hwnd_prev:
            ok = set_window_exclude_from_capture(hwnd_prev, True)
            print(f"[INFO] Exclude '{title}' from capture: {ok}")

    # 4) State
    drag = Drag()
    have_crop = False
    crop_img: Optional[np.ndarray] = None  # tile candidate (raw, not normalized)
    suit: Optional[str] = None             # 'm'/'p'/'s'
    rank: Optional[int] = None             # 1..9
    red5: bool = False                     # True -> '5mr/5pr/5sr'
    topmost = True
    show_help = True

    def on_mouse(event, x, y, flags, param):
        nonlocal drag, crop_img, have_crop
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.dragging = True
            drag.p0 = (x, y)
            drag.p1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag.dragging:
            drag.p1 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag.dragging = False
            frame = param["frame"]
            rect_xywh = drag.rect()
            if rect_xywh is not None:
                lx, ty, ww, hh = rect_xywh
                crop = frame[ty:ty+hh, lx:lx+ww].copy()
                crop_img = crop
                have_crop = True
            else:
                crop_img = None
                have_crop = False

    # Attach mouse to main window
    cv2.setMouseCallback(win_main, on_mouse, param={"frame": None})

    # 5) Loop
    last, frames, fps = time.time(), 0, 0.0
    while True:
        frame = cap.grab()
        if frame is None or frame.size == 0:
            raw = cv2.waitKey(1)
            if raw == 27: break
            continue

        # update callback param so the mouse handler can crop from *this* frame
        cv2.setMouseCallback(win_main, on_mouse, param={"frame": frame})

        vis = frame.copy()
        # draw live drag rect
        if drag.dragging:
            x0,y0 = drag.p0; x1,y1 = drag.p1
            l0, t0, r0, b0 = min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)
            cv2.rectangle(vis, (l0,t0), (r0,b0), (0,255,255), 2)

        # HUD
        frames += 1
        now = time.time()
        if now - last >= 1.0:
            fps = frames / (now - last)
            frames, last = 0, now

        status = []
        status.append(f"suit={suit or '-'}")
        status.append(f"rank={rank or '-'}")
        status.append(f"red5={'Y' if red5 else 'N'}")
        status.append(f"crop={'Y' if have_crop else 'N'}")
        status.append("TopMost=" + ("Y" if topmost else "N"))
        vis = annotate_hud(vis, {"FPS": f"{fps:.1f}", " ".join(status): ""})

        # Help overlay
        if show_help:
            lines = [
                "Drag LMB to select a tile region; release to lock.",
                "Rotate: [ = 90° CCW, ] = 90° CW, \\ = 180°",
                "Suits: m/p/s then 1..9 ; toggle red5: r (only for 5)",
                "Honors: E S W N P F C -> saves immediately",
                "Enter: save current crop with suit+rank(+r) to templates",
                "C: clear crop/selection; T: toggle TopMost; H: toggle help; Q/Esc: quit",
            ]
            y = 26*5
            for line in lines:
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                y += 24

        # Show main and crop preview
        cv2.imshow(win_main, vis)

        if have_crop and crop_img is not None:
            cv2.imshow(win_crop, crop_img)
        else:
            # show an empty placeholder
            blank = np.zeros((96,64,3), dtype=np.uint8)
            cv2.putText(blank, "No crop", (3,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow(win_crop, blank)

        # Key handling
        raw = cv2.waitKey(1)
        if raw != -1:
            key = raw & 0xFF
            if key in (27, ord('q')):  # ESC/Q
                break

            elif key == ord('h'):
                show_help = not show_help

            elif key == ord('t'):
                topmost = not topmost
                set_window_topmost(hwnd_game, topmost)

            elif key == ord('c'):
                # clear crop and selection
                have_crop = False
                crop_img = None
                suit = None
                rank = None
                red5 = False

            elif key in (ord('['), ord(']'), ord('\\')):
                # rotate crop if present
                if have_crop and crop_img is not None:
                    if key == ord('['):
                        crop_img = rotate_img(crop_img, "ccw")
                    elif key == ord(']'):
                        crop_img = rotate_img(crop_img, "cw")
                    else:
                        crop_img = rotate_img(crop_img, "180")

            # Honors (UPPERCASE) => save immediately
            elif key in (ord('E'), ord('S'), ord('W'), ord('N'), ord('P'), ord('F'), ord('C')):
                if have_crop and crop_img is not None:
                    cls = chr(key)
                    out = normalize_tile(crop_img)
                    out_dir = Path(f"assets/templates/{cls}")
                    ensure_dir(out_dir)
                    ts = int(time.time()*1000)
                    cv2.imwrite(str(out_dir / f"{ts}.png"), out)
                    print(f"[saved] {cls} -> {out_dir}/{ts}.png")

            # Suits (lowercase)
            elif key in (ord('m'), ord('p'), ord('s')):
                suit = chr(key)  # 'm','p','s'
                # if switching suit and rank==5, keep red5 state; else reset red flag
                if rank != 5:
                    red5 = False

            # Rank digits 1..9 (for suits). Do not save yet; wait for Enter so you can toggle red5.
            elif ord('1') <= key <= ord('9'):
                rank = key - ord('0')
                if rank != 5:
                    red5 = False  # red flag only relevant for 5

            # Toggle red-5 (only if current rank==5)
            elif key == ord('r'):
                if rank == 5:
                    red5 = not red5

            # Save for suited tiles
            elif key == 13:  # Enter
                if have_crop and crop_img is not None and suit is not None and rank is not None:
                    if suit in ('m','p','s') and 1 <= rank <= 9:
                        if rank == 5 and red5:
                            cls = f"5{suit}r"
                        else:
                            cls = f"{rank}{suit}"
                        out = normalize_tile(crop_img)
                        out_dir = Path(f"assets/templates/{cls}")
                        ensure_dir(out_dir)
                        ts = int(time.time()*1000)
                        cv2.imwrite(str(out_dir / f"{ts}.png"), out)
                        print(f"[saved] {cls} -> {out_dir}/{ts}.png")
                # keep crop so you can save the same crop into another class if you want

    # cleanup
    set_window_topmost(hwnd_game, False)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
