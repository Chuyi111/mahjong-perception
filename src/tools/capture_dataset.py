# Run this while your game is open and ROIs calibrated.
# Press 'h' to save hand tiles, '1/2/3/4' to save discards (top/right/bottom/left),
# 'm' to save meld slots for all seats. It will save cropped tiles under data/raw/<region>/...
from __future__ import annotations
import time, os
from pathlib import Path
import cv2

from perception.utils.window_finder import set_process_dpi_aware, find_window_by_titles
from perception.capture.screen_capture import ScreenCapturer
from perception.config.io import load_config
from perception.preprocess.roi import preprocess_rois
from perception.preprocess.orient import rotate_region_by_name
from perception.tiles.tilers import tile_hand_self, tile_discards, tile_melds

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def main():
    set_process_dpi_aware()
    cfg = load_config("configs/mahjongsoul.json")
    found = (find_window_by_titles(cfg.capture.window_title_hint)
             or find_window_by_titles("雀魂麻將")
             or find_window_by_titles("雀魂")
             or find_window_by_titles("Mahjong Soul"))
    if not found:
        print("Mahjong Soul window not found.")
        return
    hwnd, rect = found
    cap = ScreenCapturer(rect)
    rois = cfg.rois.model_dump()

    win = "Dataset Capture"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    while True:
        frame = cap.grab()
        crops = preprocess_rois(frame, rois)
        for k in list(crops.keys()):
            if crops[k] is not None:
                crops[k] = rotate_region_by_name(crops[k], k)

        vis = frame.copy()
        cv2.imshow(win, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): break

        ts = int(time.time()*1000)

        if key == ord('h') and crops.get("hand_self") is not None:
            tiles = tile_hand_self(crops["hand_self"])
            base = Path(f"data/raw/hand/{ts}")
            ensure_dir(base)
            for i, im in enumerate(tiles):
                cv2.imwrite(str(base / f"{i:02d}.png"), im)
            print(f"[saved] hand tiles -> {base}")

        elif key == ord('1') and crops.get("discards_top") is not None:
            tiles = tile_discards(crops["discards_top"])
            base = Path(f"data/raw/discards_top/{ts}"); ensure_dir(base)
            for i, im in enumerate(tiles): cv2.imwrite(str(base / f"{i:02d}.png"), im)
            print(f"[saved] discards_top -> {base}")

        elif key == ord('2') and crops.get("discards_right") is not None:
            tiles = tile_discards(crops["discards_right"])
            base = Path(f"data/raw/discards_right/{ts}"); ensure_dir(base)
            for i, im in enumerate(tiles): cv2.imwrite(str(base / f"{i:02d}.png"), im)
            print(f"[saved] discards_right -> {base}")

        elif key == ord('3') and crops.get("discards_bottom") is not None:
            tiles = tile_discards(crops["discards_bottom"])
            base = Path(f"data/raw/discards_bottom/{ts}"); ensure_dir(base)
            for i, im in enumerate(tiles): cv2.imwrite(str(base / f"{i:02d}.png"), im)
            print(f"[saved] discards_bottom -> {base}")

        elif key == ord('4') and crops.get("discards_left") is not None:
            tiles = tile_discards(crops["discards_left"])
            base = Path(f"data/raw/discards_left/{ts}"); ensure_dir(base)
            for i, im in enumerate(tiles): cv2.imwrite(str(base / f"{i:02d}.png"), im)
            print(f"[saved] discards_left -> {base}")

        elif key == ord('m'):
            for seat in ("self","top","right","left"):
                keyname = f"melds_{'self' if seat=='self' else seat}"
                if crops.get(keyname) is None: continue
                tiles = tile_melds(crops[keyname])
                base = Path(f"data/raw/{keyname}/{ts}"); ensure_dir(base)
                for i, im in enumerate(tiles): cv2.imwrite(str(base / f"{i:02d}.png"), im)
                print(f"[saved] {keyname} -> {base}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
