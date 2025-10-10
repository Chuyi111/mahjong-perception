# src/tools/collect_dataset.py
from __future__ import annotations
import argparse, time, json, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from perception.utils.window_finder import (
    set_process_dpi_aware,
    find_window_by_titles,
    set_window_topmost,
)
from perception.capture.screen_capture import ScreenCapturer
from perception.config.io import load_config
from perception.config.model import Rect
from perception.preprocess.roi import preprocess_rois  # not used directly, but keeps parity
from perception.utils.capture_exclude import set_window_exclude_from_capture

ROI_QUADS_PATH = Path("configs/roi_quads.json")

# ---------------- Class map for band semantic segmentation ----------------
# 0 background
BAND_CLASS_IDS = {
    "hand_self":        1,
    "melds_self":       2,
    "discards_bottom":  3,
    "melds_left":       4,
    "discards_left":    5,
    "melds_top":        6,
    "discards_top":     7,
    "melds_right":      8,
    "discards_right":   9,
}
ORDER_TO_DRAW = [
    # draw broad regions first if you prefer (here all similar)
    "hand_self", "melds_self",
    "discards_bottom",
    "melds_left", "discards_left",
    "melds_top", "discards_top",
    "melds_right", "discards_right",
]

def _hash_config(cfg: object) -> str:
    s = json.dumps(cfg, ensure_ascii=False, sort_keys=True, default=lambda o: getattr(o, "__dict__", str(o)))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def _rect_to_poly(rect: Rect) -> List[Tuple[int,int]]:
    l, t, w, h = rect
    return [(l, t), (l + w - 1, t), (l + w - 1, t + h - 1), (l, t + h - 1)]

def _load_roi_quads() -> Dict[str, dict]:
    if ROI_QUADS_PATH.exists():
        return json.loads(ROI_QUADS_PATH.read_text(encoding="utf-8"))
    return {}

def _poly_from_sources(name: str, rois_rect: Dict[str, Optional[Rect]], quads_json: Dict[str, dict]) -> Optional[np.ndarray]:
    """
    Prefer quad polygon if present; else fall back to rect polygon.
    Return as int32 Nx1x2 array ready for cv2.fillPoly.
    """
    if name in quads_json:
        entry = quads_json[name]
        quad = entry.get("quad")
        if quad and len(quad) == 4:
            pts = np.array([[int(x), int(y)] for (x, y) in quad], dtype=np.int32).reshape(-1, 1, 2)
            return pts
    rect = rois_rect.get(name)
    if rect is None:
        return None
    pts = np.array(_rect_to_poly(rect), dtype=np.int32).reshape(-1, 1, 2)
    return pts

def _draw_band_mask(mask: np.ndarray, rois_rect: Dict[str, Optional[Rect]], quads_json: Dict[str, dict]) -> np.ndarray:
    """
    Fill polygons onto a single-channel uint8 mask using BAND_CLASS_IDS.
    """
    for name in ORDER_TO_DRAW:
        cid = BAND_CLASS_IDS.get(name)
        if cid is None:
            continue
        poly = _poly_from_sources(name, rois_rect, quads_json)
        if poly is None:
            continue
        cv2.fillPoly(mask, [poly], int(cid))
    return mask

def main():
    set_process_dpi_aware()

    ap = argparse.ArgumentParser("MahjongSoul dataset collector")
    ap.add_argument("--dataset_root", type=str, default="dataset", help="Root folder for outputs")
    ap.add_argument("--window_hint", type=str, default="", help="Optional override of window title hint")
    ap.add_argument("--save_masks", action="store_true", help="Start with band-mask saving ON")
    ap.add_argument("--burst", type=int, default=0, help="Burst count to save when pressing B (0=off)")
    ap.add_argument("--interval", type=int, default=120, help="Burst interval in ms")
    args = ap.parse_args()

    ds_root = Path(args.dataset_root)
    img_dir = ds_root / "images" / "raw"
    msk_dir = ds_root / "labels_bands_raw"
    meta_dir = ds_root / "meta"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = "configs/mahjongsoul.json"
    cfg = load_config(cfg_path)
    window_hint = args.window_hint.strip() or getattr(cfg.capture, "window_title_hint", "") or ""

    found = (
        find_window_by_titles(window_hint) or
        find_window_by_titles("雀魂麻將") or
        find_window_by_titles("雀魂") or
        find_window_by_titles("Mahjong Soul")
    )
    if not found:
        print("[ERR] Mahjong Soul window not found. Start the game and try again.")
        return
    hwnd_game, rect = found
    l, t, r, b = rect
    W, H = r - l, b - t
    print(f"[INFO] Capturing client area {rect} size={W}x{H}")

    set_window_topmost(hwnd_game, True)
    cap = ScreenCapturer(rect)

    roi_quads = _load_roi_quads()
    rois_rect: Dict[str, Optional[Rect]] = cfg.rois.model_dump()

    # save session meta
    meta = {
        "window_rect": [int(x) for x in rect],
        "config_path": cfg_path,
        "config_hash": _hash_config(cfg.model_dump()),
        "roi_quads_present": list(roi_quads.keys()),
        "band_class_ids": BAND_CLASS_IDS,
        "time_start": int(time.time()),
    }
    (meta_dir / "session_info.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    win_title = "Collector (Space=save, M=mask toggle, B=burst, T=topmost, Q/Esc=quit)"
    cv2.namedWindow(win_title, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_title, 40, 40)
    hwnd_prev = win32_find(win_title=None)
    try:
        import win32gui
        hwnd_prev = win32gui.FindWindow(None, win_title)
        if hwnd_prev:
            set_window_exclude_from_capture(hwnd_prev, True)
    except Exception:
        pass

    topmost = True
    save_masks = bool(args.save_masks)
    burst_n = max(0, int(args.burst))
    burst_interval_ms = max(1, int(args.interval))

    print("[INFO] Controls: Space=save one, M=toggle masks, B=burst, T=topmost, Q/Esc=quit")
    print(f"[INFO] Saving to {img_dir} (masks: {'ON' if save_masks else 'OFF'})")

    while True:
        frame = cap.grab()
        if frame is None or frame.size == 0:
            raw = cv2.waitKey(1)
            if raw != -1 and (raw & 0xFF) in (27, ord('q')):
                break
            continue

        vis = frame.copy()
        # HUD-ish overlay
        status = f"{W}x{H}  masks={'ON' if save_masks else 'OFF'}  burst={burst_n}x/{burst_interval_ms}ms  topmost={'ON' if topmost else 'OFF'}"
        cv2.putText(vis, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(win_title, vis)
        raw = cv2.waitKey(1)
        if raw == -1:
            continue
        key = raw & 0xFF

        if key in (27, ord('q')):
            break
        elif key == ord('t'):
            topmost = not topmost
            set_window_topmost(hwnd_game, topmost)
        elif key == ord('m'):
            save_masks = not save_masks
            print(f"[INFO] Save masks -> {save_masks}")
        elif key == ord(' '):  # save single
            ts = int(time.time() * 1000)
            img_path = img_dir / f"ms_{ts}.png"
            cv2.imwrite(str(img_path), frame)
            if save_masks:
                mask = np.zeros((H, W), np.uint8)
                _draw_band_mask(mask, rois_rect, roi_quads)
                msk_path = msk_dir / f"ms_{ts}.png"
                cv2.imwrite(str(msk_path), mask)
            print(f"[SAVE] {img_path.name}{' + mask' if save_masks else ''}")
        elif key == ord('b') and burst_n > 0:
            print(f"[BURST] Saving {burst_n} frames every {burst_interval_ms} ms …")
            for i in range(burst_n):
                frame = cap.grab()
                if frame is None or frame.size == 0:
                    time.sleep(burst_interval_ms / 1000.0)
                    continue
                ts = int(time.time() * 1000)
                img_path = img_dir / f"ms_{ts}.png"
                cv2.imwrite(str(img_path), frame)
                if save_masks:
                    mask = np.zeros((H, W), np.uint8)
                    _draw_band_mask(mask, rois_rect, roi_quads)
                    msk_path = msk_dir / f"ms_{ts}.png"
                    cv2.imwrite(str(msk_path), mask)
                print(f"  [{i+1}/{burst_n}] {img_path.name}{' + mask' if save_masks else ''}")
                cv2.waitKey(1)
                time.sleep(burst_interval_ms / 1000.0)

    set_window_topmost(hwnd_game, False)
    cv2.destroyAllWindows()

# tiny helper (avoid hard dependency at top for win32)
def win32_find(win_title: Optional[str]) -> Optional[int]:
    try:
        import win32gui
        return win32gui.FindWindow(None, win_title)
    except Exception:
        return None

if __name__ == "__main__":
    main()
