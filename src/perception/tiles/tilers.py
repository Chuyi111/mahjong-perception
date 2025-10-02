from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2

Rect = Tuple[int,int,int,int]
Crop = np.ndarray

# ---- helpers ----
def _safe_crop(img: np.ndarray, rect: Rect) -> Optional[np.ndarray]:
    l,t,w,h = rect
    r, b = l+w, t+h
    H, W = img.shape[:2]
    if w <= 0 or h <= 0: return None
    if l >= W or t >= H or r <= 0 or b <= 0: return None
    l = max(0, l); t = max(0, t); r = min(W, r); b = min(H, b)
    if r <= l or b <= t: return None
    return img[t:b, l:r].copy()

def _uniform_slots(area_w: int, area_h: int, n: int, margin: int = 2) -> List[Rect]:
    """Split horizontally into n equal slots with small margins."""
    if n <= 0: return []
    slot_w = max(1, (area_w - margin * (n + 1)) // n)
    slots = []
    x = margin
    for _ in range(n):
        slots.append((x, margin, slot_w, max(1, area_h - 2*margin)))
        x += slot_w + margin
    return slots

def _proj_gaps(gray: np.ndarray, axis: int = 1) -> List[int]:
    """Simple gap detection by vertical projection minima; returns cut x-positions."""
    # binarize
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    proj = (255 - bw).sum(axis=0 if axis==1 else 1)  # ink along columns
    proj = cv2.normalize(proj.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    # find valleys
    minima = []
    win = max(3, gray.shape[1]//40)
    for x in range(1, len(proj)-1):
        if proj[x] < proj[x-1] and proj[x] < proj[x+1]:
            # non-maximum suppression window
            left = max(0, x - win); right = min(len(proj)-1, x + win)
            if proj[x] == proj[left:right+1].min():
                minima.append(x)
    return minima

# ---- tilers ----
def tile_hand_self(hand_roi: Crop, max_tiles: int = 14) -> List[Crop]:
    """Split the bottom-hand band into up to 14 slots. Projection first, else uniform."""
    H, W = hand_roi.shape[:2]
    # detect vertical gaps
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    cuts = _proj_gaps(gray, axis=1)
    # build slot rects between cuts
    slots: List[Rect] = []
    xs = [0] + sorted([x for x in cuts if 10 < x < W-10]) + [W]
    for i in range(len(xs)-1):
        l, r = xs[i], xs[i+1]
        if r - l > W * 0.04:  # ignore tiny slivers
            slots.append((l, 0, r-l, H))
    # fallback uniform if detection poor
    if len(slots) < 7:
        slots = _uniform_slots(W, H, max_tiles)
    # crop
    crops = []
    for rect in slots[:max_tiles]:
        c = _safe_crop(hand_roi, rect)
        if c is not None: crops.append(c)
    return crops

def tile_discards(discard_roi: Crop, cols: int = 6, rows: int = 3) -> List[Crop]:
    """Split a river band into a 6x3 grid (Mahjong Soul typical)."""
    H, W = discard_roi.shape[:2]
    margin = 2
    cell_w = (W - margin*(cols+1)) // cols
    cell_h = (H - margin*(rows+1)) // rows
    out: List[Crop] = []
    for ry in range(rows):
        for cx in range(cols):
            l = margin + cx*(cell_w+margin)
            t = margin + ry*(cell_h+margin)
            c = _safe_crop(discard_roi, (l, t, cell_w, cell_h))
            if c is not None: out.append(c)
    return out

def tile_melds(meld_roi: Crop, max_tiles: int = 12) -> List[Crop]:
    """Split the meld band into generous uniform slots (melds are clusters; a few will be empty)."""
    H, W = meld_roi.shape[:2]
    slots = _uniform_slots(W, H, max_tiles, margin=4)
    out: List[Crop] = []
    for rect in slots:
        c = _safe_crop(meld_roi, rect)
        if c is not None: out.append(c)
    return out
