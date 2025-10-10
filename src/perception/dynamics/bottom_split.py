# src/perception/dynamics/bottom_split.py
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import cv2

Rect = Tuple[int,int,int,int]

def _union_rect(a: Rect, b: Rect) -> Rect:
    l1,t1,w1,h1 = a; r1,b1 = l1+w1, t1+h1
    l2,t2,w2,h2 = b; r2,b2 = l2+w2, t2+h2
    L = min(l1,l2); T = min(t1,t2); R = max(r1,r2); B = max(b1,b2)
    return (L, T, R-L, B-T)

def _clip_rect(r: Rect, bounds_wh: Tuple[int,int]) -> Rect:
    l,t,w,h = r; W,H = bounds_wh
    l = max(0, min(l, W-1)); t = max(0, min(t, H-1))
    r = max(l+1, min(l+w, W)); b = max(t+1, min(t+h, H))
    return (l,t,r-l,b-t)

def _find_largest_valley(proj: np.ndarray, min_gap_px: int) -> Optional[int]:
    """Return x index of largest valley (gap) wider than min_gap_px."""
    k = max(5, len(proj)//200*2+1)
    sm = cv2.GaussianBlur(proj.astype(np.float32), (k,1), 0).ravel()
    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-6)
    thr = 0.25
    low = (sm < thr).astype(np.uint8)
    best_len, best_center = 0, None
    i = 0
    while i < len(low):
        if low[i]:
            j = i
            while j < len(low) and low[j]: j += 1
            length = j - i
            if length >= min_gap_px and length > best_len:
                best_len = length
                best_center = (i + j) // 2
            i = j
        else:
            i += 1
    return best_center

def dynamic_bottom_split(frame_bgr: np.ndarray,
                         hand_rect: Rect, meld_rect: Rect,
                         min_gap_px: int = 12) -> Tuple[Rect, Rect]:
    """
    Returns per-frame adjusted (meld_rect, hand_rect) by analyzing the union band,
    with the convention: **hand on the LEFT**, **melds on the RIGHT**.
    If no split is found, returns the input rects unchanged.
    """
    H, W = frame_bgr.shape[:2]
    union = _clip_rect(_union_rect(hand_rect, meld_rect), (W,H))
    l,t,w,h = union
    band = frame_bgr[t:t+h, l:l+w]
    if band.size == 0:
        return meld_rect, hand_rect

    g = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ink = 255 - bw
    proj = ink.sum(axis=0)  # columns

    valley_x = _find_largest_valley(proj, min_gap_px=min_gap_px)
    if valley_x is None:
        return meld_rect, hand_rect  # keep old

    split_x = l + int(valley_x)

    # LEFT -> hand, RIGHT -> melds
    Lh, Rh = l, split_x
    Lm, Rm = split_x, l + w
    hand_new = (Lh, t, max(8, Rh - Lh), h)
    meld_new = (Lm, t, max(8, Rm - Lm), h)
    return meld_new, hand_new
