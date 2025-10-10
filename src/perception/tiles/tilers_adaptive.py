from __future__ import annotations
from typing import List, Tuple, Dict
import cv2
import numpy as np

Rect = Tuple[int,int,int,int]

def _smooth_1d(x: np.ndarray, k: int = 15) -> np.ndarray:
    k = max(5, k | 1)  # odd
    x = cv2.GaussianBlur(x.astype(np.float32), (k, 1), 0).ravel()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x

def _find_peaks_1d(sig: np.ndarray, min_sep: int, thr: float) -> List[int]:
    """Return indices of local maxima separated by at least min_sep with height >= thr."""
    peaks = []
    N = len(sig)
    i = 1
    while i < N-1:
        if sig[i] > sig[i-1] and sig[i] > sig[i+1] and sig[i] >= thr:
            # non-maximum suppression window
            left = max(0, i - min_sep//2)
            right = min(N, i + min_sep//2 + 1)
            if sig[i] >= sig[left:right].max():
                peaks.append(i)
                i += min_sep
                continue
        i += 1
    return peaks

def _bounds_around_peak(sig: np.ndarray, c: int, frac: float = 0.35) -> Tuple[int,int]:
    """Return left/right bounds where signal falls below frac * peak."""
    peak = sig[c]
    thr = peak * frac
    L = c
    while L > 1 and sig[L] > thr:
        L -= 1
    R = c
    while R < len(sig)-2 and sig[R] > thr:
        R += 1
    return max(0, L), min(len(sig)-1, R)

def tile_melds_adaptive(
    roi_bgr: np.ndarray,
    region_name: str,
    margin: int = 2,
    min_sep_px: int = 22,
    peak_thr: float = 0.25,
) -> List[Dict]:
    """
    Segment however many tiles exist in a meld band.
    Returns list of dicts: {"img": crop, "rect": (l,t,w,h), "center": (cx,cy)} in ROI-local coords.

    Heuristic: edge density projection + peak finding.
    For left/right seats (vertical stacks), we detect along Y; for top/self seats (horizontal), along X.
    """
    H, W = roi_bgr.shape[:2]
    if H < 10 or W < 10:
        return []

    # Preprocess -> edges
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(g)
    e = cv2.Canny(g, 50, 150, L2gradient=True)

    name = region_name.lower()
    vertical_stack = name.endswith("_left") or name.endswith("_right")

    if vertical_stack:
        # sum edges along columns -> per-row signal
        sig = e.sum(axis=1).astype(np.float32)
        sig = _smooth_1d(sig, k=max(9, H // 30 * 2 + 1))
        peaks = _find_peaks_1d(sig, min_sep=max(16, H // 12, min_sep_px), thr=peak_thr)
        tiles = []
        for c in peaks:
            L, R = _bounds_around_peak(sig, c, frac=0.35)
            t = max(0, L - margin); b = min(H, R + margin)
            # take full width with small margins
            l = margin; r = W - margin
            if b - t >= 12 and r - l >= 12:
                rect = (l, t, r - l, b - t)
                crop = roi_bgr[t:b, l:r].copy()
                tiles.append({"img": crop, "rect": rect, "center": ( (l+r)//2, (t+b)//2 )})
        # sort top->bottom
        tiles.sort(key=lambda d: d["rect"][1])
        return tiles
    else:
        # horizontal row -> sum edges along rows -> per-column signal
        sig = e.sum(axis=0).astype(np.float32)
        sig = _smooth_1d(sig, k=max(9, W // 30 * 2 + 1))
        peaks = _find_peaks_1d(sig, min_sep=max(16, W // 12, min_sep_px), thr=peak_thr)
        tiles = []
        for c in peaks:
            L, R = _bounds_around_peak(sig, c, frac=0.35)
            l = max(0, L - margin); r = min(W, R + margin)
            t = margin; b = H - margin
            if r - l >= 12 and b - t >= 12:
                rect = (l, t, r - l, b - t)
                crop = roi_bgr[t:b, l:r].copy()
                tiles.append({"img": crop, "rect": rect, "center": ( (l+r)//2, (t+b)//2 )})
        # sort left->right
        tiles.sort(key=lambda d: d["rect"][0])
        return tiles
