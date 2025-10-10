from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

"""
Adaptive hand tiler for the player's hand (hand_self).
- Works on the (optionally warped) hand ROI image (BGR).
- Detects an arbitrary number of tiles using horizontal edge projection.
- Finds the largest gap between consecutive tiles ⇢ likely the draw split.
- Returns per-tile crops, rects, centers, and split index.
"""

TilePart = Dict[str, object]  # {"img": np.ndarray, "rect": (l,t,w,h), "center": (cx,cy)}
HandAdaptiveResult = Dict[str, object]  # {"parts": List[TilePart], "split_idx": Optional[int]}

def _smooth_1d(x: np.ndarray, k: int = 15) -> np.ndarray:
    k = max(5, k | 1)  # odd
    return cv2.GaussianBlur(x.astype(np.float32), (k, 1), 0).ravel()

def _find_peaks(sig: np.ndarray, min_sep: int, thr: float) -> List[int]:
    """
    sig in [0..1], find local maxima separated by >= min_sep and >= thr.
    """
    peaks = []
    N = len(sig)
    i = 1
    while i < N - 1:
        if sig[i] > sig[i - 1] and sig[i] > sig[i + 1] and sig[i] >= thr:
            # NMS window
            L = max(0, i - min_sep // 2)
            R = min(N, i + min_sep // 2 + 1)
            if sig[i] >= sig[L:R].max():
                peaks.append(i)
                i += min_sep
                continue
        i += 1
    return peaks

def _bounds_around(sig: np.ndarray, c: int, frac: float = 0.40) -> Tuple[int, int]:
    """
    Return [L,R] where signal falls below 'frac * peak' around peak center 'c'.
    """
    peak = float(sig[c])
    thr = peak * frac
    L = c
    while L > 1 and sig[L] > thr:
        L -= 1
    R = c
    while R < len(sig) - 2 and sig[R] > thr:
        R += 1
    return L, R

def tile_hand_adaptive(
    roi_bgr: np.ndarray,
    margin: int = 2,
    min_sep_px: Optional[int] = None,
    peak_thr: float = 0.25,
    band_margin_px: int = 3,
) -> HandAdaptiveResult:
    """
    Segment the player's hand (any count). Also identify the largest inter-tile gap.

    Returns:
      {
        "parts": [{"img": crop, "rect": (l,t,w,h), "center": (cx,cy)}, ...]  # in ROI-local coords
        "split_idx": Optional[int]  # index i such that gap is between parts[i] and parts[i+1]
      }
    """
    H, W = roi_bgr.shape[:2]
    if H < 12 or W < 24:
        return {"parts": [], "split_idx": None}

    # Preprocess → edges
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    e = cv2.Canny(g, 50, 150, L2gradient=True)

    # Horizontal row: sum edges along rows → per-column signal
    sig = e.sum(axis=0).astype(np.float32)  # shape (W,)
    # Normalize and smooth
    sig = _smooth_1d(sig, k=max(9, W // 30 * 2 + 1))
    if sig.max() > 0:
        sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-6)
    else:
        sig[:] = 0

    # Guess a reasonable min separation from width
    if min_sep_px is None:
        min_sep_px = max(16, W // 14)  # ~hand tiles across

    centers = _find_peaks(sig, min_sep=min_sep_px, thr=peak_thr)

    parts: List[TilePart] = []
    for c in centers:
        L, R = _bounds_around(sig, c, frac=0.40)
        l = max(0, L - margin)
        r = min(W, R + margin)
        # Full height crop with small margins
        t = band_margin_px
        b = H - band_margin_px
        if r - l >= 10 and b - t >= 10:
            rect = (l, t, r - l, b - t)
            crop = roi_bgr[t:b, l:r].copy()
            parts.append({"img": crop, "rect": rect, "center": ((l + r) // 2, (t + b) // 2)})

    # Sort left→right
    parts.sort(key=lambda d: d["rect"][0])

    # Find the largest gap between consecutive tile centers
    split_idx: Optional[int] = None
    if len(parts) >= 2:
        gaps = []
        for i in range(len(parts) - 1):
            x_i = parts[i]["center"][0]
            x_j = parts[i + 1]["center"][0]
            gaps.append((x_j - x_i, i))  # (gap width, index)
        if gaps:
            gaps.sort(reverse=True)  # largest first
            max_gap, idx = gaps[0]
            # Heuristic: must be significantly larger than median gap to qualify as draw split
            widths = [p["rect"][2] for p in parts]
            med_gap = np.median([g for g, _ in gaps]) if len(gaps) >= 3 else (np.mean([g for g, _ in gaps]) if gaps else 0)
            ref = max(12.0, float(np.median(widths)) * 0.6)  # reference width
            if max_gap >= max(1.5 * med_gap, ref):  # stronger than neighbors and > ~0.6 tile width
                split_idx = idx

    return {"parts": parts, "split_idx": split_idx}
