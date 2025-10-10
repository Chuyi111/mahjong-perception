from __future__ import annotations
from typing import List, Dict, Tuple, Literal
import cv2
import numpy as np

Rect = Tuple[int,int,int,int]

def _make_tile_template(h: int, w: int, border: int = 2, radius: int = 6) -> np.ndarray:
    """
    Procedurally build a generic 'tile edge' template: white rounded rectangle silhouette on black.
    Returned as single-channel uint8 [0..255].
    """
    h = max(14, int(h)); w = max(10, int(w))
    t = np.zeros((h, w), np.uint8)
    # rounded rectangle
    rect = (border, border, w-2*border, h-2*border)
    l,t0,w0,h0 = rect
    rr = max(2, int(radius * min(h,w)/64))
    # draw filled white rounded rect
    # approximate: draw a normal rect then blur slightly
    cv2.rectangle(t, (l, t0), (l+w0-1, t0+h0-1), 255, -1)
    t = cv2.GaussianBlur(t, (3,3), 0)
    return t

def _nms_1d(peaks: List[Tuple[int,float]], min_dist: int) -> List[Tuple[int,float]]:
    """Simple 1D non-max suppression over axis index."""
    peaks = sorted(peaks, key=lambda p: p[1], reverse=True)
    keep: List[Tuple[int,float]] = []
    taken = np.zeros(len(peaks), dtype=bool)
    for i,(idx,score) in enumerate(peaks):
        if taken[i]: continue
        keep.append((idx,score))
        # suppress neighbors
        for j,(jdx,_) in enumerate(peaks):
            if not taken[j] and abs(jdx - idx) < min_dist:
                taken[j] = True
    return sorted(keep, key=lambda p: p[0])

def locate_tiles_by_shape(
    roi_bgr: np.ndarray,
    orientation: Literal["horizontal","vertical"] = "horizontal",
    aspect: float = 0.75,
    scale_range: Tuple[float,float] = (0.9, 1.15),
    scale_steps: int = 5,
    peak_thresh: float = 0.35,
    min_separation_px: int | None = None,
    pad_px: int = 2,
) -> List[Dict]:
    """
    Find tile centers in an ROI by normalized cross-correlation with a generic tile silhouette.
    Returns list of dicts: {"img": crop, "rect": (l,t,w,h), "center": (cx,cy)} in ROI-local coords.

    - orientation='horizontal' for bands like hand_self/melds_top/self/bottom
      (tiles arranged left→right).
    - orientation='vertical' for melds_left/right (tiles stacked top→bottom).

    The ROI should be *roughly* fronto-parallel (your per-ROI quad warp is perfect).
    """
    H, W = roi_bgr.shape[:2]
    if H < 12 or W < 12:
        return []

    # Preprocess to edge-y signal to emphasize tile borders
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)).apply(g)
    e = cv2.Canny(g, 50, 150, L2gradient=True)

    # Choose nominal template size
    if orientation == "horizontal":
        th = int(0.80 * H)  # tile height ~ ROI height minus top/bottom margins
        tw = max(10, int(th * aspect))
        axis_len = W
    else:
        tw = int(0.80 * W)
        th = max(10, int(tw / aspect))
        axis_len = H

    if min_separation_px is None:
        # conservative: a bit smaller than nominal tile width/height
        min_separation_px = max(16, int(0.65 * (tw if orientation=="horizontal" else th)))

    # Build a small bank of templates across scale_range
    s_lo, s_hi = scale_range
    scalars = np.linspace(s_lo, s_hi, scale_steps).tolist()
    templates = []
    for s in scalars:
        th_s = max(12, int(round(th * s)))
        tw_s = max(10, int(round(tw * s)))
        templ = _make_tile_template(th_s, tw_s)
        templates.append((templ, th_s, tw_s))

    # Slide-match each template over the edge map
    # Use TM_CCOEFF_NORMED so higher is better and lighting-invariant-ish
    # For speed, we then reduce to 1D by max over rows or cols
    scores_1d = np.zeros(axis_len, np.float32)
    for templ, th_s, tw_s in templates:
        method = cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(e, templ, method)
        # res shape: (H-th_s+1, W-tw_s+1)
        if orientation == "horizontal":
            # For each x, take max over y
            col_max = res.max(axis=0)
            # Pad to match axis_len
            pad = axis_len - len(col_max)
            if pad > 0:
                col_max = np.pad(col_max, (0,pad), constant_values=0)
            scores_1d = np.maximum(scores_1d, col_max[:axis_len])
        else:
            # For each y, take max over x
            row_max = res.max(axis=1)
            pad = axis_len - len(row_max)
            if pad > 0:
                row_max = np.pad(row_max, (0,pad), constant_values=0)
            scores_1d = np.maximum(scores_1d, row_max[:axis_len])

    # Threshold + 1D NMS to get candidate indices
    peaks = [(i, float(scores_1d[i])) for i in range(axis_len) if scores_1d[i] >= peak_thresh]
    if not peaks:
        return []

    peaks = _nms_1d(peaks, min_separation_px)

    # Convert 1D indices to 2D rects/crops
    parts: List[Dict] = []
    for idx, sc in peaks:
        if orientation == "horizontal":
            # center x ~ idx; span tw, full height with small padding
            cx = int(idx)
            w = int(np.median([tw for _,_,tw in templates]))
            l = int(np.clip(cx - w//2, 0, W-1))
            r = int(np.clip(l + w, 1, W))
            t = pad_px
            b = H - pad_px
        else:
            cy = int(idx)
            h = int(np.median([th for _,th,_ in templates]))
            t = int(np.clip(cy - h//2, 0, H-1))
            b = int(np.clip(t + h, 1, H))
            l = pad_px
            r = W - pad_px

        if r - l < 10 or b - t < 10:
            continue

        crop = roi_bgr[t:b, l:r].copy()
        cx = (l + r)//2; cy = (t + b)//2
        parts.append({"img": crop, "rect": (l,t,r-l,b-t), "center": (cx,cy), "score": sc})

    # Sort along the band axis (L→R or T→B)
    parts.sort(key=lambda d: d["rect"][0] if orientation=="horizontal" else d["rect"][1])
    return parts
