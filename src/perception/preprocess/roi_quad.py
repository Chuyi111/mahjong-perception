# src/perception/preprocess/roi_quad.py
from __future__ import annotations
from typing import Tuple, Optional, List
import cv2
import numpy as np

Rect = Tuple[int,int,int,int]
Quad = List[Tuple[int,int]]  # [(x_tl,y_tl),(x_tr,y_tr),(x_br,y_br),(x_bl,y_bl)]

def warp_quad(img_bgr: np.ndarray, quad_xy: Quad, out_wh: Tuple[int,int]) -> np.ndarray:
    """
    Warp an arbitrary 4-point ROI (TL,TR,BR,BL) to a fronto-parallel rectangle of size out_wh.
    """
    W, H = out_wh
    if len(quad_xy) != 4:
        raise ValueError("quad_xy must have 4 points in TL,TR,BR,BL order")
    src = np.float32(quad_xy)
    dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR)

def rect_crop(img_bgr: np.ndarray, rect: Rect) -> Optional[np.ndarray]:
    """
    Safe crop by rect (l,t,w,h). Returns None if degenerate.
    """
    l, t, w, h = rect
    H, W = img_bgr.shape[:2]
    l = max(0, min(l, W-1))
    t = max(0, min(t, H-1))
    r = max(l+1, min(l + max(1, w), W))
    b = max(t+1, min(t + max(1, h), H))
    if r - l <= 1 or b - t <= 1:
        return None
    return img_bgr[t:b, l:r].copy()
