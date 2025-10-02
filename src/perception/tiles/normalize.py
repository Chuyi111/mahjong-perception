from __future__ import annotations
import cv2
import numpy as np

def normalize_tile(img_bgr: np.ndarray, out_hw: tuple[int,int] = (96, 64)) -> np.ndarray:
    """Resize + gently equalize; result is BGR (for display) and also fine to convert to gray for matching."""
    h, w = out_hw
    img = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    y2 = clahe.apply(y)
    out = cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)
    # mild bilateral to denoise
    out = cv2.bilateralFilter(out, d=3, sigmaColor=30, sigmaSpace=3)
    return out

def to_gray_unit(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g
