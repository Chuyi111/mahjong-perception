from typing import Dict, Optional, Tuple
import numpy as np
import cv2

Rect = Tuple[int, int, int, int]  # (l,t,w,h)

def crop_rect(img_bgr: np.ndarray, rect: Optional[Rect]) -> Optional[np.ndarray]:
    if rect is None:
        return None
    l, t, w, h = rect
    r, b = l + w, t + h
    h_img, w_img = img_bgr.shape[:2]
    l = max(0, min(w_img-1, l)); r = max(0, min(w_img, r))
    t = max(0, min(h_img-1, t)); b = max(0, min(h_img, b))
    if r <= l or b <= t:
        return None
    return img_bgr[t:b, l:r].copy()

def normalize_tileband(img_bgr: np.ndarray) -> np.ndarray:
    """Lightweight normalization for UI regions with tiles/text."""
    # Convert to YCrCb and equalize luminance (CLAHE)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y_eq = clahe.apply(y)
    out = cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
    # Gentle bilateral to reduce compression noise without blurring edges too much
    out = cv2.bilateralFilter(out, d=5, sigmaColor=50, sigmaSpace=5)
    return out

def preprocess_rois(frame_bgr: np.ndarray, rois: Dict[str, Optional[Rect]]) -> Dict[str, Optional[np.ndarray]]:
    out: Dict[str, Optional[np.ndarray]] = {}
    for name, rect in rois.items():
        crop = crop_rect(frame_bgr, rect)
        if crop is None:
            out[name] = None
            continue
        # normalize some typical bands; keep anchors raw
        if name.startswith("anchor_"):
            out[name] = crop
        else:
            out[name] = normalize_tileband(crop)
    return out
