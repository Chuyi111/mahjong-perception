from __future__ import annotations
import numpy as np
import cv2

def is_empty_slot(img_bgr: np.ndarray, edge_thresh: float = 0.015) -> bool:
    """
    Fast 'is there a tile here?' test.
    Returns True if the slot looks empty (background), False if likely occupied.
    Heuristic: edge density after light normalization.
    """
    if img_bgr is None or img_bgr.size == 0:
        return True
    h, w = img_bgr.shape[:2]
    if h < 8 or w < 8:  # degenerate
        return True

    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # gentle normalization (CLAHE) for stability
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    g = clahe.apply(g)
    # edges -> count nonzeros
    e = cv2.Canny(g, 40, 120, L2gradient=True)
    edge_density = float(np.count_nonzero(e)) / (h * w)

    # Optional: also check intensity variance (background tends to be flat)
    var = float(g.var()) / (255.0**2)

    # Decide: very low edges OR very low variance => empty
    return (edge_density < edge_thresh) or (var < 0.0025)
