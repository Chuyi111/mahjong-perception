# src/perception/preprocess/deskew.py
import cv2
import numpy as np
from typing import Dict

def warp_trapezoid_to_rect(img, trapezoid_pts, out_wh):
    """
    trapezoid_pts: 4 points (x,y) in order [TL, TR, BR, BL] in the source image
    out_wh: (W, H) of the rectified output
    """
    W, H = out_wh
    dst = np.float32([[0,0], [W-1,0], [W-1,H-1], [0,H-1]])
    M = cv2.getPerspectiveTransform(np.float32(trapezoid_pts), dst)
    return cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)

def mild_deskew_by_region(img, region_name: str, out_wh=(64,96)):
    """
    Cheap fixed deskew for regions typically foreshortened.
    Tuned for Mahjong Soul; adjust percents if needed.
    """
    name = region_name.lower()
    H, W = img.shape[:2]
    pad = max(2, int(0.01*W))
    # fractions for top edge “shortening”
    k = 0.08  # 8% inward on the near edge
    if name.endswith("_top"):
        # top band: top edge closer (shorter), bottom edge wider
        pts = [
            [int(W*k), 0], [int(W*(1-k)), 0],
            [W-1-pad, H-1-pad], [pad, H-1-pad]
        ]
    elif name.endswith("_left"):
        # left band: left edge closer
        pts = [
            [0, int(H*k)], [W-1-pad, pad],
            [W-1-pad, H-1-pad], [0, int(H*(1-k))]
        ]
    elif name.endswith("_right"):
        # right band: right edge closer
        pts = [
            [pad, pad], [W-1, int(H*k)],
            [W-1, int(H*(1-k))], [pad, H-1-pad]
        ]
    else:
        return img  # self/bottom or other regions: no deskew
    return warp_trapezoid_to_rect(img, pts, out_wh=(out_wh[0], out_wh[1]))
