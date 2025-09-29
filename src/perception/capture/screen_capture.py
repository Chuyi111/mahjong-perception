import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from mss import mss

class ScreenCapturer:
    def __init__(self, region: Tuple[int,int,int,int]):
        l, t, r, b = region
        self.monitor = {"left": l, "top": t, "width": r - l, "height": b - t}
        self.sct = mss()

    def grab(self) -> np.ndarray:
        frame = self.sct.grab(self.monitor)
        # mss returns BGRA; convert to BGR for OpenCV
        img = np.array(frame)[:, :, :3]
        return img

def annotate_hud(img: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
    out = img.copy()
    y = 24
    for k, v in info.items():
        cv2.putText(out, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(out, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        y += 24
    return out
