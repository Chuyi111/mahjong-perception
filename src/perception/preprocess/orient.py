# src/perception/preprocess/orient.py
import cv2
import numpy as np

def rotate_by_seat(img_bgr: np.ndarray, seat: str) -> np.ndarray:
    """
    Rotate tiles from opponent seats into self’s orientation.
    Conventions (Mahjong Soul typical):
      - top   : 180°
      - left  : 90° clockwise
      - right : 90° counter-clockwise
      - self/bottom/anything else: no rotation
    """
    s = seat.lower()
    if s in ("top", "north", "up"):
        return cv2.rotate(img_bgr, cv2.ROTATE_180)
    if s in ("left", "west"):
        return cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)
    if s in ("right", "east"):
        return cv2.rotate(img_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img_bgr

def rotate_region_by_name(img_bgr: np.ndarray, region_name: str) -> np.ndarray:
    """
    Convenience wrapper that looks at region name prefixes:
      - 'melds_top', 'melds_left', 'melds_right'
      - 'discards_top', 'discards_left', 'discards_right'
    ‘_bottom’/‘_self’/others: no rotation.
    """
    name = region_name.lower()
    if name.endswith("_top"):
        return rotate_by_seat(img_bgr, "top")
    if name.endswith("_left"):
        return rotate_by_seat(img_bgr, "left")
    if name.endswith("_right"):
        return rotate_by_seat(img_bgr, "right")
    return img_bgr
