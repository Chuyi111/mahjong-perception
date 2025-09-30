from pydantic import BaseModel, Field
from typing import Dict, Optional, Tuple

Rect = Tuple[int, int, int, int]  # (left, top, width, height), screen coords

class ROISet(BaseModel):
    # hands / discards
    hand_self: Optional[Rect] = None
    discards_top: Optional[Rect] = None
    discards_right: Optional[Rect] = None
    discards_bottom: Optional[Rect] = None
    discards_left: Optional[Rect] = None

    # melds (all seats)
    melds_self: Optional[Rect] = None
    melds_top: Optional[Rect] = None
    melds_right: Optional[Rect] = None
    melds_left: Optional[Rect] = None

    # info
    dora: Optional[Rect] = None
    winds_round: Optional[Rect] = None
    points_board: Optional[Rect] = None


class CaptureConfig(BaseModel):
    window_title_hint: str = "雀魂麻將"
    fps_target: int = 30
    baseline_width: int = Field(default=0, description="Saved client width at calibration")
    baseline_height: int = Field(default=0, description="Saved client height at calibration")

class AppConfig(BaseModel):
    capture: CaptureConfig = CaptureConfig()
    rois: ROISet = ROISet()
