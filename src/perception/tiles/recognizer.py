from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from .normalize import normalize_tile, to_gray_unit

TileClass = str

# 34 + red fives example; customize as you like
DEFAULT_CLASSES: List[TileClass] = [
    # manzu
    "1m","2m","3m","4m","5m","5mr","6m","7m","8m","9m",
    # pinzu
    "1p","2p","3p","4p","5p","5pr","6p","7p","8p","9p",
    # souzu
    "1s","2s","3s","4s","5s","5sr","6s","7s","8s","9s",
    # honors
    "E","S","W","N","P","F","C"
]

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(a.dot(b) / denom)

class TemplateRecognizer:
    def __init__(self, template_root: str | Path = "assets/templates", classes: Optional[List[TileClass]] = None):
        self.template_root = Path(template_root)
        self.classes = classes or DEFAULT_CLASSES
        self.templates: Dict[TileClass, List[np.ndarray]] = {c: [] for c in self.classes}
        self._load()

    def _load(self) -> None:
        for cls in self.classes:
            folder = self.template_root / cls
            if not folder.exists(): continue
            for p in sorted(folder.glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is None: continue
                img = normalize_tile(img)                 # resize + eq
                g = to_gray_unit(img).reshape(-1)         # gray + unit-ish
                self.templates[cls].append(g)

    def add_template(self, cls: TileClass, img_bgr: np.ndarray) -> None:
        img = normalize_tile(img_bgr)
        g = to_gray_unit(img).reshape(-1)
        (self.templates.setdefault(cls, [])).append(g)

    def classify(self, img_bgr: np.ndarray) -> Tuple[TileClass, float]:
        """Return (best_class, score). Score is cosine similarity of best match."""
        if img_bgr is None or img_bgr.size == 0:
            return ("", 0.0)
        g = to_gray_unit(normalize_tile(img_bgr)).reshape(-1)
        best_c, best_s = "", -1.0
        for cls, templs in self.templates.items():
            for t in templs:
                s = _cosine(g, t)
                if s > best_s:
                    best_s = s; best_c = cls
        return (best_c, best_s)

    def classify_many(self, imgs: List[np.ndarray], score_thresh: float = 0.55) -> List[Optional[TileClass]]:
        out: List[Optional[TileClass]] = []
        for im in imgs:
            c, s = self.classify(im)
            out.append(c if s >= score_thresh else None)
        return out
