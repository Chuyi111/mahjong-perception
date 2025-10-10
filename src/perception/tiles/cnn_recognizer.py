from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import json
import numpy as np
import cv2
import onnxruntime as ort

class CNNRecognizer:
    def __init__(self, onnx_path: str | Path = "models/tile_cnn/tile_cnn.onnx",
                 classes_path: str | Path = "models/tile_cnn/classes.json",
                 img_size: int = 128, providers: Optional[List[str]] = None):
        self.onnx_path = str(onnx_path)
        self.img_size = img_size
        self.classes = json.loads(Path(classes_path).read_text(encoding="utf-8"))
        if providers is None:
            # CPUExecutionProvider is universal; add CUDA if available
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.onnx_path, providers=providers)

    def _pre(self, bgr: np.ndarray) -> np.ndarray:
        x = cv2.resize(bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.25  # same normalize as training
        x = x.transpose(2,0,1)  # CHW
        return x

    def classify_many(self, imgs: List[np.ndarray], score_thresh: float = 0.60) -> List[Optional[str]]:
        if not imgs:
            return []
        batch = np.stack([self._pre(im) for im in imgs], axis=0)
        logits = self.sess.run(None, {"input": batch})[0]  # (B, C)
        probs = softmax(logits, axis=1)
        idx = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        out: List[Optional[str]] = []
        for i in range(len(imgs)):
            if float(conf[i]) >= score_thresh:
                out.append(self.classes[int(idx[i])])
            else:
                out.append(None)
        return out

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
