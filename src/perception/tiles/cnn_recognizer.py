# src/perception/tiles/cnn_recognizer.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
import json
import onnxruntime as ort


def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class CNNRecognizer:
    def __init__(self, onnx_path: str, classes_path: str, img_size: int = 128):
        self.img_size = img_size
        self.sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.classes: List[str] = json.loads(Path(classes_path).read_text(encoding="utf-8"))
        # normalize to [-1,1]
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def _letterbox_square_rgb(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        if h == w:
            x = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        elif h > w:
            pad = (h - w) // 2
            x = cv2.copyMakeBorder(img_bgr, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            x = cv2.resize(x, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        else:
            pad = (w - h) // 2
            x = cv2.copyMakeBorder(img_bgr, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            x = cv2.resize(x, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = x.transpose(2, 0, 1)[None]  # 1x3xHxW
        return x

    def classify_many(self, tiles_bgr: List[np.ndarray], score_thresh: float = 0.6) -> List[Optional[str]]:
        if not tiles_bgr:
            return []
        xs = [self._letterbox_square_rgb(im) for im in tiles_bgr]
        X = np.concatenate(xs, axis=0)
        logits = self.sess.run([self.out_name], {self.in_name: X})[0]
        probs = softmax_np(logits, axis=1)
        ids = probs.argmax(axis=1)
        scr = probs.max(axis=1)
        out: List[Optional[str]] = []
        for i, sc in zip(ids, scr):
            out.append(self.classes[i] if sc >= score_thresh else None)
        return out

    def classify_many_tta(
        self,
        tiles_bgr: List[np.ndarray],
        score_thresh: float = 0.6,
        angles: List[int] = [0, 90, 180, 270],
    ) -> List[Tuple[Optional[str], int]]:
        if not tiles_bgr:
            return []
        results: List[Tuple[Optional[str], int]] = []
        for im in tiles_bgr:
            best_label: Optional[str] = None
            best_score: float = -1.0
            best_angle: int = 0
            for ang in angles:
                if ang == 0:
                    rot = im
                elif ang == 90:
                    rot = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
                elif ang == 180:
                    rot = cv2.rotate(im, cv2.ROTATE_180)
                elif ang == 270:
                    rot = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    h, w = im.shape[:2]
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
                    rot = cv2.warpAffine(
                        im, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114)
                    )
                X = self._letterbox_square_rgb(rot)
                logits = self.sess.run([self.out_name], {self.in_name: X})[0][0]
                probs = softmax_np(logits[None], axis=1)[0]
                idx = int(probs.argmax())
                sc = float(probs[idx])
                if sc > best_score:
                    best_score = sc
                    best_label = self.classes[idx]
                    best_angle = ang
            results.append((best_label if best_score >= score_thresh else None, best_angle))
        return results
