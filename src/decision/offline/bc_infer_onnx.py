from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import onnxruntime as ort

def policy_topk(logits: np.ndarray, mask: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
    """
    logits: (34,) raw logits from BC policy
    mask:   (34,) 1 for legal, 0 for illegal
    returns list of (tile_id_34, prob) length k
    """
    masked = logits + (mask - 1.0) * 1e9  # illegal -> -inf
    m = masked.max()
    probs = np.exp(masked - m)
    probs *= (mask > 0)
    Z = probs.sum()
    if Z <= 0:
        legal = np.where(mask > 0)[0]
        if len(legal) == 0:
            return []
        probs = np.zeros_like(masked)
        probs[legal] = 1.0 / len(legal)
        Z = 1.0
    probs = probs / Z
    idx = np.argsort(-probs)[:k]
    return [(int(i), float(probs[i])) for i in idx]

class BCPolicyONNX:
    def __init__(self, onnx_path: str):
        self.onnx_path = str(onnx_path)
        providers = ["CPUExecutionProvider"]
        # If you installed CUDA EP for onnxruntime-gpu, you can prefer it:
        try:
            from onnxruntime.capi._pybind_state import get_available_providers
            av = get_available_providers()
            if "CUDAExecutionProvider" in av:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        self.sess = ort.InferenceSession(self.onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name   # "feat"
        self.out_name = self.sess.get_outputs()[0].name # "logits"

    def __call__(self, feat_row: np.ndarray) -> np.ndarray:
        """
        feat_row: (D,) float32 (same layout as training features)
        returns logits: (34,)
        """
        if feat_row.ndim == 1:
            feat_row = feat_row[None, :]  # (1,D)
        out = self.sess.run([self.out_name], {self.in_name: feat_row.astype(np.float32)})[0]
        return out[0]
