from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import onnxruntime as ort

# ---------- utils ----------
def _letterbox(im: np.ndarray, new_shape=(768, 768), color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5, max_det: int = 300):
    if boxes.size == 0:
        return np.empty((0,), dtype=int)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)

# ---------- detector ----------
class YoloSegOnnxDetector:
    """
    YOLOv8-seg ONNX wrapper (boxes + class logits; mask prototypes ignored for now).
    Handles both output layouts:
      - (N, 4+num_classes+num_masks)
      - (C, N) where C = 4+num_classes+num_masks
    Second output is mask prototypes: (1, num_masks, Mh, Mw).
    """
    def __init__(self, onnx_path: str, input_size: int = 768, is_xywh: Optional[bool] = None,
                 providers: Optional[list] = None,
                 class_names: Optional[List[str]] = None):
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.is_xywh = is_xywh  # None = auto-detect on first call
        self.sess = ort.InferenceSession(onnx_path, providers=providers or ort.get_available_providers())
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.class_names = list(class_names) if class_names else []

        # internal memo
        self._decided_format = False

    def _ensure_names_len(self, num_classes: int):
        if not self.class_names:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
            return
        if len(self.class_names) < num_classes:
            # extend with placeholders
            start = len(self.class_names)
            self.class_names += [f"class_{i}" for i in range(start, num_classes)]
            print(f"[YOLO ONNX] Warning: model has {num_classes} classes; extending names to match.")
        elif len(self.class_names) > num_classes:
            self.class_names = self.class_names[:num_classes]
            print(f"[YOLO ONNX] Warning: trimming provided names to {num_classes}.")

    def infer(self, frame_bgr: np.ndarray, conf_thr: float = 0.30, iou_thr: float = 0.50,
              max_det: int = 300) -> List[Dict]:
        H, W = frame_bgr.shape[:2]
        img, r, (padx, pady) = _letterbox(frame_bgr, (self.input_size, self.input_size))
        x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        x = x[None, ...]

        outs = self.sess.run(self.out_names, {self.in_name: x})
        raw = outs[0]

        # normalize to shape (N, C)
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]  # drop batch

        if raw.ndim != 2:
            raise RuntimeError(f"Unexpected YOLO ONNX output shape: {raw.shape}")

        # If channel-first (C, N) where C is small (~45) and N is large (thousands), transpose
        if raw.shape[0] <= 256 and raw.shape[1] >= 512:
            # likely (C, N) -> transpose to (N, C)
            raw = raw.T

        N, C = raw.shape
        if C < 5:
            raise RuntimeError(f"First output must have at least 5 columns, got {C}")

        # Get num_masks from prototypes (outs[1]) if present, else guess common 32
        num_masks = 32
        if len(outs) >= 2 and outs[1] is not None and hasattr(outs[1], "shape"):
            # prototypes: (1, num_masks, Mh, Mw) or similar
            proto = outs[1]
            try:
                nm = int(proto.shape[1])
                if 0 < nm < 512:
                    num_masks = nm
            except Exception:
                pass

        # If we know number of class names, use that; otherwise infer from C - 4 - num_masks
        num_classes = C - 4 - num_masks
        if self.class_names:
            num_classes = len(self.class_names)
        else:
            if num_classes <= 0:
                # fallback: treat last 32 as masks and next as classes if any
                num_masks = min(32, max(0, C - 4 - 1))  # keep sane
                num_classes = max(1, C - 4 - num_masks)

        # Ensure names length is valid
        self._ensure_names_len(num_classes)

        # Split fields
        boxes = raw[:, :4].copy()                              # xywh or xyxy (model space)
        cls_logits = raw[:, 4:4 + num_classes].copy()          # (N, num_classes)
        # Remaining (raw[:, 4 + num_classes:]) are mask coeffs -> ignored here

        # Decide format once if needed
        if self.is_xywh is None and not self._decided_format:
            # Try both interpretations on a small sample
            sample = boxes[:min(200, len(boxes))].copy()
            def count_plausible(b_xyxy):
                bz = b_xyxy.copy()
                bz[:, [0, 2]] -= padx; bz[:, [1, 3]] -= pady
                bz /= r
                w = (bz[:, 2] - bz[:, 0]); h = (bz[:, 3] - bz[:, 1])
                ok = (w > 1) & (h > 1) & (w < W * 1.1) & (h < H * 1.1)
                return int(ok.sum())
            # assume xyxy
            cnt_xyxy = count_plausible(sample)
            # assume xywh -> xyxy
            cx, cy, w0, h0 = sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3]
            sample2 = np.stack([cx - w0/2, cy - h0/2, cx + w0/2, cy + h0/2], axis=1)
            cnt_xywh = count_plausible(sample2)
            self.is_xywh = (cnt_xywh >= cnt_xyxy)
            self._decided_format = True
            print(f"[YOLO ONNX] Auto box format = {'xywh' if self.is_xywh else 'xyxy'}")

        # Convert boxes to xyxy
        if self.is_xywh:
            cx, cy, w0, h0 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            x1 = cx - w0 / 2; y1 = cy - h0 / 2
            x2 = cx + w0 / 2; y2 = cy + h0 / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Map to original image (undo pad/scale) and clip
        boxes[:, [0, 2]] -= padx
        boxes[:, [1, 3]] -= pady
        boxes /= r
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H - 1)

        # Per-row class selection
        cls_indices = cls_logits.argmax(axis=1)
        cls_scores = cls_logits[np.arange(cls_logits.shape[0]), cls_indices]

        # confidence filter
        keep = cls_scores >= conf_thr
        if not np.any(keep):
            return []

        boxes = boxes[keep]
        cls_indices = cls_indices[keep]
        cls_scores = cls_scores[keep]

        # Per-class NMS
        out: List[Dict] = []
        for c in np.unique(cls_indices):
            idxs = np.where(cls_indices == c)[0]
            keep_idx = _nms(boxes[idxs], cls_scores[idxs], iou_thr=iou_thr, max_det=max_det)
            for j in idxs[keep_idx]:
                cid = int(cls_indices[j])
                cname = self.class_names[cid] if 0 <= cid < len(self.class_names) else f"class_{cid}"
                out.append({
                    "xyxy": boxes[j].astype(float),
                    "cls_id": cid,
                    "score": float(cls_scores[j]),
                    "cls_name": cname,
                })
        return out
