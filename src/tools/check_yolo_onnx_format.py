# src/tools/check_yolo_onnx_format.py
import sys, json
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort

def letterbox(im, size=768):
    h, w = im.shape[:2]
    r = min(size / h, size / w)
    new = (int(round(w * r)), int(round(h * r)))
    im = cv2.resize(im, new, interpolation=cv2.INTER_LINEAR)
    dw, dh = size - new[0], size - new[1]
    top, bot = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    im = cv2.copyMakeBorder(im, top, bot, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (left, top)

def main(onnx_path: str, image_path: str, input_size: int = 768):
    onnx_path = str(Path(onnx_path))
    image_path = str(Path(image_path))
    sess = ort.InferenceSession(onnx_path, providers=ort.get_available_providers())
    in0 = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"[INFO] providers: {sess.get_providers()}")
    print(f"[INFO] input name: {in0}")
    print(f"[INFO] outputs: {out_names}")

    img0 = cv2.imread(image_path)
    if img0 is None:
        raise SystemExit(f"[ERR] cannot read image: {image_path}")
    H0, W0 = img0.shape[:2]
    img, r, (px, py) = letterbox(img0, input_size)
    x = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    x = x[None]

    outs = sess.run(out_names, {in0: x})
    o0 = outs[0]
    print(f"[INFO] outs[0].shape = {o0.shape}")

    # normalize to (N, C)
    if o0.ndim == 3 and o0.shape[0] == 1:
        o0 = o0[0]
    if o0.ndim != 2 or o0.shape[1] < 5:
        raise SystemExit(f"[ERR] unexpected first output shape {o0.shape}. Need (N, 4+num_classes).")

    N, C = o0.shape
    num_classes = C - 4
    print(f"[INFO] N rows: {N}, num_classes (from tensor) = {num_classes}")

    # Peek at first few rows
    print("[INFO] first 3 rows (box[0:4], top-3 class scores idx):")
    for i in range(min(3, N)):
        box = o0[i, :4]
        scores = o0[i, 4:]
        top3_idx = np.argsort(scores)[-3:][::-1]
        print(f"  row {i}: box={box.tolist()}  top3_cls={top3_idx.tolist()}  top3_scores={[float(scores[j]) for j in top3_idx]}")

    # Try to decide xywh vs xyxy by counting plausible boxes after unpad/scale
    def count_plausible(b_xyxy):
        b = b_xyxy.copy()
        b[:, [0, 2]] -= px; b[:, [1, 3]] -= py
        b /= r
        x1, y1, x2, y2 = b.T
        w = (x2 - x1); h = (y2 - y1)
        ok = (w > 1) & (h > 1) & (w < W0 * 1.1) & (h < H0 * 1.1) \
             & (x1 > -W0 * 0.1) & (y1 > -H0 * 0.1) & (x2 < W0 * 1.1) & (y2 < H0 * 1.1)
        return int(ok.sum())

    boxes = o0[:, :4].copy()
    # assume xyxy directly
    cnt_xyxy = count_plausible(boxes)
    # assume xywh and convert
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    boxes2 = np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=1)
    cnt_xywh = count_plausible(boxes2)
    fmt = "xywh (center)" if cnt_xywh >= cnt_xyxy else "xyxy (corners)"
    print(f"[INFO] plausible boxes if assume xyxy: {cnt_xyxy}")
    print(f"[INFO] plausible boxes if assume xywh: {cnt_xywh}")
    print(f"[RESULT] box format appears to be: {fmt}")

    # Dump a tiny JSON summary for copy/paste
    summary = {
        "num_classes": int(num_classes),
        "box_format": "xywh" if fmt.startswith("xywh") else "xyxy",
        "first_output_shape": list(outs[0].shape),
        "providers": sess.get_providers(),
        "output_names": out_names,
    }
    print("[SUMMARY]", json.dumps(summary, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.tools.check_yolo_onnx_format <model.onnx> <any_frame.png>")
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
