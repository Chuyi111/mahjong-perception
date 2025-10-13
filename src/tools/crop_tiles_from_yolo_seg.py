# src/tools/crop_tiles_from_yolo_seg.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

def load_names_from_yaml(yaml_path: Path) -> List[str]:
    if yaml is None or not yaml_path.exists():
        return []
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = y.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda z: int(z))]
    if not isinstance(names, list):
        nc = int(y.get("nc", 0)) if "nc" in y else 0
        names = [f"class_{i}" for i in range(nc)]
    return names

def _looks_normalized(vals: List[float]) -> bool:
    # True if most values are in [0,1]
    if not vals: return True
    arr = np.array(vals, dtype=float)
    frac = np.mean((arr >= -1e-6) & (arr <= 1+1e-6))
    return frac > 0.8

def parse_yolov8_seg_line(line: str) -> Tuple[int, Optional[Tuple[float,float,float,float]], List[Tuple[float,float]], bool]:
    """
    Returns (cls, bbox_cxcywh or None, polygon_points, normalized)
    Accepts:
      A) "cls x1 y1 x2 y2 ... "  (polygon-only, normalized 0..1 or pixels)
      B) "cls cx cy w h x1 y1 x2 y2 ..." (bbox + polygon, normalized 0..1)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError("line too short")
    cls = int(float(parts[0]))
    vals = list(map(float, parts[1:]))

    # Case B heuristic: if at least 5 numbers and the rest even, and the first 4 look like cx,cy,w,h in [0,1]
    bbox = None
    poly_vals = None
    normalized = True

    if len(vals) >= 6 and ((len(vals) - 4) % 2 == 0):
        cx, cy, w, h = vals[0], vals[1], vals[2], vals[3]
        rest = vals[4:]
        if _looks_normalized([cx, cy, w, h]) and len(rest) >= 6:
            # treat as bbox+poly
            bbox = (cx, cy, w, h)
            poly_vals = rest
            normalized = _looks_normalized(rest)
        else:
            # treat whole list as polygon-only
            bbox = None
            poly_vals = vals
            normalized = _looks_normalized(poly_vals)
    else:
        # polygon-only (most Roboflow exports)
        bbox = None
        poly_vals = vals
        normalized = _looks_normalized(poly_vals)

    # make pairs
    if len(poly_vals) % 2 == 1:
        poly_vals = poly_vals[:-1]
    poly = [(poly_vals[i], poly_vals[i+1]) for i in range(0, len(poly_vals), 2)]
    if len(poly) < 3 and bbox is not None:
        # fallback to a rectangle from bbox if polygon is unusable
        cx, cy, w, h = bbox
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
        poly = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
    return cls, bbox, poly, normalized

def denorm_poly(poly, W, H, normalized: bool) -> np.ndarray:
    if normalized:
        pts = np.array([[min(max(px*W,0),W-1), min(max(py*H,0),H-1)] for (px,py) in poly], dtype=np.float32)
    else:
        pts = np.array([[min(max(px,0),W-1), min(max(py,0),H-1)] for (px,py) in poly], dtype=np.float32)
    return pts

def expand_box(x1,y1,x2,y2,W,H, pad_px=0, pad_pct=0.0):
    w = x2-x1+1; h = y2-y1+1
    dx = int(round(max(pad_px, w*pad_pct)))
    dy = int(round(max(pad_px, h*pad_pct)))
    return max(0,x1-dx), max(0,y1-dy), min(W-1,x2+dx), min(H-1,y2+dy)

def crop_polygon(img, poly_xy, alpha, pad_px, pad_pct, min_area):
    H,W = img.shape[:2]
    x,y,w,h = cv2.boundingRect(poly_xy.astype(np.int32))
    if w*h < max(1,min_area): return None
    x1,y1,x2,y2 = expand_box(x, y, x+w-1, y+h-1, W,H, pad_px, pad_pct)
    crop = img[y1:y2+1, x1:x2+1].copy()
    if not alpha: return crop
    mask = np.zeros((y2-y1+1, x2-x1+1), np.uint8)
    ps = (poly_xy - np.array([x1,y1], np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [ps], 255)
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgra[:,:,3] = mask
    return bgra

def iter_label_files(labels_dir: Path):
    return sorted(labels_dir.glob("*.txt"))

def paired_image_path(label_path: Path, images_dir: Path) -> Optional[Path]:
    stem = label_path.stem
    for ext in (".png",".jpg",".jpeg",".bmp",".webp"):
        p = images_dir / f"{stem}{ext}"
        if p.exists(): return p
    return None

def collect_pairs(ds_root: Path, split: str) -> List[Tuple[Path,Path]]:
    # Roboflow layout: ds/train/images, ds/train/labels, ds/valid/images, ds/valid/labels
    pairs = []
    def add_split(sname: str):
        imdir = ds_root / sname / "images"
        lbdir = ds_root / sname / "labels"
        if imdir.exists() and lbdir.exists():
            for lf in iter_label_files(lbdir):
                ip = paired_image_path(lf, imdir)
                if ip: pairs.append((ip, lf))
    if split == "all":
        for s in ("train","val","valid","test"):
            add_split(s)
    else:
        s = split
        if s == "val" and not (ds_root/"val").exists() and (ds_root/"valid").exists():
            s = "valid"
        add_split(s)
    return pairs

def main():
    ap = argparse.ArgumentParser("Crop tile polygons from YOLOv8 segmentation dataset")
    ap.add_argument("--dataset", required=True, help="Dataset ROOT (contains data.yaml and train/ valid/ etc.)")
    ap.add_argument("--split", default="all", choices=["all","train","val","valid","test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", action="store_true")
    ap.add_argument("--pad_px", type=int, default=2)
    ap.add_argument("--pad_pct", type=float, default=0.06)
    ap.add_argument("--min_area", type=int, default=100)
    ap.add_argument("--limit_per_image", type=int, default=0)
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()

    ds = Path(args.dataset)
    data_yaml = ds / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"[ERR] Expect dataset root with data.yaml, got: {ds}")

    names = load_names_from_yaml(data_yaml)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(ds, args.split)
    if args.max_images > 0: pairs = pairs[:args.max_images]

    print(f"[INFO] Dataset: {ds}")
    print(f"[INFO] Split: {args.split}, pairs: {len(pairs)}")
    print(f"[INFO] Output: {out_dir} (alpha={args.alpha}, pad_px={args.pad_px}, pad_pct={args.pad_pct}, min_area={args.min_area})")

    saved = 0
    for idx, (img_path, lbl_path) in enumerate(pairs, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] cannot read image: {img_path}")
            continue
        H,W = img.shape[:2]
        try:
            lines = [ln for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception as e:
            print(f"[WARN] cannot read label {lbl_path}: {e}")
            continue

        count_this = 0
        for li, line in enumerate(lines):
            try:
                cls, bbox, poly01, norm = parse_yolov8_seg_line(line)
            except Exception as e:
                print(f"[WARN] bad line {lbl_path.name}#{li}: {e}")
                continue

            poly_xy = denorm_poly(poly01, W, H, normalized=norm)
            if len(poly_xy) < 3:
                continue

            crop = crop_polygon(img, poly_xy, args.alpha, args.pad_px, args.pad_pct, args.min_area)
            if crop is None: 
                continue

            cls_name = names[cls] if 0 <= cls < len(names) else f"class_{cls}"
            out_name = f"{img_path.stem}_obj{li:03d}_{cls_name}.png"
            cv2.imwrite(str(out_dir / out_name), crop)
            saved += 1
            count_this += 1
            if args.limit_per_image and count_this >= args.limit_per_image:
                break

        if idx % 50 == 0:
            print(f"[PROG] {idx}/{len(pairs)} processed; saved {saved}")
    print(f"[DONE] Crops saved: {saved} -> {out_dir}")

if __name__ == "__main__":
    main()
