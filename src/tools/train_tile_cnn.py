# src/tools/train_tile_cnn.py
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

# Albumentations for strong rotation/perspective aug
# Albumentations for strong rotation/perspective aug
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as e:
    import traceback
    msg = (
        "Import error loading Albumentations pipeline.\n"
        "Try installing in THIS environment:\n"
        "  python -m pip install albumentations==1.4.14 qudida==0.0.4 scikit-image==0.22.0\n"
        "  python -m pip install opencv-python==4.10.0.84 numpy==1.26.4\n\n"
        f"Original error:\n{traceback.format_exc()}"
    )
    raise SystemExit(msg)


# ------------------------------ Utils ------------------------------ #
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def letterbox_square(img_bgr: np.ndarray, size: int, border: int = 114) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h == w:
        return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    if h > w:
        pad = (h - w) // 2
        img = cv2.copyMakeBorder(img_bgr, 0, 0, pad, h - w - pad, cv2.BORDER_CONSTANT, value=(border, border, border))
    else:
        pad = (w - h) // 2
        img = cv2.copyMakeBorder(img_bgr, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=(border, border, border))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

class AlbumentationsImageFolder(Dataset):
    """ImageFolder but with custom (letterbox_square -> A.Compose) pipeline."""
    def __init__(self, img_paths: List[Path], class_indices: List[int], img_size: int, train: bool):
        self.paths = img_paths
        self.targets = class_indices
        self.img_size = img_size
        # augmentations
        if train:
            self.aug = A.Compose([
                # --- geometric invariances ---
                A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114), p=0.8),
                A.RandomRotate90(p=0.6),
                A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.06), shear=(-6, 6),
                         cval=(114,114,114), p=0.5),
                A.Perspective(scale=(0.02, 0.06), keep_size=True,
                              pad_mode=cv2.BORDER_CONSTANT, pad_val=(114,114,114), p=0.35),
                # --- photometric robustness ---
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                    A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),
                ], p=0.4),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02, p=0.6),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4,4), p=0.2),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2(),
            ])
        else:
            self.aug = A.Compose([
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2(),
            ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, i: int):
        p = str(self.paths[i])
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        # If alpha provided, composite over neutral gray
        if img.ndim == 3 and img.shape[2] == 4:
            b,g,r,a = cv2.split(img)
            bg = np.full_like(b, 114)
            for c in (b,g,r):
                c[:] = (c.astype(np.uint16) * a + bg.astype(np.uint16) * (255 - a)) // 255
            img = cv2.merge([b,g,r])
        sq = letterbox_square(img, size=self.img_size, border=114)
        sq = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        t = self.targets[i]
        tens = self.aug(image=sq)["image"]
        return tens, t

def split_paths(imagefolder: ImageFolder, val_split: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    by_class = {}
    for idx, (_, cls) in enumerate(imagefolder.samples):
        by_class.setdefault(cls, []).append(idx)
    train_idx, val_idx = [], []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_split)))
        val_idx.extend(idxs[:n_val]); train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx)
    return train_idx, val_idx

def build_datasets(data_root: Path, img_size: int, val_split: float, seed: int):
    # Load via ImageFolder just to get mapping & file list
    base_tf = transforms.ToTensor()  # dummy
    imgf = ImageFolder(str(data_root), transform=base_tf)
    classes = imgf.classes
    class_to_idx = imgf.class_to_idx
    samples = [Path(s[0]) for s in imgf.samples]
    targets = [s[1] for s in imgf.samples]

    train_ids, val_ids = split_paths(imgf, val_split, seed)
    train_paths = [samples[i] for i in train_ids]
    train_targets = [targets[i] for i in train_ids]
    val_paths = [samples[i] for i in val_ids]
    val_targets = [targets[i] for i in val_ids]

    ds_train = AlbumentationsImageFolder(train_paths, train_targets, img_size, train=True)
    ds_val   = AlbumentationsImageFolder(val_paths,   val_targets,   img_size, train=False)
    return ds_train, ds_val, classes

# ------------------------------ Model ------------------------------ #
def create_model(num_classes: int, arch: str = "mobilenet_v3_small", pretrained: bool = False):
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None if not pretrained else models.MobileNet_V3_Small_Weights.DEFAULT)
        in_feats = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_feats, num_classes)
        return m
    elif arch == "resnet18":
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported arch: {arch}")

# ------------------------------ Train / Val ------------------------------ #
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return (loss_sum / max(1,total), correct / max(1,total))

def train_one_epoch(model, loader, opt, device, scaler=None):
    model.train()
    ce = nn.CrossEntropyLoss()
    running = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast(True):
                logits = model(x)
                loss = ce(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

# ------------------------------ Main ------------------------------ #
def main():
    ap = argparse.ArgumentParser("Train rotation-robust tile classifier")
    ap.add_argument("--data", required=True, help="Root folder of class-sorted crops (ImageFolder layout)")
    ap.add_argument("--out", default="runs/tile_cnn", help="Output dir")
    ap.add_argument("--imgsz", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", type=str, default="mobilenet_v3_small", choices=["mobilenet_v3_small","resnet18"])
    ap.add_argument("--pretrained", action="store_true", help="Use ImageNet weights (requires internet)")
    ap.add_argument("--export_onnx", action="store_true", help="Export ONNX at the end")
    args = ap.parse_args()

    set_seed(args.seed)
    data_root = Path(args.data)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets/loaders
    ds_train, ds_val, classes = build_datasets(data_root, args.imgsz, args.val_split, args.seed)
    print(f"[INFO] Classes ({len(classes)}): {classes}")
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    # Model / Opt / Sched
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(len(classes), arch=args.arch, pretrained=args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc, best_path = 0.0, None
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, loader_train, opt, device, scaler)
        va_loss, va_acc = evaluate(model, loader_val, device)
        sched.step()

        dt = time.time() - t0
        print(f"[E{epoch:03d}] train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc*100:.2f}%  lr={sched.get_last_lr()[0]:.5f}  ({dt:.1f}s)")

        # Save last & best
        torch.save({"epoch": epoch, "model": model.state_dict(), "classes": classes, "args": vars(args)},
                   out_dir / "last.pt")
        if va_acc >= best_acc:
            best_acc = va_acc
            best_path = out_dir / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "classes": classes, "args": vars(args)},
                       best_path)

    print(f"[DONE] best_val_acc={best_acc*100:.2f}%  best={best_path}")

    # Optional ONNX export (for runtime)
    if args.export_onnx:
        try:
            model.eval()
            dummy = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
            onnx_path = out_dir / "tile_cnn.onnx"
            torch.onnx.export(
                model, dummy, str(onnx_path),
                input_names=["images"], output_names=["logits"],
                opset_version=12, dynamic_axes=None, do_constant_folding=True
            )
            print(f"[EXPORT] ONNX -> {onnx_path}")
        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}\nHint: pip install onnx==1.16.1 onnxruntime==1.17.3")

if __name__ == "__main__":
    main()
