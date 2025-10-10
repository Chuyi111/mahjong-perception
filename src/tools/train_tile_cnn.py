# src/tools/train_tile_cnn.py
from __future__ import annotations
import argparse, os, json, collections
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

def build_transforms(img_size=128):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # Dialed-down aug so validation isn't a different universe
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
        transforms.RandomPerspective(distortion_scale=0.06, p=0.4),
        transforms.RandomAffine(degrees=5, translate=(0.04,0.04), scale=(0.97,1.03), shear=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25]),
    ])
    return train_tf, val_tf

def build_loaders(data_root, img_size=128, batch_size=128, workers=2):
    train_tf, val_tf = build_transforms(img_size)
    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    tr = datasets.ImageFolder(train_dir, transform=train_tf)
    va = datasets.ImageFolder(val_dir,   transform=val_tf)

    if tr.classes != va.classes:
        raise RuntimeError(
            "Train/val class lists differ.\n"
            f"train: {tr.classes}\nval  : {va.classes}\n"
            "Make sure both splits have the same subfolders (even if some are empty)."
        )
    num_classes = len(tr.classes)

    # Print class counts to spot issues
    tc = collections.Counter(tr.targets)
    vc = collections.Counter(va.targets)
    print("[INFO] Classes:", tr.classes)
    print("[INFO] Train counts (first 10):", [tc[i] for i in range(min(10,num_classes))], "…")
    print("[INFO] Val   counts (first 10):", [vc[i] for i in range(min(10,num_classes))], "…")

    # Class-balanced sampler
    tr_targets = np.array(tr.targets)
    class_counts = np.bincount(tr_targets, minlength=num_classes)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[tr_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr), replacement=True)

    train_loader = DataLoader(tr, batch_size=batch_size, sampler=sampler, num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, tr.classes

def build_model(num_classes, pretrained: bool):
    if pretrained:
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    else:
        m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_features, num_classes)
    return m

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1,total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tiles")
    ap.add_argument("--img", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrained", type=int, default=1)  # 1 = use ImageNet weights
    ap.add_argument("--out", default="models/tile_cnn")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, classes = build_loaders(args.data, args.img, args.batch)

    model = build_model(len(classes), pretrained=bool(args.pretrained)).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)

    best_acc = 0.0
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0; steps = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += loss.item(); steps += 1

        acc_tr = evaluate(model, train_loader, device)   # see overfit trend
        acc_va = evaluate(model, val_loader, device)
        sched.step()
        print(f"Epoch {epoch:02d} | train loss {running/max(1,steps):.4f} | "
              f"train acc {acc_tr*100:.1f}% | val acc {acc_va*100:.1f}%")

        if acc_va > best_acc:
            best_acc = acc_va
            torch.save({"model": model.state_dict(), "classes": classes}, str(out_dir / "best.pt"))

    # Export ONNX (if onnx installed)
    try:
        import onnx  # noqa: F401
        model.load_state_dict(torch.load(str(out_dir / "best.pt"), map_location=device)["model"])
        model.eval()
        dummy = torch.randn(1,3,args.img,args.img, device=device)
        onnx_path = out_dir / "tile_cnn.onnx"
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["logits"],
            opset_version=14,
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
        )
        (out_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
        print(f"Exported ONNX to {onnx_path}")
    except Exception as e:
        print("[WARN] ONNX export skipped:", e)

if __name__ == "__main__":
    main()
