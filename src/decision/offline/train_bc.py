from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# --- dataset & collator ---
from .sqlite_dataset import MahjongSQLiteDataset
from .collate import bc_collator


# ================= Utils =================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ================= Model =================
class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, num_actions: int = 34, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits (unmasked)


# ================= Masking & Metrics =================
def apply_action_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Illegal actions (mask==0) -> -inf via large negative shift."""
    return logits + (mask - 1.0) * 1e9

def masked_ce_loss(logits: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    masked = apply_action_mask(logits, mask)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)(masked, targets)

@torch.no_grad()
def topk_accuracy_from_masked(masked_logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    k = min(k, masked_logits.size(1))
    topk_idx = torch.topk(masked_logits, k=k, dim=1).indices  # (B,k)
    correct = (topk_idx == targets.unsqueeze(1)).any(dim=1).float()
    return correct.mean().item()


# ================= Eval =================
@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device, label_smoothing: float) -> tuple[float, float, float, int]:
    model.eval()
    total_loss, acc1_sum, acc3_sum, total_n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        X, M, y = batch
        X, M, y = X.to(device), M.to(device), y.to(device)
        logits = model(X)
        loss = masked_ce_loss(logits, M, y, label_smoothing=label_smoothing)
        masked_logits_ = apply_action_mask(logits, M)

        # top-1
        pred = masked_logits_.argmax(dim=1)
        acc1 = (pred == y).float().sum().item()
        # top-3
        acc3 = topk_accuracy_from_masked(masked_logits_, y, k=3) * X.size(0)

        n = X.size(0)
        total_loss += loss.item() * n
        acc1_sum   += acc1
        acc3_sum   += acc3
        total_n    += n

    if total_n == 0:
        return float("inf"), 0.0, 0.0, 0
    return total_loss / total_n, acc1_sum / total_n, acc3_sum / total_n, total_n


# ================= Main =================
def main():
    ap = argparse.ArgumentParser("Train Behavior Cloning policy from SQLite logs (masked CE, top-3 metrics)")
    ap.add_argument("--db", required=True, help="Path to extracted .sqlite database")
    ap.add_argument("--tables", default="Discard", help="Comma-separated tables to read (e.g. 'Discard,Riichi')")
    ap.add_argument("--json_col", default=None, help="Specify JSON column if known; else auto-detect")
    ap.add_argument("--id_col",   default="ID", help="Primary key column; if missing the dataset may auto-fallback")
    ap.add_argument("--sample_limit", type=int, default=200_000, help="Max samples to index from DB")
    ap.add_argument("--sample_stride", type=int, default=1, help="Use every Nth id to thin huge tables")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    ap.add_argument("--out", required=True, help="Output folder")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch",  type=int, default=1024)
    ap.add_argument("--lr",     type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # --- dataset (lazy, gz/plain JSON from SQLite) ---
    table_list = tuple([t.strip() for t in args.tables.split(",") if t.strip()])
    base_ds = MahjongSQLiteDataset(
        db_path=args.db,
        tables=table_list,
        sample_limit=args.sample_limit,
        sample_stride=args.sample_stride,
        shuffle_index=True,
        json_col=args.json_col,
        id_col=args.id_col,
    )
    N = len(base_ds)
    n_val = max(1, int(round(N * args.val_ratio)))
    n_tr  = max(1, N - n_val)

    # deterministic split
    g = torch.Generator()
    g.manual_seed(args.seed)
    ds_tr, ds_va = random_split(base_ds, [n_tr, n_val], generator=g)

    # infer feature dim (peek with workers=0), then clear any main-process connection
    tmp_loader = DataLoader(ds_tr, batch_size=64, shuffle=False, num_workers=0, collate_fn=bc_collator)
    feat_dim = None
    for b in tmp_loader:
        if b is None:
            continue
        X, M, y = b
        feat_dim = int(X.shape[1])
        break
    try:
        if hasattr(ds_tr, "dataset") and hasattr(ds_tr.dataset, "_conn"):
            ds_tr.dataset._conn = None
        if hasattr(ds_va, "dataset") and hasattr(ds_va.dataset, "_conn"):
            ds_va.dataset._conn = None
    except Exception:
        pass
    if feat_dim is None:
        raise SystemExit("[ERR] Could not infer feature dimension (no valid samples). Check DB/tables/filters.")

    # loaders
    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=bc_collator,
        persistent_workers=(args.workers > 0)
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=bc_collator,
        persistent_workers=(args.workers > 0)
    )

    # model / opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPPolicy(in_dim=feat_dim, hidden=args.hidden, num_actions=34, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc1 = 0.0
    best_path = out_dir / "bc_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tr_loss, tr_acc1, tr_acc3, tr_n = 0.0, 0.0, 0.0, 0

        for batch in dl_tr:
            if batch is None:
                continue
            X, M, y = batch
            X = X.to(device, non_blocking=True)
            M = M.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            loss = masked_ce_loss(logits, M, y, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            masked_logits_ = apply_action_mask(logits, M)
            pred = masked_logits_.argmax(dim=1)
            tr_acc1 += (pred == y).float().sum().item()
            tr_acc3 += topk_accuracy_from_masked(masked_logits_, y, k=3) * X.size(0)
            tr_loss += loss.item() * X.size(0)
            tr_n    += X.size(0)

        tr_loss = tr_loss / max(1, tr_n)
        tr_acc1 = tr_acc1 / max(1, tr_n)
        tr_acc3 = tr_acc3 / max(1, tr_n)

        va_loss, va_acc1, va_acc3, va_n = eval_epoch(model, dl_va, device, args.label_smoothing)
        dt = time.time() - t0
        print(f"[E{epoch:03d}] "
              f"train_loss={tr_loss:.4f} acc1={tr_acc1*100:.2f}% acc3={tr_acc3*100:.2f}% | "
              f"val_loss={va_loss:.4f} acc1={va_acc1*100:.2f}% acc3={va_acc3*100:.2f}% | "
              f"n_tr={tr_n} n_va={va_n} ({dt:.1f}s)")

        # save last and best
        torch.save({"model": model.state_dict(), "in_dim": feat_dim, "args": vars(args)}, out_dir / "bc_last.pt")
        if va_acc1 >= best_acc1:
            best_acc1 = va_acc1
            torch.save({"model": model.state_dict(), "in_dim": feat_dim, "args": vars(args)}, best_path)

    print(f"[DONE] best_val_acc1={best_acc1*100:.2f}% -> {best_path}")

    # export ONNX from CPU
    try:
        model.eval()
        model_cpu = model.to("cpu")
        dummy = torch.zeros(1, feat_dim, dtype=torch.float32)
        onnx_path = out_dir / "bc_policy.onnx"
        torch.onnx.export(
            model_cpu, dummy, str(onnx_path),
            input_names=["feat"], output_names=["logits"],
            opset_version=12, do_constant_folding=True,
            dynamic_axes={"feat": {0: "batch"}, "logits": {0: "batch"}}
        )
        print(f"[EXPORT] ONNX -> {onnx_path}")
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")

    # classes.json (34 tile labels)
    names = []
    for t in range(34):
        if t >= 27:
            names.append(["E","S","W","N","P","F","C"][t-27])
        else:
            suit = "m" if t < 9 else ("p" if t < 18 else "s")
            n = (t % 9) + 1
            names.append(f"{n}{suit}")
    (out_dir / "classes.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
