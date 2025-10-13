# src/decision/offline/train_calls.py
from __future__ import annotations
import argparse, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .calls_dataset import CallsDataset
from .calls_collate import calls_collator


def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


class CallPolicyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, X):  # X: (sumK, D)
        return self.net(X).squeeze(-1)  # (sumK,)


def grouped_loss(
    scores: torch.Tensor,      # (sumK,)
    group_ptrs: torch.Tensor,  # (N,2)
    labels: torch.Tensor,      # (N,)
    *,
    skip_weight: float = 2.0,  # <— weight rows where label==0
    label_smoothing: float = 0.05,  # small smoothing
    entropy_reg: float = 1e-3,      # optional entropy bonus
) -> tuple[torch.Tensor, float, float]:
    """
    Returns (loss, nll, entropy) where:
      loss = weighted NLL (with smoothing) - lambda * entropy
    """
    device = scores.device
    N = group_ptrs.size(0)
    total_nll = torch.zeros((), device=device)
    total_H   = torch.zeros((), device=device)
    total_w   = torch.zeros((), device=device)

    for i in range(N):
        start, length = group_ptrs[i]
        sl = scores[start:start+length]              # (K,)
        logp = sl.log_softmax(dim=0)                 # grouped softmax
        p = logp.exp()                               # (K,)

        # label smoothing target
        y = labels[i].item()
        K = int(length.item()) if torch.is_tensor(length) else int(length)
        eps = label_smoothing
        target = torch.full((K,), fill_value=eps / max(1, K-1), device=device)
        target[y] = 1.0 - eps

        nll_i = -(target * logp).sum()

        # row weight
        w_i = torch.tensor(skip_weight if y == 0 else 1.0, device=device)
        total_nll = total_nll + w_i * nll_i
        total_H   = total_H + w_i * (-(p * logp).sum())  # entropy
        total_w   = total_w + w_i

    if total_w.item() == 0:
        return torch.zeros((), device=device), 0.0, 0.0

    nll = total_nll / total_w
    H   = total_H / total_w
    loss = nll - entropy_reg * H
    return loss, float(nll.item()), float(H.item())


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    total_callrate = 0.0
    for batch in loader:
        if batch is None: continue
        X_cat, group_ptrs, y_all = batch
        X_cat = X_cat.to(device); group_ptrs = group_ptrs.to(device); y_all = y_all.to(device)
        scores = model(X_cat)
        # accuracy + call rate
        Nrows = group_ptrs.size(0)
        acc = 0.0; callrate = 0.0
        for i in range(Nrows):
            s, L = group_ptrs[i]
            sl = scores[s:s+L]
            pred = sl.argmax().item()
            if pred == int(y_all[i].item()):
                acc += 1.0
            if pred != 0:
                callrate += 1.0
        total_acc  += acc
        total_callrate += callrate
        total_n    += Nrows
    if total_n == 0: return 0.0, 0.0, 0.0, 0
    return 0.0, total_acc/total_n, total_callrate/total_n, total_n


def main():
    ap = argparse.ArgumentParser("Train Call/Skip policy (balanced)")
    ap.add_argument("--db", required=True)
    ap.add_argument("--tables", default="Chi,Pon,DaiMinKan,ShouMinKan,AnKan,Skip")
    ap.add_argument("--json_col", default=None)
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--sample_limit", type=int, default=200_000)
    ap.add_argument("--sample_stride", type=int, default=1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_rows", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    # NEW balancing knobs
    ap.add_argument("--p_keep_call", type=float, default=0.5, help="undersample call rows (keep prob)")
    ap.add_argument("--skip_weight", type=float, default=2.0, help="weight for skip rows in loss")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--entropy_reg", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ds = CallsDataset(
        db_path=args.db,
        tables=tuple([t.strip() for t in args.tables.split(",") if t.strip()]),
        sample_limit=args.sample_limit,
        sample_stride=args.sample_stride,
        shuffle_index=True,
        json_col=args.json_col,
        id_col=args.id_col,
        p_keep_call=args.p_keep_call,   # <— undersampling
        seed=args.seed,
    )
    N = len(ds)
    n_val = max(1, int(round(N * args.val_ratio)))
    n_tr  = max(1, N - n_val)
    g = torch.Generator(); g.manual_seed(args.seed)
    ds_tr, ds_va = random_split(ds, [n_tr, n_val], generator=g)

    # Peek to infer D
    tmp = DataLoader(ds_tr, batch_size=1, shuffle=False, num_workers=0, collate_fn=calls_collator)
    in_dim = None
    for b in tmp:
        if b is None: continue
        X_cat, group_ptrs, y_all = b
        in_dim = int(X_cat.shape[1]); break
    if in_dim is None:
        raise SystemExit("[ERR] could not infer feature dim; dataset empty?")

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_rows, shuffle=True,
                       num_workers=args.workers, collate_fn=calls_collator,
                       pin_memory=True, persistent_workers=(args.workers>0))
    dl_va = DataLoader(ds_va, batch_size=args.batch_rows, shuffle=False,
                       num_workers=args.workers, collate_fn=calls_collator,
                       pin_memory=True, persistent_workers=(args.workers>0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CallPolicyMLP(in_dim=in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        t0 = time.time()
        tr_acc = tr_callrate = 0.0
        tr_rows = 0
        tr_loss_val = tr_nll_val = tr_H_val = 0.0

        for batch in dl_tr:
            if batch is None: continue
            X_cat, group_ptrs, y_all = batch
            X_cat = X_cat.to(device, non_blocking=True)
            group_ptrs = group_ptrs.to(device, non_blocking=True)
            y_all = y_all.to(device, non_blocking=True)

            scores = model(X_cat)
            loss, nll, H = grouped_loss(
                scores, group_ptrs, y_all,
                skip_weight=args.skip_weight,
                label_smoothing=args.label_smoothing,
                entropy_reg=args.entropy_reg,
            )
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            # metrics
            Nrows = group_ptrs.size(0)
            with torch.no_grad():
                acc = 0.0; callrate = 0.0
                for i in range(Nrows):
                    s, L = group_ptrs[i]
                    pred = scores[s:s+L].argmax().item()
                    if pred == int(y_all[i].item()):
                        acc += 1.0
                    if pred != 0:
                        callrate += 1.0
            tr_rows += Nrows
            tr_acc += acc
            tr_callrate += callrate
            tr_loss_val += float(loss.item()) * Nrows
            tr_nll_val  += float(nll) * Nrows
            tr_H_val    += float(H) * Nrows

        tr_acc /= max(1, tr_rows)
        tr_callrate /= max(1, tr_rows)
        tr_loss_val /= max(1, tr_rows)
        tr_nll_val  /= max(1, tr_rows)
        tr_H_val    /= max(1, tr_rows)

        va_loss_dummy, va_acc, va_callrate, va_n = eval_epoch(model, dl_va, device)
        dt = time.time() - t0
        print(f"[E{epoch:03d}] tr_loss={tr_loss_val:.4f} (nll={tr_nll_val:.4f}, H={tr_H_val:.3f}) "
              f"acc={tr_acc*100:.1f}% callrate={tr_callrate*100:.1f}% | "
              f"va_acc={va_acc*100:.1f}% va_callrate={va_callrate*100:.1f}% rows tr/va={tr_rows}/{va_n} ({dt:.1f}s)")

        if va_acc >= best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "in_dim": in_dim, "args": vars(args)}, out_dir / "calls_best.pt")

    print(f"[DONE] best_val_acc={best_acc*100:.1f}% -> {out_dir/'calls_best.pt'}")

    # ONNX export
    try:
        model.eval()
        dummy = torch.zeros(1, in_dim, dtype=torch.float32)
        onnx_path = out_dir / "calls_policy.onnx"
        torch.onnx.export(model.to("cpu"), dummy, str(onnx_path),
                          input_names=["feat"], output_names=["score"],
                          opset_version=12, do_constant_folding=True,
                          dynamic_axes={"feat": {0: "batch"}, "score": {0: "batch"}})
        print(f"[EXPORT] ONNX -> {onnx_path}")
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")


if __name__ == "__main__":
    main()
