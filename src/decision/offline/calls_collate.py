from __future__ import annotations
from typing import List, Optional, Tuple
import torch

def calls_collator(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]):
    """
    Input items: (X_i, y_i)
      X_i: (K_i, D)
      y_i: scalar in [0..K_i-1]
    Output:
      X_cat: (sum K_i, D)
      group_ptrs: (N, 2) tensor of (start, length) per row
      y_all: (N,) labels (index relative to each row)
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    D = batch[0][0].shape[1]
    group_ptrs = []
    feat_list = []
    labels = []
    cursor = 0
    for X, y in batch:
        K = X.shape[0]
        feat_list.append(X)
        group_ptrs.append((cursor, K))
        labels.append(y)
        cursor += K
    X_cat = torch.cat(feat_list, dim=0)  # (sumK, D)
    group_ptrs = torch.tensor(group_ptrs, dtype=torch.long)  # (N,2)
    y_all = torch.stack(labels, dim=0)  # (N,)
    return X_cat, group_ptrs, y_all
