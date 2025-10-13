from __future__ import annotations
from typing import List, Optional, Tuple
import torch

def bc_collator(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    X = torch.stack([b[0] for b in batch], dim=0)
    M = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    return X, M, y
