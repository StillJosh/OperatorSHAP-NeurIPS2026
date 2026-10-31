"""Agreement of an attribution with a reference, per sample along the last dim: ``(..., n)`` → ``(...)``."""

import torch
from torch import Tensor


def pearson_r(pred: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    """Pearson correlation."""
    pred_c = pred - pred.mean(dim=-1, keepdim=True)
    target_c = target - target.mean(dim=-1, keepdim=True)
    return (pred_c * target_c).sum(dim=-1) / (pred_c.norm(dim=-1) * target_c.norm(dim=-1)).clamp_min(eps)


def nrmse(pred: Tensor, target: Tensor, eps: float = 1e-12) -> Tensor:
    """``|pred - target| / |target|``."""
    return torch.sqrt((pred - target).pow(2).sum(dim=-1)) / torch.sqrt(target.pow(2).sum(dim=-1)).clamp_min(eps)
