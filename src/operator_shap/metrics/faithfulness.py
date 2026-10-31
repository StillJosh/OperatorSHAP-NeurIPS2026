"""Faithfulness of an attribution as the R² of its linear game model on Shapley-kernel coalitions."""

import torch
from torch import Tensor

from operator_shap.utils import sample_coalitions


def kernel_r2(backbone, domain, phi: Tensor, n_coalitions: int, seed: int, *, chunk: int = 512, batch_size: int = 512) -> Tensor:
    """R² of ``v(empty) + S · phi`` against the true ``v(S)`` on ``n_coalitions`` Shapley-kernel coalitions.

    The coalitions are drawn once on a forked CPU generator seeded with
    ``seed`` and shared by every sample and method.

    Parameters
    ----------
    domain : Domain
        Game of the samples ``phi`` was computed for.
    phi : Tensor
        ``(..., n_samples, n_queries, n_players)``; leading dims (one per method) share the game evaluations.
    chunk : int
        Coalitions per :meth:`Domain.coalition_values` call.
    batch_size : int
        Inputs per backbone forward.

    Returns
    -------
    Tensor
        ``(..., n_samples, n_queries)`` on CPU.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        masks = sample_coalitions(1, n_coalitions // 2, domain.n_players, torch.device("cpu"))[0].to(domain.device)
    masks = masks.unsqueeze(0).expand(domain.n_samples, -1, -1)
    with torch.no_grad():
        v = torch.cat([domain.coalition_values(backbone, masks[:, k0 : k0 + chunk], batch_size) for k0 in range(0, masks.shape[1], chunk)], dim=1)
        _, v_empty = domain.game_values(backbone)
    pred = v_empty.unsqueeze(-2) + masks @ phi.to(domain.device).transpose(-1, -2)
    ss_res = (pred - v).pow(2).sum(dim=-2)
    ss_tot = (v - v.mean(dim=-2, keepdim=True)).pow(2).sum(dim=-2).clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).cpu()
