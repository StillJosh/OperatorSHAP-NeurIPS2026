"""Integrated Gradients along the straight path from the baseline to ``u0``.

The per-node density ``h ⊙ φ_IG`` with ``h = u0 - baseline`` and
``φ_IG(y) = ∫_0^1 ∂_y G(baseline + t h)[x*] dt`` gives any masked
attribution as ``(density * mask).sum()`` and the attribution of a
partition as :meth:`~operator_shap.domain.Domain.discretize`.
"""

from __future__ import annotations

import torch

from operator_shap.domain import Domain


@torch.enable_grad()
def ig_density(backbone: torch.nn.Module, domain: Domain, *, n_steps: int, batch_size: int) -> torch.Tensor:
    """IG density ``h ⊙ φ_IG`` of every (sample, query), ``(n_samples, n_queries, n_points)``.

    Midpoint rule with ``n_steps`` points, summed over channels so that
    ``density.sum(-1) ≈ v_grand - v_empty``.

    Parameters
    ----------
    n_steps : int
        Integration points along the path.
    batch_size : int
        Path points per backbone forward/backward pass.
    """
    ts = (torch.arange(n_steps, device=domain.device, dtype=domain.u0.dtype) + 0.5) / n_steps
    out = torch.empty(domain.n_samples, domain.n_queries, domain.n_points, device=domain.device)
    for i in range(domain.n_samples):
        h = domain.u0[i] - domain.baseline[i]                                  # (n_points, C)
        for q in range(domain.n_queries):
            grad = torch.zeros_like(h)
            for k0 in range(0, n_steps, batch_size):
                t = ts[k0 : k0 + batch_size].view(-1, 1, 1)
                path = (domain.baseline[i] + t * h).requires_grad_(True)       # (chunk, n_points, C)
                x_idx = domain.x_star[i : i + 1, q : q + 1].expand(path.shape[0], 1)
                y = domain.read(backbone, domain.query_values(backbone, path, x_idx), x_idx.new_full((path.shape[0],), i), slice(q, q + 1))
                grad += torch.autograd.grad(y.sum(), path)[0].sum(dim=0)
            out[i, q] = (h * grad / n_steps).sum(dim=-1)
    return out.detach()
