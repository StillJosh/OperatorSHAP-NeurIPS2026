"""Run bootstrapping, baselines, coalition sampling, query encoding, and hard normalisation."""

import json
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

DEBUG_SECTIONS: tuple[str, ...] = ("model", "attribution")
_START_TIME = time.time()


def approx_budget_segment(approx: str, budget) -> str:
    """Run-name segment of a classical run: ``gt`` for ProxySHAP at 300k, ``exact``, else ``ks_b{budget}`` / ``ps_b{budget}``."""
    if approx == "exact":
        return "exact"
    if approx == "proxyshap" and int(budget) == 300000:
        return "gt"
    return f"{dict(kernelshap='ks', proxyshap='ps')[approx]}_b{budget}"


OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("approx_budget_segment", approx_budget_segment, replace=True)


def apply_debug_overrides(cfg: DictConfig) -> None:
    """Deep-merge ``cfg.<section>.debug_params`` into each section of :data:`DEBUG_SECTIONS` when ``cfg.debug`` is set."""
    if not cfg.debug:
        return
    for section in DEBUG_SECTIONS:
        if section in cfg and "debug_params" in cfg[section]:
            OmegaConf.update(cfg, section, OmegaConf.merge(cfg[section], cfg[section].debug_params), merge=False)


def setup_run(cfg: DictConfig) -> Path:
    """Logging, seeding, debug overrides, and the run directory ``cfg.out_dir`` of one entry point.

    Returns
    -------
    Path
        The created run directory.
    """
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    apply_debug_overrides(cfg)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global _START_TIME
    _START_TIME = time.time()
    return out_dir


def finish_run(cfg: DictConfig, **summary) -> None:
    """Write ``summary.json`` with the wall-clock seconds since :func:`setup_run`, the device, and ``summary``."""
    summary = {"seconds": time.time() - _START_TIME, "device": str(cfg.device), **summary}
    (Path(cfg.out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))


def crop_to_cutout(mesh_pos: np.ndarray, cutout: tuple[float, ...]) -> np.ndarray:
    """Keep-mask of the nodes inside the physical ``[xmin, xmax, ymin, ymax]`` box."""
    xmin, xmax, ymin, ymax = cutout
    return (mesh_pos[:, 0] >= xmin) & (mesh_pos[:, 0] <= xmax) & (mesh_pos[:, 1] >= ymin) & (mesh_pos[:, 1] <= ymax)


def build_baseline(u0: torch.Tensor, mode: str, stat: torch.Tensor | None) -> torch.Tensor:
    """Inactive-node baseline of the same shape as the node-first sample ``u0`` ``(n_points, C)``.

    Parameters
    ----------
    mode : str
        ``"zero"``, ``"sample_mean"``, ``"per_node_mean"``, or ``"global_mean"``.
    stat : torch.Tensor or None
        Train statistic in the served frame: ``(n_points, C)`` for
        ``per_node_mean``, ``(C,)`` for ``global_mean``.
    """
    if mode == "zero":
        return torch.zeros_like(u0)
    if mode == "sample_mean":
        return u0.mean(dim=0, keepdim=True).expand_as(u0).contiguous()
    if mode == "per_node_mean":
        return stat
    if mode == "global_mean":
        return stat.reshape(1, -1).expand_as(u0).contiguous()
    raise ValueError(f"baseline must be zero | sample_mean | per_node_mean | global_mean; got {mode!r}")


def sample_coalitions(n_samples: int, n_pairs: int, n_players: int, device: torch.device, sampling: str = "shapley_kernel") -> torch.Tensor:
    """Binary coalitions ``(n_samples, 2 * n_pairs, n_players)`` in ``[S, complement(S)]`` pairs.

    Parameters
    ----------
    sampling : str
        ``"shapley_kernel"`` draws the coalition size with probability
        proportional to ``1 / (s (n - s))`` and the members uniformly, so
        each coalition has the Shapley-kernel weight; ``"uniform"`` draws
        the size uniformly from ``1 .. n - 1``.
    """
    if sampling == "shapley_kernel":
        sizes_range = torch.arange(1, n_players, device=device)
        weights = 1.0 / (sizes_range * (n_players - sizes_range))
        sizes = torch.multinomial(weights, n_samples * n_pairs, replacement=True).reshape(n_samples, n_pairs, 1) + 1
    else:
        sizes = torch.randint(1, n_players, (n_samples, n_pairs, 1), device=device)
    # The |S| lowest ranks of iid noise join the coalition.
    ranks = torch.rand(n_samples, n_pairs, n_players, device=device).argsort(dim=-1).argsort(dim=-1)
    masks_half = (ranks < sizes).float()
    return torch.cat([masks_half, 1.0 - masks_half], dim=1)


def encode_query_nodes(coords: torch.Tensor, x_star_nodes: torch.Tensor, radius: float, kernel_type: str = "gaussian") -> torch.Tensor:
    """Bump over the distance of every node to each query node, ``(n_queries, n_nodes)``.

    ``radius`` is a distance in the ``[0, 1]^d`` frame of ``coords``; the
    bump peaks at 1 on the query node: tent ``max(0, 1 - d / r)``,
    gaussian ``exp(-d² / 2r²)``, box ``d <= r``.
    """
    d = (coords.unsqueeze(0) - coords[x_star_nodes].unsqueeze(1)).norm(dim=-1) / float(radius)
    if kernel_type == "tent":
        return (1.0 - d).clamp_min(0.0)
    if kernel_type == "gaussian":
        return torch.exp(-0.5 * d * d)
    if kernel_type == "box":
        return (d <= 1.0).float()
    raise ValueError(f"query kernel must be tent | gaussian | box; got {kernel_type!r}")


def predict_coalition_values(masks: torch.Tensor, phi_norm: torch.Tensor, v_empty: torch.Tensor) -> torch.Tensor:
    """Linear game model ``v(empty) + S · phi``: masks ``(B, K, n)``, phi ``(B, Q, n)`` → ``(B, K, Q)``."""
    return v_empty.unsqueeze(1) + masks @ phi_norm.transpose(1, 2)


def hard_normalize(phi: torch.Tensor, v_grand: torch.Tensor, v_empty: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Additive efficient normalisation: spread ``(v_grand - v_empty) - sum(phi)`` over the players.

    Parameters
    ----------
    phi : torch.Tensor
        ``(..., n_players)``; the leading dims match ``v_grand`` and ``v_empty``.
    weights : torch.Tensor
        Per-player measures ``(n_players,)``; the share of a player is
        ``w_i / sum(w)``, so zero-measure players stay null.
    """
    residual = (v_grand - v_empty) - phi.sum(dim=-1)
    w = weights.to(phi.device)
    return phi + residual.unsqueeze(-1) * (w / w.sum())


class _ForwardCounter:
    """Process-global count of backbone evaluations (value-function work, not explainer forwards)."""

    def __init__(self) -> None:
        self.inputs = 0
        self.calls = 0

    def add(self, n_inputs: int) -> None:
        self.inputs += int(n_inputs)
        self.calls += 1

    def reset(self) -> None:
        self.inputs = 0
        self.calls = 0

    def snapshot(self) -> dict[str, int]:
        return {"backbone_evals": self.inputs, "backbone_calls": self.calls}


backbone_forward_counter = _ForwardCounter()
