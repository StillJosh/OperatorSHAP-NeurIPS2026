"""``shapiq.Game`` over the backbone's value function on a :class:`~operator_shap.domain.Domain`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import shapiq
from torch import nn

if TYPE_CHECKING:
    from operator_shap.domain import Domain


def build_approximator(kind: str, n_players: int, *, seed: int | None):
    """shapiq approximator ``"proxyshap"`` or ``"kernelshap"``; ``None`` for ``"exact"`` (built per game)."""
    if kind == "proxyshap":
        from shapiq.approximator import ProxySHAP

        return ProxySHAP(n=n_players, max_order=1, index="SII", pairing_trick=True, random_state=seed)
    if kind == "kernelshap":
        from shapiq.approximator import KernelSHAP

        return KernelSHAP(n=n_players, pairing_trick=True, random_state=seed)
    if kind == "exact":
        return None
    raise ValueError(f"approximator must be proxyshap | kernelshap | exact; got {kind!r}")


def run_approximator(kind: str, approximator, game, *, n_players: int, budget: int):
    """Shapley values of one game as shapiq ``InteractionValues``."""
    if kind == "exact":
        from shapiq import ExactComputer

        return ExactComputer(game=game, n_players=n_players)("SV")
    return approximator.approximate(budget=budget, game=game)


def extract_shapley_values(iv, n_players: int) -> np.ndarray:
    """Per-player values ``(n_players,)`` of a shapiq ``InteractionValues``."""
    return np.array([iv[(p,)] for p in range(n_players)], dtype=np.float32)


class CoalitionGame(shapiq.Game):
    """Cooperative game whose value function is ``domain.forward_masked`` at one query point.

    Parameters
    ----------
    backbone : nn.Module
    domain : Domain
        Single-sample domain (partition, query points, masked forward).
    max_batch : int
        Coalitions per backbone forward.
    query : int
        Index into the domain's query points.
    """

    def __init__(self, backbone: nn.Module, domain: Domain, *, max_batch: int, query: int = 0) -> None:
        self.backbone = backbone
        self.domain = domain
        self.query = int(query)
        self.max_batch = int(max_batch)
        self.n_players = domain.n_players
        empty_value = float(self.value_function(np.zeros((1, self.n_players), dtype=bool))[0])
        super().__init__(n_players=self.n_players, normalize=True, normalization_value=empty_value)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """``(n_coalitions, n_players)`` → ``(n_coalitions,)`` float32."""
        return self.domain.forward_masked(self.backbone, coalitions, self.max_batch, self.query)
