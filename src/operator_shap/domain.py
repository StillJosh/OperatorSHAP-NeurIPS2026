"""The coalition game of a batch, shared by FastSHAP, classical SHAP, IG, and the metrics.

A sample is a set of nodes with coordinates in ``[0, 1]^d`` and quadrature
weights; a grid is the special case of lattice nodes at the cell centres
with uniform weights. The players are the ``n_per_dim ** d`` boxes of the
unit cube. Inside the domain every field is node-first,
``(n_samples, n_points, n_channels)``; the backbone's own layout
(channel-first tensor for an FNO, node dict with the geometry for GINO)
is confined to :meth:`Domain._pack` and :meth:`Domain._field`.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from operator_shap.utils import backbone_forward_counter, encode_query_nodes, hard_normalize

logger = logging.getLogger(__name__)

GINO_GEOMETRY_KEYS = ("input_grid", "output_grid", "latent_grid")


def n_per_dim_from_players(n_players: int, n_spatial_dims: int) -> int:
    """Per-axis box count of an isotropic ``n_players`` partition; ``n_players`` must be an exact power."""
    n_per_dim = int(round(n_players ** (1.0 / n_spatial_dims)))
    if n_per_dim ** n_spatial_dims != n_players:
        raise ValueError(f"{n_spatial_dims}D partition needs n_players = n_per_dim ** {n_spatial_dims}; got {n_players}")
    return n_per_dim


def node_quadrature_weights(coords: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Lumped-mass quadrature weights ``(n_nodes,)`` of a triangle mesh: a third of every incident triangle's area."""
    tri = coords[cells]                                          # (m, 3, 2)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    share = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).abs() / 3.0
    w = torch.zeros(coords.shape[0])
    for k in range(3):
        w.scatter_add_(0, cells[:, k], share)
    return w


class Domain:
    """Partition, masked value function, and coarsening of one batch.

    Build via :meth:`from_batch`. A coalition's value is the backbone's
    scalar field of the masked input read at ``x*``; the grand- and
    empty-coalition values are memoised.

    Parameters
    ----------
    u0, baseline : torch.Tensor
        Node-first fields ``(n_samples, n_points, n_channels)`` on ``device``.
    coords : torch.Tensor
        Node coordinates ``(n_points, d)`` in ``[0, 1]^d``, shared by the batch.
    node_weights : torch.Tensor
        Quadrature weights ``(n_points,)``; coarsening integrates
        densities with them and the normalisation measures are the
        per-player masses.
    x_star : torch.Tensor
        Long node indices of the query points, ``(n_samples, n_queries)``.
    n_players : int
        Number of boxes of the partition, an exact ``d``-th power.
    device : str
    spatial : tuple[int, ...] or None, keyword-only
        Channel-first spatial shape a grid backbone expects; ``None``
        for a mesh backbone taking node dicts.
    static : dict or None, keyword-only
        Mesh only: the GINO geometry tensors merged into every input dict.
    quantity : str, keyword-only
        Scalar read off the backbone output: ``"channel0"`` or the
        channel ``"norm"``.
    readout : str, keyword-only
        Value of a coalition: the ``"point"`` scalar at ``x*``, or the
        ``"midpoint"`` event that this scalar is closer to its value at
        ``u0`` than to its value at the baseline.
    query_encoding : tuple[float, str] or None, keyword-only
        ``(radius, kernel)`` of the query bump; builds the explainer input
        ``u0`` repeated per query with the bump as an extra channel.
        ``None`` for domains that never call the explainer.
    max_batch : int, keyword-only
        Inputs per backbone forward in :meth:`game_values`.
    """

    def __init__(
        self,
        u0: torch.Tensor,
        baseline: torch.Tensor,
        coords: torch.Tensor,
        node_weights: torch.Tensor,
        x_star: torch.Tensor,
        n_players: int,
        device: str,
        *,
        spatial: tuple[int, ...] | None = None,
        static: dict | None = None,
        quantity: str = "channel0",
        readout: str = "point",
        query_encoding: tuple[float, str] | None = None,
        max_batch: int = 512,
    ) -> None:
        if baseline.shape != u0.shape:
            raise ValueError(f"baseline shape {tuple(baseline.shape)} != u0 shape {tuple(u0.shape)}")
        if readout not in ("point", "midpoint"):
            raise ValueError(f"readout must be 'point' or 'midpoint'; got {readout!r}")
        self.u0, self.baseline, self.coords, self.node_weights = u0, baseline, coords, node_weights
        self.x_star = x_star
        self.device = device
        self.spatial = spatial
        self.quantity = quantity
        self.readout = readout
        self._static = static or {}
        self._max_batch = int(max_batch)
        self._raw_ends: tuple[torch.Tensor, torch.Tensor] | None = None

        # Player of every node: row-major box index over the unit cube.
        self.n_players = int(n_players)
        self.n_per_dim = n_per_dim_from_players(self.n_players, coords.shape[1])
        box = (coords * self.n_per_dim).long().clamp(0, self.n_per_dim - 1)
        self.player_of_node = torch.zeros(coords.shape[0], dtype=torch.long, device=device)
        for k in range(coords.shape[1]):
            self.player_of_node = self.player_of_node * self.n_per_dim + box[:, k]
        self.cell_measures = self.discretize(node_weights)

        expl = u0
        if query_encoding is not None:
            radius, kernel = query_encoding
            enc = encode_query_nodes(coords, x_star.reshape(-1), float(radius), str(kernel)).unsqueeze(-1)
            expl = torch.cat([u0.repeat_interleave(self.n_queries, dim=0), enc], dim=-1)
        self._explainer_batch = self._pack(expl)

    @classmethod
    def from_batch(
        cls,
        data: dict,
        n_players: int | str,
        device: str,
        *,
        mesh: bool,
        quantity: str,
        readout: str = "point",
        query_encoding: tuple[float, str] | None = None,
        max_batch: int = 512,
        random_query: bool = False,
    ) -> "Domain":
        """Domain of one loader batch (``u0``, ``baseline``, ``coords``, ``node_weights``, ``x_star_node``).

        ``mesh`` selects the backbone layout: node dicts with the GINO
        geometry (one mesh per batch) or channel-first grids.
        ``n_players="identity"`` makes every node a player. With
        ``random_query`` one query node per sample is drawn with the
        quadrature weights (uniform over the domain) instead of the
        served ``x_star_node``.
        """
        u0, baseline = data["u0"].to(device), data["baseline"].to(device)
        coords, weights = data["coords"][0].to(device), data["node_weights"][0].to(device)
        if mesh:
            spatial, static = None, {k: data[k][:1].to(device) for k in GINO_GEOMETRY_KEYS}
        else:
            spatial, static = tuple(u0.shape[2:]), None
            u0, baseline = u0.flatten(2).transpose(1, 2), baseline.flatten(2).transpose(1, 2)
        if random_query:
            x_star = torch.multinomial(weights.expand(u0.shape[0], -1), 1)
        else:
            x_star = data["x_star_node"].to(device)
        n_players = u0.shape[1] if n_players == "identity" else int(n_players)
        return cls(
            u0, baseline, coords, weights, x_star, n_players, device, spatial=spatial, static=static,
            quantity=quantity, readout=readout, query_encoding=query_encoding, max_batch=max_batch,
        )

    # ------------------------------------------------------------------
    # Partition
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return int(self.u0.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.u0.shape[1])

    @property
    def n_queries(self) -> int:
        return int(self.x_star.shape[1])

    def expand_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Player-level mask ``(..., n_players)`` → node-level ``(..., n_points)``."""
        return mask.index_select(-1, self.player_of_node.to(mask.device))

    def discretize(self, field: torch.Tensor) -> torch.Tensor:
        """Sum a node field ``(..., n_points)`` per player → ``(..., n_players)``."""
        idx = self.player_of_node.to(field.device).expand(field.shape)
        return torch.zeros(*field.shape[:-1], self.n_players, device=field.device, dtype=field.dtype).scatter_add_(-1, idx, field)

    # ------------------------------------------------------------------
    # Backbone layout
    # ------------------------------------------------------------------

    def _pack(self, u0_flat: torch.Tensor) -> dict:
        """Node-first ``(n_inputs, n_points, n_channels)`` → backbone input batch."""
        if self.spatial is None:
            return {"u0": u0_flat, **self._static}
        return {"u0": u0_flat.transpose(1, 2).reshape(u0_flat.shape[0], -1, *self.spatial)}

    def _field(self, pred: torch.Tensor) -> torch.Tensor:
        """Backbone output → per-node scalar field ``(n_inputs, n_points)``."""
        if self.spatial is not None:
            pred = pred.flatten(2).transpose(1, 2)
        if pred.ndim == 2:
            return pred
        return pred[..., 0] if self.quantity == "channel0" else pred.norm(dim=-1)

    # ------------------------------------------------------------------
    # Value function
    # ------------------------------------------------------------------

    def query_values(self, backbone: nn.Module, u0_flat: torch.Tensor, x_idx: torch.Tensor) -> torch.Tensor:
        """Backbone scalars of node-first inputs at the ``(n_inputs, n_queries)`` query indices; differentiable."""
        backbone_forward_counter.add(u0_flat.shape[0])
        return self._field(backbone(self._pack(u0_flat))).gather(1, x_idx)

    @torch.no_grad()
    def _values(self, backbone: nn.Module, u0_flat: torch.Tensor, x_idx: torch.Tensor, max_batch: int) -> torch.Tensor:
        """Chunked :meth:`query_values`."""
        out = torch.empty(x_idx.shape, device=self.device)
        for i in range(0, u0_flat.shape[0], max_batch):
            out[i : i + max_batch] = self.query_values(backbone, u0_flat[i : i + max_batch], x_idx[i : i + max_batch])
        return out

    def coalition_values(self, backbone: nn.Module, masks: torch.Tensor, max_batch: int) -> torch.Tensor:
        """Masks ``(n_samples, n_coalitions, n_players)`` → ``(n_samples, n_coalitions, n_queries)``.

        Fractional masks blend linearly between ``u0`` and ``baseline``.
        """
        n_samples, n_coalitions, _ = masks.shape
        m = self.expand_mask(masks).unsqueeze(-1)
        blended = m * self.u0.unsqueeze(1) + (1.0 - m) * self.baseline.unsqueeze(1)
        v = self._values(
            backbone, blended.reshape(n_samples * n_coalitions, self.n_points, -1),
            self.x_star.repeat_interleave(n_coalitions, dim=0), max_batch,
        )
        samples = torch.arange(n_samples, device=self.device).repeat_interleave(n_coalitions)
        return self.read(backbone, v, samples).reshape(n_samples, n_coalitions, -1)

    def read(self, backbone: nn.Module, raw: torch.Tensor, samples: torch.Tensor, queries: slice = slice(None)) -> torch.Tensor:
        """Coalition values of backbone scalars ``raw`` ``(n_inputs, n_queries)`` of inputs built from ``samples``.

        ``"point"`` returns ``raw``; ``"midpoint"`` the indicator that
        ``raw`` is closer to the sample's value at ``u0`` than at the
        baseline, differentiable with gradient zero almost everywhere. ``queries`` selects the query columns ``raw`` holds.
        """
        if self.readout == "point":
            return raw
        grand, empty = (e[samples][:, queries] for e in self._ends(backbone))
        return ((raw - grand).abs() < (raw - empty).abs()).to(raw.dtype) + 0.0 * raw

    def _ends(self, backbone: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Memoised backbone scalars at ``u0`` and at the baseline, each ``(n_samples, n_queries)``."""
        if self._raw_ends is None:
            self._raw_ends = (
                self._values(backbone, self.u0, self.x_star, self._max_batch),
                self._values(backbone, self.baseline, self.x_star, self._max_batch),
            )
        return self._raw_ends

    @torch.no_grad()
    def forward_masked(self, backbone: nn.Module, coalitions: np.ndarray, max_batch: int, query: int = 0) -> np.ndarray:
        """shapiq bridge: ``(n_coalitions, n_players)`` → ``(n_coalitions,)`` float32 at query ``query``."""
        assert self.n_samples == 1, "forward_masked is single-sample; use coalition_values"
        coals = torch.as_tensor(coalitions, dtype=torch.float32, device=self.device)
        v = self.coalition_values(backbone, coals[None], max_batch)[0, :, query]
        return v.cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def game_values(self, backbone: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """``(v_grand, v_empty)``, each ``(n_samples, n_queries)``; memoised."""
        samples = torch.arange(self.n_samples, device=self.device)
        return tuple(self.read(backbone, e, samples) for e in self._ends(backbone))

    # ------------------------------------------------------------------
    # Explainer surface
    # ------------------------------------------------------------------

    def coarsen(self, raw: torch.Tensor) -> torch.Tensor:
        """Explainer densities (one per (sample, query)) → discrete Shapley ``(n_samples, n_queries, n_players)``.

        Each player integrates its density with the quadrature weights.
        """
        varphi = raw.reshape(self.n_samples, self.n_queries, self.n_points)
        return self.discretize(varphi * self.node_weights)

    def explainer_input(self) -> dict:
        """Batched input dict the explainer network is called with."""
        return self._explainer_batch


@torch.no_grad()
def explain(explainer: nn.Module, backbone: nn.Module, domain: Domain) -> torch.Tensor:
    """One explainer pass on a domain → hard-normalised discrete Shapley ``(n_samples, n_queries, n_players)``."""
    phi = domain.coarsen(explainer(domain.explainer_input()))
    v_grand, v_empty = domain.game_values(backbone)
    return hard_normalize(phi, v_grand, v_empty, weights=domain.cell_measures)


def build_domains(cfg, target_loader, *, query_encoding="cfg"):
    """Per-batch domain builder ``domain_of_batch(data, *, eval_mode=False) -> Domain``.

    ``target_loader`` is the split domains are built for; its dataset
    must have been built with a baseline mode so every batch carries the
    aligned game fields. Eval calls use the dataset-served query points,
    training calls draw one random query node per sample.
    ``query_encoding`` overrides the ``attribution.query`` encoding of
    the explainer input: the ``(radius, kernel)`` a loaded explainer was
    trained with, or ``None`` for paths that never call an explainer.
    """
    attr_cfg = cfg.attribution
    ds = target_loader.dataset.dataset                                    # unwrap MultiResSubset
    if query_encoding == "cfg":
        query_encoding = (float(attr_cfg.query.radius), str(attr_cfg.query.kernel))
    n_players = attr_cfg.n_players
    if n_players != "identity" and not isinstance(n_players, int):
        raise ValueError(f"attribution.n_players must be 'identity' or an int; got {n_players!r}")
    max_batch = int(attr_cfg.max_coalition_batch)

    def domain_of_batch(data, *, eval_mode: bool = False) -> Domain:
        return Domain.from_batch(
            data, n_players, cfg.device, mesh=ds.is_mesh, quantity=ds.quantity, readout=str(cfg.dataset.readout),
            query_encoding=query_encoding, max_batch=max_batch, random_query=not eval_mode,
        )

    logger.info(f"domains: n_players={n_players} baseline={ds.baseline_mode} n_eval={len(target_loader.dataset)}")
    return domain_of_batch
