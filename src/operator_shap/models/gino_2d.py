"""Dict-input wrapper of :class:`operator_shap.models.gino.GINO` for 2D meshes."""

from __future__ import annotations

import torch
from neuralop.layers import integral_transform
from neuralop.layers.neighbor_search import NeighborSearch

_segment_csr = integral_transform.segment_csr


def _segment_csr_fp32(src: torch.Tensor, indptr: torch.Tensor, reduction: str, use_scatter: bool = True) -> torch.Tensor:
    """neuralop's neighbourhood reduction in fp32; under bf16 autocast it would average the kernel outputs in bf16."""
    return _segment_csr(src.float(), indptr, reduction, use_scatter)


integral_transform.segment_csr = _segment_csr_fp32


class _CachedNeighborSearch(torch.nn.Module):
    """``NeighborSearch`` that reuses its last result while the geometry is unchanged.

    neuralop searches on every forward, but the coalition forwards of one
    attribution step differ only in the field values, and the MGN airfoil
    mesh is fixed for the whole run. The cached tensors are held, so a
    mesh that is rebuilt per batch still hits. Carries no parameters, so
    the wrapped tree loads the same state dict.
    """

    def __init__(self, search: NeighborSearch) -> None:
        super().__init__()
        self.search = search
        self._geometry: tuple[torch.Tensor, torch.Tensor, float] | None = None
        self._neighbors: dict[str, torch.Tensor] | None = None

    def _hit(self, data: torch.Tensor, queries: torch.Tensor, radius: float) -> bool:
        if self._geometry is None:
            return False
        cached_data, cached_queries, cached_radius = self._geometry
        if radius != cached_radius or data.shape != cached_data.shape or queries.shape != cached_queries.shape:
            return False
        return torch.equal(data, cached_data) and torch.equal(queries, cached_queries)

    def forward(self, data: torch.Tensor, queries: torch.Tensor, radius: float) -> dict[str, torch.Tensor]:
        if not self._hit(data, queries, radius):
            self._neighbors = self.search(data, queries, radius)
            self._geometry = (data, queries, radius)
        return self._neighbors


def _cache_neighbor_searches(net: torch.nn.Module) -> None:
    """Wrap every ``NeighborSearch`` of ``net`` in :class:`_CachedNeighborSearch`, in place."""
    # Collected first: replacing while walking would re-enter the wrapper's own child.
    targets = [(module, name) for module in list(net.modules()) for name, child in module.named_children() if isinstance(child, NeighborSearch)]
    for module, name in targets:
        setattr(module, name, _CachedNeighborSearch(getattr(module, name)))


class GINOBackbone(torch.nn.Module):
    """GINO on ``{input_grid, output_grid, latent_grid, u0}`` batches.

    GINO batches only inputs that share their geometry, so the first
    geometry entry of the batch is used for all of ``u0``.

    Parameters
    ----------
    net : torch.nn.Module
        GINO with ``gno_coord_dim`` 2, or 3 when ``coord_pad_3d`` is set.
    coord_pad_3d : bool
        Append a zero z-coordinate so open3d's KNN takes its fast 3D path;
        weights trained with one setting do not load under the other.
    """

    def __init__(self, net: torch.nn.Module, coord_pad_3d: bool) -> None:
        super().__init__()
        self.coord_pad_3d = bool(coord_pad_3d)
        self.model = net
        _cache_neighbor_searches(self.model)

    @staticmethod
    def _pad3d(coords: torch.Tensor) -> torch.Tensor:
        return torch.cat([coords, torch.zeros_like(coords[..., :1])], dim=-1)

    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        input_geom = data["input_grid"][:1]
        latent_queries = data["latent_grid"][:1]
        output_queries = data.get("output_grid", data["input_grid"])[:1]
        if self.coord_pad_3d:
            input_geom, latent_queries, output_queries = map(self._pad3d, (input_geom, latent_queries, output_queries))
        out = self.model(input_geom=input_geom, latent_queries=latent_queries, output_queries=output_queries, x=data.get("u0"))
        # Scalar targets are served as (n_nodes,): drop the channel axis so the loss broadcasts.
        return out.squeeze(-1) if out.ndim == 3 and out.shape[-1] == 1 else out
