"""Generated PDE data from ``{train,val,test}.npz``, served channel-first at any resolution that divides the native grid."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from operator_shap import baseline_stats


class NPZDataset(Dataset):
    """``(u0, y_true)`` pairs of one PDE with the game geometry of the served resolution.

    Samples are indexed by ``(resolution, idx)`` and strided isotropically
    from the native grid. Besides ``u0`` and ``y_true`` ``(C, *spatial)`` a
    sample carries cell-centre ``coords`` in ``[0, 1]^d``, uniform
    ``node_weights``, the query nodes ``x_star_node``, and the ``baseline``
    field strided like ``u0``.

    Parameters
    ----------
    data_dir : str
        Data root holding ``<pde_name>/{train,val,test}.npz``.
    pde_name : str
    input_key, target_key : str
        NPZ keys of the input and the solution.
    t_idx : int or None
        Time index of the target; ``None`` for static problems (Darcy).
    target_var_idx : int or None
        Variable of a multi-variable solution (shallow water stores ``[h, hu, hv]``, ``u0`` is ``h``).
    max_samples : int or None
    split : str
    baseline : str or None
        Inactive-node baseline mode (see :func:`~operator_shap.utils.build_baseline`); ``None`` omits the key.
    x_star : list[list[float]] or None
        Query points in per-dim domain fractions; multiples of ``1 / min(resolutions)`` hit the same native point at every stride.
    n_spatial_dims : int or None
        Needed for 3D data, whose rank equals that of 2D data with channels.
    """

    is_mesh: bool = False
    quantity: str = "channel0"

    def __init__(
        self,
        data_dir: str,
        pde_name: str,
        input_key: str = "u0",
        target_key: str = "solution",
        t_idx: int | None = 50,
        target_var_idx: int | None = None,
        max_samples: int | None = None,
        split: str = "train",
        baseline: str | None = None,
        x_star: list[list[float]] | None = None,
        n_spatial_dims: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) / pde_name
        npz_path = self.data_dir / f"{split}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} not found; create it with `python generate_data.py datagen={pde_name} data_root={data_dir}`")
        data = np.load(npz_path)
        u0 = torch.from_numpy(data[input_key][:max_samples].astype(np.float32))
        target = data[target_key][:max_samples]
        if t_idx is not None:
            target = target[:, t_idx]
        if target_var_idx is not None:
            target = target[:, target_var_idx]
        # Copy only the served snapshot: a view would keep the full trajectory array alive in every loader worker.
        uT = torch.from_numpy(np.ascontiguousarray(target, dtype=np.float32))

        # Channel-first (N, C, *spatial); rank 4 is 2D with channels unless n_spatial_dims says 3D.
        if n_spatial_dims is not None:
            self.n_spatial_dims = int(n_spatial_dims)
        elif u0.ndim in (2, 3, 4):
            self.n_spatial_dims = 1 if u0.ndim == 2 else 2
        else:
            raise ValueError(f"unsupported u0.ndim={u0.ndim}; pass n_spatial_dims for 3D data")
        if u0.ndim == self.n_spatial_dims + 1:
            u0 = u0.unsqueeze(1)
        if uT.ndim == self.n_spatial_dims + 1:
            uT = uT.unsqueeze(1)
        elif uT.ndim != self.n_spatial_dims + 2:
            raise ValueError(f"target ndim={uT.ndim} incompatible with n_spatial_dims={self.n_spatial_dims}")
        self.u0, self.uT = u0, uT
        self.spatial_res = tuple(u0.shape[2:])

        self.x_star = [[float(f) for f in p] for p in x_star] if x_star is not None else None
        if self.x_star is not None and any(len(p) != self.n_spatial_dims for p in self.x_star):
            raise ValueError(f"x_star points {self.x_star} need {self.n_spatial_dims} fractions each")
        self._geometry: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}

        # Train statistic in the native frame: per_node_mean (C, *spatial), global_mean (C,).
        self.baseline_mode = baseline
        self._baseline_stat = baseline_stats.load_stat(self.data_dir, baseline) if baseline in ("per_node_mean", "global_mean") else None

    def __len__(self) -> int:
        return self.u0.shape[0]

    def geometry(self, spatial: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Cell-centre ``coords`` ``(n, d)`` and uniform ``node_weights`` ``(n,)``, cached per resolution."""
        if spatial not in self._geometry:
            axes = [(torch.arange(s, dtype=torch.float32) + 0.5) / s for s in spatial]
            coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(spatial))
            self._geometry[spatial] = (coords, torch.full((coords.shape[0],), 1.0 / coords.shape[0]))
        return self._geometry[spatial]

    def _x_star_node(self, spatial: tuple[int, ...]) -> torch.Tensor:
        """Row-major flat ``(n_queries,)`` query indices at the served resolution."""
        idx = [[int(f * s) for f, s in zip(p, spatial)] for p in self.x_star]
        return torch.from_numpy(np.ravel_multi_index(np.asarray(idx).T, spatial))

    def _baseline_field(self, u0: torch.Tensor, sl: tuple) -> torch.Tensor:
        """Baseline strided like ``u0``."""
        if self.baseline_mode == "zero":
            return torch.zeros_like(u0)
        if self.baseline_mode == "sample_mean":
            return u0.mean(dim=tuple(range(1, u0.ndim)), keepdim=True).expand_as(u0).contiguous()
        if self.baseline_mode == "global_mean":
            return self._baseline_stat.view((u0.shape[0],) + (1,) * (u0.ndim - 1)).expand_as(u0).contiguous()
        return self._baseline_stat[sl]

    def __getitem__(self, idx: tuple[int, int]) -> dict:
        r, i = idx
        stride = self.spatial_res[0] // r
        sl = (slice(None),) + (slice(None, None, stride),) * self.n_spatial_dims
        u0 = self.u0[i][sl]
        spatial = tuple(u0.shape[1:])
        coords, node_weights = self.geometry(spatial)
        out = {"u0": u0, "y_true": self.uT[i][sl], "sample_idx": i, "coords": coords, "node_weights": node_weights}
        if self.x_star is not None:
            out["x_star_node"] = self._x_star_node(spatial)
        if self.baseline_mode is not None:
            out["baseline"] = self._baseline_field(u0, sl)
        return out
