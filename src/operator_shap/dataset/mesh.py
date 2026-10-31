"""MeshGraphNets datasets (cylinder_flow, airfoil) as GINO sample dicts.

One ``.npz`` per trajectory, written by :mod:`operator_shap.datagen.mgn_convert`
into ``<ds_name>/{train,val,test}/``, holds ``mesh_pos (n, 2)``,
``node_type (n,)``, ``cells (m, 3)``, and ``velocity (T, n, 2)``. A sample
is the velocity field at ``t0`` (``u0``) and at ``t0 + t_lookahead``
(``y_true``) on the mesh normalised to ``[0, 1]^2``, plus the game
geometry (``coords``, lumped-mass ``node_weights``, ``x_star_node``).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from operator_shap import baseline_stats
from operator_shap.domain import node_quadrature_weights
from operator_shap.utils import build_baseline, crop_to_cutout

logger = logging.getLogger(__name__)


def _make_latent_grid(size: int) -> torch.Tensor:
    ax = torch.linspace(0.0, 1.0, size)
    gx, gy = torch.meshgrid(ax, ax, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def minmax_normalize(points, lo, hi):
    """Map ``points`` into ``[0, 1]`` per axis given the extent ``lo`` / ``hi`` (numpy or torch)."""
    return (points - lo) / (hi - lo).clip(min=1e-12)


def nearest_nodes(coords: torch.Tensor, targets) -> torch.Tensor:
    """Indices ``(n_targets,)`` of the mesh nodes closest to the ``(n_targets, 2)`` points."""
    t = torch.as_tensor(targets, dtype=coords.dtype, device=coords.device).reshape(-1, coords.shape[-1])
    return torch.cdist(t, coords).argmin(dim=-1)


def t0_options(n_timesteps: int, t0_stride: int, t_lookahead: int) -> list[int]:
    """Valid ``t0`` draws ``{0, t0_stride, ...}`` with ``t0 + t_lookahead <= n_timesteps - 1``."""
    upper = n_timesteps - t_lookahead
    if upper <= 0:
        raise ValueError(f"trajectory length {n_timesteps} must exceed t_lookahead={t_lookahead}")
    return list(range(0, upper, t0_stride))


class MGNDataset(Dataset):
    """MeshGraphNets trajectories for the GINO ``u(t0) → u(t0 + t_lookahead)`` task.

    Parameters
    ----------
    data_dir : str
        Data root holding ``<ds_name>/{train,val,test}/*.npz``.
    ds_name : str
        ``"mgn_cylinder_flow"`` or ``"mgn_airfoil"``.
    split : str
        ``"train"``, ``"val"``, or ``"test"``; the on-disk folder.
    t0, t_star : int
        Fixed input and target timesteps when ``t0_stride`` is ``None``; ``t0`` is also the
        timestep :mod:`~operator_shap.datagen.compute_baseline_stats` averages.
    latent_grid_size : int
        Side length of the GINO latent grid on ``[0, 1]^2``.
    cutout : list[float] or None
        Physical ``[xmin, xmax, ymin, ymax]`` box; nodes and triangles outside are dropped.
    vel_scale : float or None
        Divisor of ``u0`` and ``y_true`` bringing the velocities to O(1).
    t0_stride, t_lookahead : int or None
        Time-window augmentation: ``t0`` is drawn from :func:`t0_options`
        per call on the train split and by sample index on val/test.
    max_samples : int or None
    x_star : list[list[float]] or None
        Physical query points; each sample carries the nearest nodes
        ``x_star_node (n_queries,)``.
    baseline : str or None
        Inactive-node baseline mode served as ``baseline`` (see
        :func:`~operator_shap.utils.build_baseline`); the stat-backed
        modes read ``baseline_stats.npz`` and get the crop and velocity
        scaling of ``u0``. ``None`` omits the key.
    """

    is_mesh = True
    quantity = "norm"

    def __init__(
        self,
        data_dir: str,
        ds_name: str,
        *,
        split: str = "train",
        t0: int = 0,
        t_star: int = 100,
        latent_grid_size: int = 64,
        cutout: list[float] | None = None,
        vel_scale: float | None = None,
        t0_stride: int | None = None,
        t_lookahead: int | None = None,
        max_samples: int | None = None,
        x_star: list[list[float]] | None = None,
        baseline: str | None = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.data_dir = Path(data_dir) / ds_name
        split_dir = self.data_dir / split
        self._files = sorted(split_dir.glob("*_[0-9][0-9][0-9][0-9].npz"))
        if not self._files:
            raise FileNotFoundError(f"no .npz files in {split_dir}; create them with `python download_mgn.py --data-root {data_dir}`")
        if max_samples is not None:
            self._files = self._files[:max_samples]
        self.split = split
        self.t0, self.t_star = int(t0), int(t_star)
        self._latent = _make_latent_grid(latent_grid_size)
        self.spatial_res = (latent_grid_size, latent_grid_size)
        self.n_spatial_dims = 2
        self.x_star = [[float(v) for v in p] for p in x_star] if x_star is not None else None
        self.cutout = tuple(float(v) for v in cutout) if cutout is not None else None
        self.vel_scale = float(vel_scale) if vel_scale is not None else None

        self.t_lookahead = int(t_lookahead) if t_lookahead is not None else None
        self._t0_options = None
        if t0_stride is not None:
            if self.t_lookahead is None:
                raise ValueError("t0_stride requires t_lookahead to be set")
            with np.load(self._files[0]) as z:
                n_timesteps = int(z["velocity"].shape[0])
            self._t0_options = t0_options(n_timesteps, int(t0_stride), self.t_lookahead)

        # Train statistic in the served frame; zero / sample_mean are built per sample.
        self.baseline_mode = baseline
        self._baseline_stat = None
        if baseline in ("per_node_mean", "global_mean"):
            stat = baseline_stats.load_stat(self.data_dir, baseline)
            if baseline == "per_node_mean" and self.cutout is not None:
                with np.load(self._files[0]) as z:
                    stat = stat[torch.from_numpy(crop_to_cutout(z["mesh_pos"], self.cutout))]
            if self.vel_scale is not None:
                stat = stat / self.vel_scale
            self._baseline_stat = stat

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx) -> dict:
        i = idx[1]
        with np.load(self._files[i]) as z:
            mesh_pos, node_type, vel = z["mesh_pos"], z["node_type"], z["velocity"]
            cells = z["cells"]

        # Cutout: drop nodes outside the box and triangles with a dropped corner.
        if self.cutout is not None:
            keep = crop_to_cutout(mesh_pos, self.cutout)
            mesh_pos, node_type, vel = mesh_pos[keep], node_type[keep], vel[:, keep, :]
            remap = -np.ones(len(keep), dtype=np.int64)
            remap[np.nonzero(keep)[0]] = np.arange(int(keep.sum()))
            cells = remap[cells[keep[cells].all(axis=1)]]

        # Time window: random on train, by sample index on val/test.
        if self._t0_options is not None:
            t0_i = random.choice(self._t0_options) if self.split == "train" else self._t0_options[i % len(self._t0_options)]
            t_star_i = t0_i + self.t_lookahead
        else:
            t0_i, t_star_i = self.t0, self.t_star
        u0 = vel[t0_i].astype(np.float32)
        y_true = vel[t_star_i].astype(np.float32)
        if self.vel_scale is not None:
            u0, y_true = u0 / self.vel_scale, y_true / self.vel_scale

        # Geometry in the per-sample normalised frame [0, 1]^2 (per axis, so the aspect ratio is not preserved).
        c_min, c_max = mesh_pos.min(axis=0, keepdims=True), mesh_pos.max(axis=0, keepdims=True)
        coords = torch.from_numpy(minmax_normalize(mesh_pos, c_min, c_max).astype(np.float32))
        node_weights = node_quadrature_weights(coords, torch.from_numpy(cells.astype(np.int64)))

        out = {
            "input_grid": coords,
            "output_grid": coords,
            "latent_grid": self._latent,
            "coords": coords,
            "node_weights": node_weights,
            "u0": torch.from_numpy(u0),
            "y_true": torch.from_numpy(y_true),
            "node_type": torch.from_numpy(node_type),
            "mesh_pos_raw": torch.from_numpy(mesh_pos.astype(np.float32)),
            "coord_min": torch.from_numpy(c_min.squeeze(0).astype(np.float32)),
            "coord_max": torch.from_numpy(c_max.squeeze(0).astype(np.float32)),
            "sample_idx": int(i),
        }
        if self.baseline_mode is not None:
            out["baseline"] = build_baseline(out["u0"], self.baseline_mode, self._baseline_stat)
        if self.x_star is not None:
            out["x_star_node"] = nearest_nodes(coords, minmax_normalize(torch.tensor(self.x_star), out["coord_min"], out["coord_max"]))
        return out
