"""Train-split baseline statistics (``per_node_mean``, ``global_mean``) of every listed dataset, written next to its data.

Grids average the input field of ``<data_root>/<pde_name>/train.npz``; meshes average ``velocity[t0]`` over the
train trajectories in ``<data_root>/<ds_name>/train/`` with node weighting, and store ``per_node_mean`` only when
the node count is fixed. The crop and velocity scaling of a mesh dataset are applied when it serves the baseline.
Each dataset gets ``baseline_stats.npz`` in its own folder.

Usage
-----
    python -m operator_shap.datagen.compute_baseline_stats
    python -m operator_shap.datagen.compute_baseline_stats data_root=/path/to/data datasets=[heat_1d,burgers_1d]
"""

from __future__ import annotations

import logging

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from operator_shap.baseline_stats import write_stats

logger = logging.getLogger(__name__)

# Meshes whose node count varies across samples: per_node_mean is undefined,
# only global_mean is stored.
VARIABLE_MESH: tuple[str, ...] = ("mgn_cylinder_flow",)


def _grid_stats(ds):
    """Grid stats averaged over the whole train split."""
    u0 = ds.u0.numpy()  # (N, C, *spatial), channel-first
    return {
        "per_node_mean": u0.mean(axis=0),
        "global_mean": u0.mean(axis=(0, *range(2, u0.ndim))),
    }


def _mesh_stats(ds):
    """Mesh stats from ``u0 = velocity[t0]`` over the train trajectories."""
    name = ds.data_dir.name
    per_node = name not in VARIABLE_MESH
    global_sum = np.zeros(2, dtype=np.float64)
    count = 0
    per_node_sum = None
    for path in tqdm(ds._files, desc=f"{name}: train u0", leave=False):
        with np.load(path) as z:
            u0 = z["velocity"][ds.t0].astype(np.float64)  # (n, 2)
        global_sum += u0.sum(axis=0)
        count += u0.shape[0]
        if per_node:
            per_node_sum = u0 if per_node_sum is None else per_node_sum + u0

    stats = {"global_mean": (global_sum / count).reshape(1, -1)}
    if per_node:
        stats["per_node_mean"] = per_node_sum / len(ds._files)
    return stats


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="compute_baseline_stats",
)
def main(cfg: DictConfig) -> None:
    for name in cfg.datasets:
        name = str(name)
        ds_cfg = hydra.compose(config_name="compute_baseline_stats", overrides=[f"+dataset={name}", f"data_root={cfg.data_root}"]).dataset
        ds = hydra.utils.instantiate(ds_cfg.class_config, split="train", max_samples=None)
        stats = _mesh_stats(ds) if ds.is_mesh else _grid_stats(ds)
        path = write_stats(ds.data_dir, stats)
        shapes = {k: tuple(v.shape) for k, v in stats.items()}
        logger.info(f"{name}: wrote {shapes} -> {path}")


if __name__ == "__main__":
    main()
