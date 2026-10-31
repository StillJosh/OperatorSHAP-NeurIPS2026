"""Read/write precomputed train-split baseline statistics.

:mod:`operator_shap.datagen.compute_baseline_stats` writes ``baseline_stats.npz`` next
to a dataset's data files, holding the ``per_node_mean`` and
``global_mean`` train statistics; attribution runs read the baseline
field from it instead of recomputing it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

STATS_FILENAME = "baseline_stats.npz"


def stats_path(data_dir: str | Path) -> Path:
    """Path of the baseline-stats file inside a dataset's data directory."""
    return Path(data_dir) / STATS_FILENAME


def write_stats(data_dir: str | Path, stats: dict[str, np.ndarray]) -> Path:
    """Write ``{kind: array}`` train statistics to ``data_dir``."""
    path = stats_path(data_dir)
    np.savez(path, **{k: np.asarray(v, dtype=np.float32) for k, v in stats.items()})
    return path


def load_stat(data_dir: str | Path, kind: str) -> torch.Tensor:
    """Read one statistic array as a float32 tensor."""
    path = stats_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; create it with `python -m operator_shap.datagen.compute_baseline_stats`")
    with np.load(path) as z:
        return torch.from_numpy(np.asarray(z[kind], dtype=np.float32))
