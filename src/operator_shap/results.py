"""Run outputs under ``outputs/{dataset_short}/{family}/{run_name}`` for the tables and figure scripts.

Run names: ``backbone``, ``explainer``, ``shap_n{n}_{segment}`` (segment ``gt``, ``exact``, ``ps_b{B}``, ``ks_b{B}``),
``metrics_n{n}``, ``metrics_n{n}_m{m}``, ``timing_n{n}``, and ``timing_n{n}_m{m}``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from operator_shap import utils  # noqa: F401  registers the config resolvers

REPO = Path(__file__).resolve().parents[2]
OUTPUTS = REPO / "outputs"
RESULTS = REPO / "results"


def run_dir(ds: str, fam: str, name: str) -> Path:
    """Directory of the run ``name`` of dataset ``ds`` (``dataset_short``) and model family ``fam``."""
    return OUTPUTS / ds / fam / name


def load_phi(ds: str, fam: str, n: int, segment: str) -> torch.Tensor:
    """Classical Shapley values ``(n_samples, n_queries, n_players)`` of the run ``shap_n{n}_{segment}``."""
    with np.load(run_dir(ds, fam, f"shap_n{n}_{segment}") / "shapley_values.npz") as data:
        return torch.from_numpy(data["shapley_values"].astype(np.float32))


def load_metrics(ds: str, fam: str, n: int, suffix: str = "") -> pd.DataFrame:
    """``metrics.csv`` of the run ``metrics_n{n}{suffix}``, one row per (method, sample, query)."""
    return pd.read_csv(run_dir(ds, fam, f"metrics_n{n}{suffix}") / "metrics.csv")


def load_timing(ds: str, fam: str, n: int, suffix: str = "") -> pd.DataFrame:
    """``timing.csv`` of the run ``timing_n{n}{suffix}``."""
    return pd.read_csv(run_dir(ds, fam, f"timing_n{n}{suffix}") / "timing.csv")


def load_summary(ds: str, fam: str, name: str) -> dict:
    """``summary.json`` of a run: wall-clock seconds, device, and run-specific counts such as ``backbone_evals``."""
    return json.loads((run_dir(ds, fam, name) / "summary.json").read_text())


def compose_config(config_name: str, overrides: list[str]) -> DictConfig:
    """Config of the entry point ``config_name`` with ``overrides``, paths resolved against the repository."""
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        return compose(config_name, overrides=[f"output_root={OUTPUTS}", f"data_root={REPO / 'data'}", *overrides])


def write_numbers(path: Path, values: dict[str, float], digits: int = 3) -> None:
    """Write ``values`` as LaTeX definitions read with ``\\res{key}``.

    Parameters
    ----------
    path : Path
        Output ``.tex`` file.
    values : dict[str, float]
        Number per key, e.g. ``"darcy2dmean/localFNO/1024/OperatorSHAP/pearson"``.
    digits : int
        Decimals, the precision of the tables.
    """
    lines = [f"\\expandafter\\def\\csname res@{k}\\endcsname{{{v:.{digits}f}}}" for k, v in sorted(values.items())]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")
