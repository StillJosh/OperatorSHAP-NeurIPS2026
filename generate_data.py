"""Generate a PDE dataset: random initial conditions solved in parallel, written as train/val/test NPZ files.

Writes ``<data_root>/<datagen.name>/{train,val,test}.npz``. Sample ``i`` draws its initial condition from
``SeedSequence(seed).spawn(n_samples)[i]``, so the output is fixed by the config and does not depend on ``n_workers``.

Usage
-----
    python generate_data.py datagen=burgers_1d
    python generate_data.py -m datagen=advection_1d,heat_1d,burgers_1d data_root=/path/to/data
"""

import os
from pathlib import Path

import hydra
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from operator_shap.datagen import tqdm_joblib
from operator_shap.datagen.writer import write_splits

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _build_grid(cfg_dg: DictConfig) -> tuple[dict, tuple]:
    """Build the spatial grid from a datagen sub-config; returns (grid_dict, ic_grid_args)."""
    res = cfg_dg.grid.resolution
    if "domain_x" in cfg_dg.grid:
        axes = ["x", "y"] + (["z"] if "domain_z" in cfg_dg.grid else [])
        grid = {
            ax: np.linspace(*cfg_dg.grid[f"domain_{ax}"], res, endpoint=False)
            for ax in axes
        }
        return grid, tuple(grid.values())
    x = np.linspace(cfg_dg.grid.domain[0], cfg_dg.grid.domain[1], res, endpoint=False)
    return {"x": x}, (x,)


def _resolve_n_workers(cfg: DictConfig) -> int:
    """Resolve ``cfg.n_workers``; ``-1`` → ``os.cpu_count()``."""
    return (os.cpu_count() or 1) if cfg.n_workers == -1 else int(cfg.n_workers)


def _solve_one(
    i: int,
    child_seed: np.random.SeedSequence,
    ic_fn,
    ic_params: dict,
    solver,
    ic_grid_args: tuple,
    t_eval: np.ndarray | None,
):
    """Generate one sample: seeded IC draw + one solver call.

    Called inside worker processes. ``t_eval=None`` signals a static
    solver (Darcy); otherwise the solver is time-dependent.
    """
    rng = np.random.default_rng(child_seed)
    u0 = ic_fn(*ic_grid_args, rng, **ic_params)
    sol = solver.solve(u0) if t_eval is None else solver.solve(u0, t_eval)
    return i, u0, sol


def _run_parallel(
    cfg: DictConfig,
    ic_fn,
    ic_params: dict,
    solver,
    ic_grid_args: tuple,
    t_eval: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch ``n_samples`` independent solves across workers; return stacked arrays."""
    n = cfg.n_samples
    n_workers = _resolve_n_workers(cfg)
    child_seeds = np.random.SeedSequence(cfg.seed).spawn(n)

    with tqdm_joblib(tqdm(total=n, desc=cfg.datagen.name)):
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_solve_one)(
                i, child_seeds[i], ic_fn, ic_params, solver, ic_grid_args, t_eval,
            )
            for i in range(n)
        )

    results.sort(key=lambda r: r[0])
    u0_arr = np.stack([r[1] for r in results], axis=0)
    sol_arr = np.stack([r[2] for r in results], axis=0)
    return u0_arr, sol_arr


def _generate_time_dependent(cfg: DictConfig) -> dict[str, np.ndarray]:
    """Generate data for a time-dependent PDE in any spatial dimension."""
    cfg_dg = cfg.datagen
    grid_dict, ic_grid_args = _build_grid(cfg_dg)
    t_eval = np.linspace(0, cfg_dg.time.t_final, cfg_dg.time.n_snapshots)

    solver = hydra.utils.instantiate(cfg_dg.solver, grid=grid_dict)
    ic_fn = hydra.utils.get_method(cfg_dg.ic._target_)
    ic_params = {k: v for k, v in cfg_dg.ic.items() if k != "_target_"}

    all_u0, all_sol = _run_parallel(cfg, ic_fn, ic_params, solver, ic_grid_args, t_eval)
    return {"u0": all_u0, "solution": all_sol, **grid_dict, "t": t_eval}


def _generate_darcy(cfg: DictConfig) -> dict[str, np.ndarray]:
    """Generate data for 2D Darcy flow (static)."""
    cfg_dg = cfg.datagen
    grid_dict, ic_grid_args = _build_grid(cfg_dg)
    x, y = grid_dict["x"], grid_dict["y"]

    solver = hydra.utils.instantiate(cfg_dg.solver, grid=grid_dict)
    ic_fn = hydra.utils.get_method(cfg_dg.ic._target_)
    ic_params = {k: v for k, v in cfg_dg.ic.items() if k != "_target_"}

    all_perm, all_pressure = _run_parallel(
        cfg, ic_fn, ic_params, solver, ic_grid_args, t_eval=None,
    )
    return {"permeability": all_perm, "pressure": all_pressure, "x": x, "y": y}


# Dispatch table: PDE name -> generator function
_GENERATORS = {
    "advection_1d": _generate_time_dependent,
    "heat_1d": _generate_time_dependent,
    "burgers_1d": _generate_time_dependent,
    "allen_cahn_1d": _generate_time_dependent,
    "diffusion_sorption_1d": _generate_time_dependent,
    "heat_2d": _generate_time_dependent,
    "heat_3d": _generate_time_dependent,
    "darcy_2d": _generate_darcy,
    "reaction_diffusion_2d": _generate_time_dependent,
    "shallow_water_2d": _generate_time_dependent,
}


@hydra.main(version_base=None, config_path="configs", config_name="generate_data")
def main(cfg: DictConfig) -> None:
    name = cfg.datagen.name
    n_workers = _resolve_n_workers(cfg)
    print(
        f"Generating {name} — {cfg.n_samples} samples, seed={cfg.seed}, "
        f"n_workers={n_workers}"
    )

    generator = _GENERATORS[name]
    data = generator(cfg)

    write_splits(
        data,
        output_dir=Path(cfg.data_root),
        pde_name=name,
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )
    print("Done.")


if __name__ == "__main__":
    main()
