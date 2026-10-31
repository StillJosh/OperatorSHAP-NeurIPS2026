"""Loaders for every entry point, built from ``cfg.dataset.class_config`` by Hydra instantiation."""

from typing import Any

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from operator_shap.domain import n_per_dim_from_players

from .samplers import MultiResSubset, ResolutionBatchSampler


def _x_star_kwargs(cfg: DictConfig) -> dict:
    """Dataset ``x_star`` kwarg from ``attribution.query.points``; backbone training has no query block."""
    if "attribution" not in cfg or "query" not in cfg.attribution:
        return {}
    return {"x_star": cfg.attribution.query.points}


def build_test_loader(cfg: DictConfig, baseline: str, split: str = "test") -> tuple[Any, DataLoader, int]:
    """``split`` and its batch-of-1 loader at the identity-partition resolution of ``attribution.n_players``.

    Returns
    -------
    test_ds
        The test dataset serving ``baseline`` and the query points.
    test_loader : DataLoader
    n_eval : int
        ``min(attribution.n_samples, len(test_ds))``.
    """
    test_ds = hydra.utils.instantiate(cfg.dataset.class_config, split=split, max_samples=None, baseline=baseline, **_x_star_kwargs(cfg))
    n_eval = min(int(cfg.attribution.n_samples), len(test_ds))
    eval_res = test_ds.spatial_res[0] if test_ds.is_mesh else n_per_dim_from_players(int(cfg.attribution.n_players), test_ds.n_spatial_dims)
    return test_ds, build_eval_loader(test_ds, n_eval, eval_res), n_eval


def build_eval_loader(ds, n_eval: int, resolution: int) -> DataLoader:
    """Batch-of-1 loader over the first ``n_eval`` samples of ``ds`` at ``resolution``, in order."""
    subset = MultiResSubset(ds, list(range(n_eval)))
    sampler = ResolutionBatchSampler(n_eval, [resolution], batch_size=1, shuffle=False, drop_last=False)
    return DataLoader(subset, batch_sampler=sampler, num_workers=0)


def build_data_loaders(cfg: DictConfig, baseline: str | None = None) -> tuple[DataLoader, DataLoader]:
    """``(train_loader, val_loader)`` over ``cfg.dataset.resolutions`` from the train and val splits on disk.

    Every resolution must divide the native one. ``baseline`` is forwarded
    to the datasets so attribution batches carry the inactive-node field;
    backbone training passes ``None``.
    """
    x_star_kwargs = _x_star_kwargs(cfg)
    train_full = hydra.utils.instantiate(cfg.dataset.class_config, split="train", baseline=baseline, **x_star_kwargs)
    val_full = hydra.utils.instantiate(cfg.dataset.class_config, split="val", baseline=baseline, **x_star_kwargs)
    full_res = train_full.spatial_res[0]
    train_subset = MultiResSubset(train_full, list(range(len(train_full))))
    val_subset = MultiResSubset(val_full, list(range(len(val_full))))

    resolutions = list(cfg.dataset.resolutions)
    for r in resolutions:
        if full_res % r != 0:
            raise ValueError(f"full_res={full_res} not divisible by resolution {r}")

    n_workers = cfg.training.num_workers
    pin = cfg.device != "cpu"
    train_sampler = ResolutionBatchSampler(len(train_subset), resolutions, cfg.training.batch_size, shuffle=True)
    val_sampler = ResolutionBatchSampler(len(val_subset), resolutions, cfg.training.batch_size, shuffle=False, drop_last=False)
    train_loader = DataLoader(train_subset, batch_sampler=train_sampler, pin_memory=pin, num_workers=n_workers, persistent_workers=(n_workers > 0))
    val_loader = DataLoader(val_subset, batch_sampler=val_sampler, pin_memory=pin, num_workers=n_workers, persistent_workers=(n_workers > 0))
    return train_loader, val_loader
