"""Base class of the attribution methods dispatched by ``run_attribution.py``."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class AttributionMethod(ABC):
    """One attribution method; ``run()`` trains or attributes and saves to ``out_dir``.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config; the method's own section is ``cfg.attribution``.
    backbone : nn.Module
        Frozen neural operator providing the value function.
    train_loader, val_loader : DataLoader
    out_dir : Path
        Run directory created by :func:`operator_shap.utils.setup_run`.
    """

    def __init__(self, cfg: DictConfig, backbone: nn.Module, train_loader: DataLoader, val_loader: DataLoader, out_dir: Path) -> None:
        self.cfg = cfg
        self.backbone = backbone
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out_dir = out_dir

    @abstractmethod
    def run(self) -> None:
        """Train or attribute and save the outputs."""
