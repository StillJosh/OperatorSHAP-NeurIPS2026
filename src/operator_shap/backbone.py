"""Model construction from the config and checkpoint loading for backbones and explainers."""

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf


def build_model(cfg: DictConfig, state_dict: dict | None = None) -> torch.nn.Module:
    """Instantiate ``cfg.model.arch`` on ``cfg.device``, and load ``state_dict`` if given."""
    model = hydra.utils.instantiate(cfg.model.arch)
    if state_dict is not None:
        model.load_state_dict({k: v for k, v in state_dict.items() if k != "_metadata"}, strict=True)
    return model.to(cfg.device)


def load_model(path: str | Path, device: str) -> tuple[torch.nn.Module, DictConfig]:
    """``(model, config)`` of the checkpoint at ``path``, rebuilt on ``device`` from the config it was trained with.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist, naming the run that writes it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; train it first (train_backbone.py for a backbone, run_attribution.py for an explainer)")
    cp = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(cp["config"])
    cfg.device = device
    return build_model(cfg, state_dict=cp["model_state_dict"]), cfg


def load_frozen(path: str | Path, device: str) -> tuple[torch.nn.Module, DictConfig]:
    """:func:`load_model` in eval mode with gradients disabled for the parameters."""
    model, cfg = load_model(path, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def load_pair(cfg: DictConfig) -> tuple[torch.nn.Module, torch.nn.Module, str, tuple[float, str]]:
    """Frozen backbone and explainer of ``cfg`` with the explainer's training baseline and query encoding ``(radius, kernel)``."""
    backbone, _ = load_frozen(cfg.backbone_checkpoint, cfg.device)
    explainer, expl_cfg = load_frozen(cfg.explainer_checkpoint, cfg.device)
    query = expl_cfg.attribution.query
    return backbone, explainer, str(expl_cfg.attribution.baseline), (float(query.radius), str(query.kernel))
