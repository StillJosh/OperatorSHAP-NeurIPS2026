"""FastSHAP: an explainer trained online to predict Shapley values in one forward pass.

The FastSHAP objective (Jethani et al., ICLR 2022) with hard normalisation
on the coalition game of a :class:`~operator_shap.domain.Domain`; grids
at several resolutions and meshes share the trainer.
"""

import logging

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from operator_shap.backbone import build_model
from operator_shap.domain import build_domains
from operator_shap.methods.base import AttributionMethod
from operator_shap.trainer import Trainer
from operator_shap.utils import hard_normalize, predict_coalition_values, sample_coalitions

logger = logging.getLogger(__name__)


class FastSHAPTrainer(Trainer):
    """Trainer whose batch loss is the FastSHAP objective on the batch's domain.

    Parameters
    ----------
    backbone : nn.Module
        Frozen neural operator.
    fastshap_cfg : DictConfig
        ``attribution`` config with ``n_coalitions``, ``sampling``, ``max_coalition_batch``.
    domain_of_batch : callable
        ``domain_of_batch(data) -> Domain`` for one loader batch.
    """

    def __init__(self, *, backbone: nn.Module, fastshap_cfg: DictConfig, domain_of_batch, **trainer_kwargs) -> None:
        super().__init__(**trainer_kwargs)
        self.backbone = backbone
        self.domain_of_batch = domain_of_batch
        self.n_coalitions = int(fastshap_cfg.n_coalitions)
        self.sampling = str(fastshap_cfg.sampling)
        self.max_coalition_batch = int(fastshap_cfg.max_coalition_batch)

    def evaluate(self, epoch: int) -> dict[str, float]:
        """Validation loss on the same coalitions and query points every epoch.

        The loss draws coalitions and query nodes at random; resetting a forked
        generator to ``config.seed`` makes epochs comparable, so early stopping
        and best-checkpoint selection react to the model and not to the draw.
        """
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(self.config.seed))
            return super().evaluate(epoch)

    def _run_batch(self, data: dict[str, torch.Tensor], crit: str = "train") -> torch.Tensor:
        dom = self.domain_of_batch(data)
        masks = sample_coalitions(dom.n_samples, self.n_coalitions, dom.n_players, self.device, self.sampling)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
            v_S = dom.coalition_values(self.backbone, masks, self.max_coalition_batch)
            v_grand, v_empty = dom.game_values(self.backbone)
        phi = hard_normalize(dom.coarsen(self.model(dom.explainer_input())), v_grand, v_empty, weights=dom.cell_measures)
        crit_fn = self.criterion if crit == "train" else self.val_criterion
        return crit_fn(predict_coalition_values(masks, phi, v_empty), v_S)


class FastSHAP(AttributionMethod):
    """Train the explainer with :class:`FastSHAPTrainer`; the best validation checkpoint is ``best_model.pt``."""

    def run(self) -> None:
        cfg = self.cfg
        domain_of_batch = build_domains(cfg, self.train_loader)
        explainer = build_model(cfg)
        optimizer = instantiate(cfg.optimizer)(explainer.parameters())
        trainer = FastSHAPTrainer(
            backbone=self.backbone,
            fastshap_cfg=cfg.attribution,
            domain_of_batch=domain_of_batch,
            model=explainer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            criterion=instantiate(cfg.train_loss),
            val_criterion=instantiate(cfg.val_loss),
            optimizer=optimizer,
            scheduler=instantiate(cfg.scheduler)(optimizer),
            config=cfg,
        )
        trainer.train()
        logger.info(f"Done. best_val={trainer.best_val_loss:.5f}")
