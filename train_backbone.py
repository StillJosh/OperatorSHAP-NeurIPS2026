"""Train a backbone ``u0 -> u(., t*)``; writes ``best_model.pt`` to ``outputs/{dataset}/{family}/backbone``.

Usage
-----
    python train_backbone.py dataset=burgers_1d
    python train_backbone.py dataset=mgn_cylinder_flow training=backbone_gino
    python train_backbone.py dataset=burgers_1d mode=debug           # cpu, tiny data and model
"""

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from operator_shap.backbone import build_model
from operator_shap.dataset import build_data_loaders
from operator_shap.trainer import Trainer
from operator_shap.utils import finish_run, setup_run

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train_backbone")
def main(cfg: DictConfig) -> None:
    setup_run(cfg)
    train_loader, val_loader = build_data_loaders(cfg)
    model = build_model(cfg)
    logger.info(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} params={sum(p.numel() for p in model.parameters())}")

    optimizer = instantiate(cfg.optimizer)(model.parameters())
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=instantiate(cfg.train_loss),
        val_criterion=instantiate(cfg.val_loss),
        optimizer=optimizer,
        scheduler=instantiate(cfg.scheduler)(optimizer),
        config=cfg,
    )
    trainer.train()
    logger.info(f"Done. best_val={trainer.best_val_loss:.5f}")
    finish_run(cfg, best_val_loss=trainer.best_val_loss)


if __name__ == "__main__":
    main()
