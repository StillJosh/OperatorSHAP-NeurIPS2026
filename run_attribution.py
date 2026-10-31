"""Run one attribution method (``cfg.attribution._target_``) against the trained backbone of the dataset.

``attribution=fastshap`` trains the explainer (``outputs/{dataset}/{family}/explainer``), ``attribution=shap``
computes classical Shapley values with shapiq (``outputs/{dataset}/{family}/shap_n{n}_{segment}``).

Usage
-----
    python run_attribution.py dataset=burgers_1d
    python run_attribution.py dataset=burgers_1d +experiment=budget_runs attribution.n_players=64 attribution.budget=300000
    python run_attribution.py dataset=mgn_airfoil +experiment=mgn
    python run_attribution.py dataset=burgers_1d mode=debug           # cpu, tiny data and model
"""

import hydra
from omegaconf import DictConfig

from operator_shap import utils
from operator_shap.backbone import load_frozen
from operator_shap.dataset import build_data_loaders


@hydra.main(version_base=None, config_path="configs", config_name="run_attribution")
def main(cfg: DictConfig) -> None:
    out_dir = utils.setup_run(cfg)
    backbone, _ = load_frozen(cfg.backbone_checkpoint, cfg.device)
    train_loader, val_loader = build_data_loaders(cfg, baseline=str(cfg.attribution.baseline))
    method = hydra.utils.get_class(cfg.attribution._target_)(cfg=cfg, backbone=backbone, train_loader=train_loader, val_loader=val_loader, out_dir=out_dir)
    utils.backbone_forward_counter.reset()
    method.run()
    utils.finish_run(cfg, **utils.backbone_forward_counter.snapshot())


if __name__ == "__main__":
    main()
