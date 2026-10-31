"""Attribution metrics of one (dataset, player count): Pearson, NRMSE, and the faithfulness R².

Scores the classical Shapley values of ``metrics.references`` (``shapley_values.npz`` of the ``shap`` runs),
the explainer ``explainer_checkpoint`` at the query points, and Integrated Gradients against ``Reference``
and the true game. Writes one ``metrics.csv`` row per (method, sample, query) for ``make_tables.py``.

Usage
-----
    python run_metrics.py dataset=burgers_1d attribution.n_players=64
    python run_metrics.py dataset=mgn_cylinder_flow attribution=eval_mgn attribution.n_players=64
"""

import logging

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from operator_shap.backbone import load_pair
from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains, explain
from operator_shap.methods.integrated_gradient import ig_density
from operator_shap.metrics import kernel_r2, nrmse, pearson_r
from operator_shap.utils import finish_run, setup_run

logger = logging.getLogger(__name__)


def _load_saved_phi(path: str, x_star: np.ndarray, baseline: str) -> np.ndarray:
    """``shapley_values`` ``(n_samples, n_queries, n_players)`` of a classical run, checked against the game's query points and baseline."""
    with np.load(path) as data:
        phi = np.asarray(data["shapley_values"], dtype=np.float32)
        saved_x_star, saved_baseline = np.asarray(data["x_star"]).reshape(-1), str(data["baseline"])
    if not np.array_equal(saved_x_star, x_star.reshape(-1)):
        raise ValueError(f"{path}: query points {saved_x_star.tolist()} != expected {x_star.reshape(-1).tolist()}")
    if saved_baseline != baseline:
        raise ValueError(f"{path}: baseline {saved_baseline!r} != explainer baseline {baseline!r}")
    return phi


@hydra.main(version_base=None, config_path="configs", config_name="run_metrics")
def main(cfg: DictConfig) -> None:
    out_dir = setup_run(cfg)
    backbone, explainer, baseline_mode, query_encoding = load_pair(cfg)
    logger.info(f"explainer={cfg.explainer_checkpoint} baseline={baseline_mode} query_encoding={query_encoding}")

    test_ds, test_loader, n_eval = build_test_loader(cfg, baseline_mode)
    domain_of_batch = build_domains(cfg, test_loader, query_encoding=query_encoding)
    x_star = np.asarray(test_ds.x_star, dtype=np.float32)
    n_queries, n_players = len(x_star), int(cfg.attribution.n_players)
    saved = {label: _load_saved_phi(path, x_star, baseline_mode) for label, path in cfg.metrics.references.items()}
    for label, phi in saved.items():
        if phi.shape[0] < n_eval or phi.shape[1:] != (n_queries, n_players):
            raise ValueError(f"{label}: shape {phi.shape} does not cover {(n_eval, n_queries, n_players)}")
    labels = [*saved, "OperatorSHAP", "IG"]

    rows = []
    for si, batch in enumerate(test_loader):
        dom = domain_of_batch(batch, eval_mode=True)
        phis = [torch.from_numpy(saved[label][si]) for label in saved]
        phis.append(explain(explainer, backbone, dom)[0].cpu())
        ig = ig_density(backbone, dom, n_steps=int(cfg.metrics.ig_steps), batch_size=int(cfg.metrics.ig_batch_size))[0]
        phis.append(dom.discretize(ig).cpu())                                      # per-node density summed per player
        phi = torch.stack(phis)                                                   # (n_methods, n_queries, n_players)
        r2 = kernel_r2(
            backbone, dom, phi.unsqueeze(1), int(cfg.metrics.n_coalitions), int(cfg.seed),
            batch_size=int(cfg.attribution.max_coalition_batch),
        )[:, 0]
        ref = phi[labels.index("Reference")]
        pear, err = pearson_r(phi, ref), nrmse(phi, ref)                          # (n_methods, n_queries)
        for m, label in enumerate(labels):
            for q in range(n_queries):
                rows.append({
                    "pde": cfg.dataset.dataset_short, "family": cfg.model.family, "n_players": n_players,
                    "query": q, "x_star": str(x_star[q].tolist()), "method": label, "sample": si,
                    "pearson": float(pear[m, q]), "nrmse": float(err[m, q]), "r2": float(r2[m, q]),
                })
        logger.info(f"  sample {si + 1}/{n_eval}  " + "  ".join(f"{l}: r2={r2[m, 0]:.3f}" for m, l in enumerate(labels)))

    df = pd.DataFrame(rows)
    out_path = out_dir / "metrics.csv"
    df.to_csv(out_path, index=False)
    summary = df[df["query"] == 0].groupby("method")[["pearson", "nrmse", "r2"]].median()
    logger.info(f"wrote {out_path}\n{summary}")
    finish_run(cfg)


if __name__ == "__main__":
    main()
