"""Classical Shapley values of the backbone's coalition game via shapiq, one game per (test sample, query point)."""

import logging

import numpy as np

from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains
from operator_shap.games import CoalitionGame, build_approximator, extract_shapley_values, run_approximator
from operator_shap.methods.base import AttributionMethod

logger = logging.getLogger(__name__)


class ClassicalSHAP(AttributionMethod):
    """shapiq ``proxyshap`` / ``kernelshap`` / ``exact`` at a fixed ``attribution.n_players`` partition.

    Writes ``shapley_values.npz`` with ``(n_samples, n_queries, n_players)`` values.
    """

    def run(self) -> None:
        attr_cfg = self.cfg.attribution
        budget, approx_kind, n_players = int(attr_cfg.budget), str(attr_cfg.approximator), int(attr_cfg.n_players)
        test_ds, test_loader, n_eval = build_test_loader(self.cfg, str(attr_cfg.baseline))
        domain_of_batch = build_domains(self.cfg, test_loader, query_encoding=None)
        approximator = build_approximator(approx_kind, n_players, seed=self.cfg.seed)
        logger.info(f"approximator={approx_kind}  budget={budget}  n_eval={n_eval}")

        shapley_all = None
        for si, batch in enumerate(test_loader):
            domain = domain_of_batch(batch, eval_mode=True)
            if shapley_all is None:
                shapley_all = np.zeros((n_eval, domain.n_queries, n_players), dtype=np.float32)
            for q in range(domain.n_queries):
                game = CoalitionGame(self.backbone, domain, max_batch=int(attr_cfg.max_coalition_batch), query=q)
                iv = run_approximator(approx_kind, approximator, game, n_players=n_players, budget=budget)
                shapley_all[si, q] = extract_shapley_values(iv, n_players)
            logger.info(f"  sample {si + 1}/{n_eval}  peak_player={int(np.argmax(np.abs(shapley_all[si, 0])))}")

        out_path = self.out_dir / "shapley_values.npz"
        np.savez(
            out_path, shapley_values=shapley_all, n_players=n_players, budget=budget,
            baseline=test_ds.baseline_mode, x_star=np.asarray(test_ds.x_star, dtype=np.float32),
        )
        logger.info(f"Saved → {out_path}")
