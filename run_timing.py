"""Wall-clock seconds per explained sample of OperatorSHAP, Integrated Gradients, RMSR, and KernelSHAP.

Explains ``attribution.n_samples`` validation inputs one at a time at the dataset's query points with the
trained explainer and backbone on ``timing.n_threads`` torch threads, and writes the seconds per (method, sample)
to ``timing.csv`` together with the hardware, the IG steps ``metrics.ig_steps``, and the classical budget
``metrics.cheap_budget``. Each method runs ``timing.n_warmup`` untimed passes first.

Usage
-----
    python run_timing.py dataset=burgers_1d attribution.n_players=256 device=cpu
    python run_timing.py -m +experiment=ig_budget_multiplier device=cpu attribution.n_samples=10 timing.n_warmup=1
"""

import logging
import platform
import time
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig

from operator_shap.backbone import load_pair
from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains, explain
from operator_shap.games import CoalitionGame, build_approximator, extract_shapley_values, run_approximator
from operator_shap.methods.integrated_gradient import ig_density
from operator_shap.utils import finish_run, setup_run

logger = logging.getLogger(__name__)


def _hardware(device: str) -> str:
    """GPU name on CUDA, otherwise the CPU model."""
    if device.startswith("cuda"):
        return torch.cuda.get_device_name()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        return next(line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines() if line.startswith("model name"))
    return platform.processor()


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@hydra.main(version_base=None, config_path="configs", config_name="run_timing")
def main(cfg: DictConfig) -> None:
    out_dir = setup_run(cfg)
    device = str(cfg.device)
    torch.set_num_threads(int(cfg.timing.n_threads))
    backbone, explainer, baseline_mode, query_encoding = load_pair(cfg)
    _, loader, n_eval = build_test_loader(cfg, baseline_mode, split="val")
    domain_of_batch = build_domains(cfg, loader, query_encoding=query_encoding)

    n_players, budget = int(cfg.attribution.n_players), int(cfg.metrics.cheap_budget)

    def classical(kind: str, dom) -> list:
        """Shapley values of every query point of ``dom`` with the shapiq approximator ``kind`` at ``budget`` coalitions."""
        approximator = build_approximator(kind, n_players, seed=int(cfg.seed))
        games = (CoalitionGame(backbone, dom, max_batch=int(cfg.attribution.max_coalition_batch), query=q) for q in range(dom.n_queries))
        return [extract_shapley_values(run_approximator(kind, approximator, g, n_players=n_players, budget=budget), n_players) for g in games]

    methods = {
        "OperatorSHAP": lambda dom: explain(explainer, backbone, dom),
        "IG": lambda dom: dom.discretize(ig_density(backbone, dom, n_steps=int(cfg.metrics.ig_steps), batch_size=int(cfg.metrics.ig_batch_size))),
        "RMSR": lambda dom: classical("proxyshap", dom),
        "KernelSHAP": lambda dom: classical("kernelshap", dom),
    }
    hardware, n_threads = _hardware(device), torch.get_num_threads()
    logger.info(f"device={device} hardware={hardware} torch_threads={n_threads}")
    rows = []
    for si, batch in enumerate(loader):
        for label, method in methods.items():
            if si == 0:
                for _ in range(int(cfg.timing.n_warmup)):
                    method(domain_of_batch(batch, eval_mode=True))
            dom = domain_of_batch(batch, eval_mode=True)
            _synchronize(device)
            t0 = time.perf_counter()
            method(dom)
            _synchronize(device)
            rows.append({
                "pde": cfg.dataset.dataset_short, "family": cfg.model.family, "n_players": n_players, "n_queries": dom.n_queries,
                "device": device, "hardware": hardware, "threads": n_threads, "ig_steps": int(cfg.metrics.ig_steps), "budget": budget,
                "method": label, "sample": si, "seconds": time.perf_counter() - t0,
            })
        logger.info(f"  sample {si + 1}/{n_eval}  " + "  ".join(f"{r['method']}: {r['seconds']:.4f}s" for r in rows[-len(methods):]))

    df = pd.DataFrame(rows)
    out_path = out_dir / "timing.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"wrote {out_path}\n{df.groupby('method')['seconds'].median()}")
    finish_run(cfg)


if __name__ == "__main__":
    main()
