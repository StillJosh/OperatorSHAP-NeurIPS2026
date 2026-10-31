"""Structural stability across resolutions.

SSIM between attribution maps of the same test sample served at two resolutions, centre query, on Heat 2D and Shallow Water 2D
(the 2D PDEs with references at every n in {64, 256, 1024}). The discrete values at every partition are integrals of one density,
so the comparison lives at the coarse partition: discrete phi is multiplied by n to a density, the finer map is average-pooled to
the coarser grid, both maps are nearest-upsampled to a common 32x32 canvas (identical blockiness on both sides, so the 7x7 Gaussian
SSIM window is defined on the 8x8 partition too), standardized to zero mean and unit variance, and compared with SSIM (data range
= joint min-max of the pair). Median and IQR over the 30 test samples.

Methods: OperatorSHAP (explainer, hard-normalised, at each resolution), the reference (RegressionMSR at 3e5, ``gt``), and
RegressionMSR at the cheap budget max(2n, 1024) (``ps_b1024`` at 64 and 256, ``ps_b2048`` at 1024).
Writes ``results/tables/ssim.tex``.
"""

# %%
import pandas as pd
import torch
import torch.nn.functional as F

from operator_shap.backbone import load_pair
from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains, explain
from operator_shap.results import RESULTS, compose_config, load_phi

DEVICE = "cpu"
FAM = "localFNO"
TABLE = RESULTS / "tables" / "ssim.tex"

PDES = [("Heat 2D", "heat_2d", "heat2d"), ("Shallow Water 2D", "shallow_water_2d", "sw2d")]
N_PLAYERS = (64, 256, 1024)
PAIRS = ((64, 256), (256, 1024), (64, 1024))
METHODS = ("OperatorSHAP", "Reference", "RegressionMSR")
TARGET_R = 32                            # common comparison grid (finest per-dim)


# %%
def _gaussian_window(size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = torch.outer(g, g)
    return (k / k.sum())[None, None]


_WIN = _gaussian_window()


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean SSIM of two (H, W) maps; valid convolution, joint data range."""
    lum = float(torch.max(a.max(), b.max()) - torch.min(a.min(), b.min()))
    c1, c2 = (0.01 * lum) ** 2, (0.03 * lum) ** 2
    x, y = a[None, None], b[None, None]
    mu_x, mu_y = F.conv2d(x, _WIN), F.conv2d(y, _WIN)
    sxx = F.conv2d(x * x, _WIN) - mu_x ** 2
    syy = F.conv2d(y * y, _WIN) - mu_y ** 2
    sxy = F.conv2d(x * y, _WIN) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float((num / den).mean())


def _standardize(m: torch.Tensor) -> torch.Tensor:
    return (m - m.mean()) / m.std().clamp_min(1e-12)


def _on_canvas(density: torch.Tensor, r_coarse: int) -> torch.Tensor:
    """(B, r, r) density -> pooled to ``r_coarse`` -> standardized (B, 32, 32).

    Average pooling realises the density integral over the coarse cells;
    nearest upsampling gives both pair members identical blockiness.
    """
    r = density.shape[-1]
    d = F.avg_pool2d(density[:, None], r // r_coarse) if r > r_coarse else density[:, None]
    up = F.interpolate(d, size=(TARGET_R, TARGET_R), mode="nearest")[:, 0]
    return torch.stack([_standardize(m) for m in up])


# %%
rows = []
for label, dataset, ds in PDES:
    cfgs = [compose_config("run_metrics", [f"dataset={dataset}", f"attribution.n_players={n}", f"device={DEVICE}"]) for n in N_PLAYERS]
    backbone, explainer, baseline_mode, query_encoding = load_pair(cfgs[0])
    dens: dict[tuple[str, int], torch.Tensor] = {}
    for n, cfg in zip(N_PLAYERS, cfgs):
        _, test_loader, n_eval = build_test_loader(cfg, baseline_mode)
        domain_of_batch = build_domains(cfg, test_loader, query_encoding=query_encoding)
        phi = {
            "OperatorSHAP": torch.stack([explain(explainer, backbone, domain_of_batch(b, eval_mode=True))[0, 0].cpu() for b in test_loader]),
            "Reference": load_phi(ds, FAM, n, "gt")[:n_eval, 0],
            "RegressionMSR": load_phi(ds, FAM, n, f"ps_b{cfg.metrics.cheap_budget}")[:n_eval, 0],
        }
        r = int(round(n ** 0.5))
        for method, p in phi.items():
            dens[(method, n)] = p.reshape(-1, r, r) * n
    for method in METHODS:
        for n_a, n_b in PAIRS:
            r_a = int(round(n_a ** 0.5))
            coarse, pooled = _on_canvas(dens[(method, n_a)], r_a), _on_canvas(dens[(method, n_b)], r_a)
            vals = torch.tensor([ssim(coarse[s], pooled[s]) for s in range(len(coarse))])
            rows.append({"pde": label, "method": method, "pair": f"{n_a}--{n_b}", "ssim_med": float(vals.median()),
                         "ssim_q25": float(vals.quantile(0.25)), "ssim_q75": float(vals.quantile(0.75)), "n_samples": len(vals)})

scores = pd.DataFrame(rows)
print(scores.pivot_table(index=["pde", "pair"], columns="method", values="ssim_med", sort=False)[list(METHODS)].round(3))

# %%
lines = [
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"PDE & players & OperatorSHAP & Reference ($3\!\cdot\!10^5$) & RegressionMSR ($\max(2n, 1024)$) \\",
    r"\midrule",
]
for k, (label, _, _) in enumerate(PDES):
    if k:
        lines.append(r"\midrule")
    for j, (n_a, n_b) in enumerate(PAIRS):
        g = scores[(scores.pde == label) & (scores.pair == f"{n_a}--{n_b}")].set_index("method").loc[list(METHODS)]
        head = rf"\multirow{{{2 * len(PAIRS)}}}{{*}}{{{label}}}" if j == 0 else ""
        lines.append(f"{head} & {n_a}--{n_b} & " + " & ".join(f"{v:.3f}" for v in g.ssim_med) + r" \\")
        lines.append(" &  & " + " & ".join(rf"\tiny{{[{a:.3f}, {b:.3f}]}}" for a, b in zip(g.ssim_q25, g.ssim_q75)) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
TABLE.parent.mkdir(parents=True, exist_ok=True)
TABLE.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
