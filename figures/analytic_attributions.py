"""Analytic attributions of the linear 1D PDEs.

For a linear solution operator F(u_0)(x*) = int K_{t*}(x*, y) u_0(y) dy the game is linear in the mask fractions, so the
Aumann-Shapley density is phi(y) = K_{t*}(x*, y) (u_0(y) - ubar(y)) with the zero baseline ubar = 0.

- Heat 1D: K is the periodic heat kernel with symbol exp(-kappa k^2 t*). The FNO- and MWT-based explainers are scored by Pearson
  against the closed form per test sample (median and IQR over the test set).
- Advection 1D: K is a shift, the density is a point mass at the characteristic foot point x* - beta t* (periodic). Reported: where
  the explainer's |phi| peaks relative to the foot point, and the spike at x*: |f(x*)| over the median |f| of the 24 neighbouring
  nodes, for the explainer f = phi and for the backbone's leave-one-out values f_i = v(N) - v(N minus {i}) (the exact Shapley values
  of a linear game).

Explainers and backbones are the ``heat1d`` / ``adv1d`` runs of the families ``fno`` and ``mwtGupta``, evaluated at the centre query
(query 0) with n=1024 players (the native grid, one player per node). Writes ``results/figures/analytic_attributions.pdf`` and the
median Pearson of the heat 1D explainers to ``results/tables/numbers_analytic.tex``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

from operator_shap.backbone import load_pair
from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains, explain
from operator_shap.figures import save_fig
from operator_shap.figures.style import MCOLORS
from operator_shap.metrics import pearson_r
from operator_shap.results import REPO, RESULTS, compose_config, write_numbers

# %%
DATASETS = {"heat1d": "heat_1d", "adv1d": "advection_1d"}
FAMILIES = {"fno": ("FNO", "fno1d"), "mwtGupta": ("MWT", "mwt_gupta_1d")}
N_PLAYERS = 1024
HEAT_SAMPLES, ADV_SAMPLES = (0, 1, 2), (0, 5, 7)
DEVICE = "cpu"

KAPPA = float(OmegaConf.load(REPO / "configs/datagen/heat_1d.yaml").solver.params.kappa)
BETA = float(OmegaConf.load(REPO / "configs/datagen/advection_1d.yaml").solver.params.beta)


# %%
def explain_test_set(ds: str, fam: str) -> dict:
    """Explainer of ``ds`` and model family ``fam`` on the test set at the centre query, with the games it explains.

    Returns
    -------
    dict
        ``phi``, ``u0``, ``baseline`` ``(n_eval, n)`` hard-normalised attributions and game fields, cell-centre ``x`` ``(n,)``, query node
        ``node``, target time ``t_star``, the ``backbone``, and the per-sample ``domains``.
    """
    overrides = [f"dataset={DATASETS[ds]}", f"model={FAMILIES[fam][1]}", f"attribution.n_players={N_PLAYERS}", f"device={DEVICE}"]
    cfg = compose_config("run_metrics", overrides)
    backbone, explainer, baseline_mode, query_encoding = load_pair(cfg)
    test_ds, test_loader, _ = build_test_loader(cfg, baseline_mode)
    domain_of_batch = build_domains(cfg, test_loader, query_encoding=query_encoding)
    domains = [domain_of_batch(batch, eval_mode=True) for batch in test_loader]
    with np.load(test_ds.data_dir / "test.npz") as d:
        t_star = float(d["t"][cfg.dataset.class_config.t_idx])
    return {
        "phi": torch.cat([explain(explainer, backbone, dom)[:, 0].cpu() for dom in domains]),
        "u0": torch.cat([dom.u0[..., 0].cpu() for dom in domains]),
        "baseline": torch.cat([dom.baseline[..., 0].cpu() for dom in domains]),
        "x": domains[0].coords[:, 0].cpu().numpy(),
        "node": int(domains[0].x_star[0, 0]),
        "t_star": t_star,
        "backbone": backbone,
        "domains": domains,
    }


def quartiles(t: torch.Tensor) -> str:
    """``median [q25, q75]`` of a 1D tensor."""
    q = torch.quantile(t.float(), torch.tensor([0.25, 0.5, 0.75]))
    return f"{q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}]"


runs = {(ds, fam): explain_test_set(ds, fam) for ds in DATASETS for fam in FAMILIES}

# %%
# Heat 1D: closed-form density from the periodic heat kernel row at the query node.
heat = runs["heat1d", "fno"]
n, node, t_heat = N_PLAYERS, heat["node"], heat["t_star"]
k = 2 * np.pi * np.fft.fftfreq(n, d=1 / n)
kernel_row = torch.from_numpy(np.roll(np.real(np.fft.ifft(np.exp(-KAPPA * k**2 * t_heat))), node).astype(np.float32))
analytic = kernel_row * (heat["u0"] - heat["baseline"])

pearson = {fam: pearson_r(runs["heat1d", fam]["phi"], analytic) for fam in FAMILIES}
for fam, (label, _) in FAMILIES.items():
    print(f"Heat 1D {label}: Pearson vs analytic, median [IQR] over {len(analytic)} samples = {quartiles(pearson[fam])}")
write_numbers(RESULTS / "tables/numbers_analytic.tex", {f"analytic/heat1d/{fam}/pearson": float(pearson[fam].median()) for fam in FAMILIES})

# %%
# Advection 1D: peak of |phi| relative to the foot point, and the spike at x* in the explainer and in the backbone game.
adv = runs["adv1d", "fno"]
x, node_a, t_adv = adv["x"], adv["node"], adv["t_star"]
foot = (x[node_a] - BETA * t_adv) % 1.0
neighbours = [*range(node_a - 12, node_a), *range(node_a + 1, node_a + 13)]


def spike(f: torch.Tensor) -> torch.Tensor:
    """``|f(x*)|`` over the median ``|f|`` of the neighbouring nodes, per sample."""
    return f[:, node_a].abs() / f[:, neighbours].abs().median(-1).values


def leave_one_out(run: dict, nodes: list[int]) -> torch.Tensor:
    """Backbone ``v(N) - v(N minus {i})`` per sample and node, ``(n_eval, len(nodes))``."""
    masks = torch.ones(1 + len(nodes), N_PLAYERS)
    masks[torch.arange(1, 1 + len(nodes)), nodes] = 0.0
    v = torch.cat([dom.coalition_values(run["backbone"], masks[None].to(DEVICE), max_batch=64)[..., 0].cpu() for dom in run["domains"]])
    return v[:, :1] - v[:, 1:]


for fam, (label, _) in FAMILIES.items():
    run = runs["adv1d", fam]
    offset = (x[run["phi"].abs().argmax(-1).numpy()] - foot + 0.5) % 1.0 - 0.5
    loo = torch.zeros(len(run["phi"]), N_PLAYERS)
    loo[:, [node_a, *neighbours]] = leave_one_out(run, [node_a, *neighbours])
    print(
        f"Adv 1D {label}: argmax|phi| - foot (cells) {quartiles(torch.from_numpy(offset * N_PLAYERS))}  "
        f"x* spike explainer {quartiles(spike(run['phi']))}  backbone {quartiles(spike(loo))}"
    )

# %%
# Drawn at the paper text width (5.5 in), so the fonts keep their size in the PDF.
fig = plt.figure(figsize=(5.5, 3.6), layout="constrained")
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.04, hspace=0.04, wspace=0.02)
axes = fig.subplots(2, 3, sharex=True, sharey="row")
fno_style = {"color": MCOLORS["op"], "ls": "--", "lw": 0.9}
mwt_style = {"color": MCOLORS["ks"], "lw": 0.6, "alpha": 0.85}

for j, (ax, s) in enumerate(zip(axes[0], HEAT_SAMPLES)):
    scale = float(analytic[s].abs().max())
    ax.plot(x, heat["u0"][s], color="gray", lw=0.7, alpha=0.55)
    ax.plot(x, analytic[s] / scale, color="black", lw=1.1)
    ax.plot(x, runs["heat1d", "fno"]["phi"][s] / scale, **fno_style)
    ax.plot(x, runs["heat1d", "mwtGupta"]["phi"][s] / scale, **mwt_style)
    ax.axvline(x[node], color="gray", lw=0.9, alpha=0.5)
    ax.set_title(f"sample {j + 1}", fontsize=8)
axes[0, 0].set_ylabel("a) Heat 1D\nscaled attribution / $u_0$", fontsize=8)

for ax, s in zip(axes[1], ADV_SAMPLES):
    ax.plot(x, adv["u0"][s], color="gray", lw=0.7, alpha=0.55)
    for fam, style in (("fno", fno_style), ("mwtGupta", mwt_style)):
        phi = runs["adv1d", fam]["phi"][s]
        ax.plot(x, phi / phi.abs().max(), **style)
    ax.axvline(x[node_a], color="gray", lw=0.9, alpha=0.5)
    ax.axvline(foot, color="black", ls=":", lw=1.0)
    ax.set_xlabel("$y$", fontsize=8)
axes[1, 0].set_ylabel("b) Advection 1D\nscaled attribution / $u_0$", fontsize=8)

# Column-major fill: row 1 holds u0 and the markers, row 2 the attributions.
fig.legend(handles=[
    Line2D([], [], color="gray", lw=0.9, alpha=0.6, label="$u_0$"),
    Line2D([], [], color="black", lw=1.1, label="analytic attribution (heat kernel)"),
    Line2D([], [], color="gray", lw=0.9, label=r"$x^\star$"),
    Line2D([], [], **fno_style, label="OperatorSHAP (FNO)"),
    Line2D([], [], color="black", ls=":", lw=1.0, label=r"foot point $x^\star - \beta t^\star$"),
    Line2D([], [], **mwt_style, label="OperatorSHAP (MWT)"),
], loc="outside upper center", ncol=3, fontsize=7.5, frameon=False, handlelength=1.8, columnspacing=1.2, borderaxespad=0.6)
for ax in axes.flat:
    ax.tick_params(labelsize=7)

save_fig(fig, "analytic_attributions", RESULTS / "figures", fmts=("pdf",))
