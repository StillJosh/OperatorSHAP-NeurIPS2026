"""Shapley values vs. Aumann-Shapley density.

One Burgers 1D test sample, centre query, at n=32 and n=128 players: the reference (``gt`` run, 3e5 evaluations) scaled by n as
bars, and the OperatorSHAP density (burgers1d/fno explainer, hard-normalised discrete values divided by the cell measure 1/n) as a
line. Writes ``results/figures/Comparison_Shap_Operator.pdf``.
"""

# %%
import matplotlib.pyplot as plt

from operator_shap.backbone import load_pair
from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains, explain
from operator_shap.figures import save_fig
from operator_shap.metrics import pearson_r
from operator_shap.results import RESULTS, compose_config, load_phi

# %%
DATASET, DS, FAM = "burgers_1d", "burgers1d", "fno"
N_PLAYERS = (32, 128)
SAMPLE_IDX = 4
DEVICE = "cpu"

# %%
cfgs = [compose_config("run_metrics", [f"dataset={DATASET}", f"attribution.n_players={n}", f"device={DEVICE}"]) for n in N_PLAYERS]
backbone, explainer, baseline_mode, query_encoding = load_pair(cfgs[0])

panels = []
for n, cfg in zip(N_PLAYERS, cfgs):
    _, test_loader, _ = build_test_loader(cfg, baseline_mode)
    domain_of_batch = build_domains(cfg, test_loader, query_encoding=query_encoding)
    dom = domain_of_batch(next(b for i, b in enumerate(test_loader) if i == SAMPLE_IDX), eval_mode=True)
    op = explain(explainer, backbone, dom)[0, 0].cpu()
    ref = load_phi(DS, FAM, n, "gt")[SAMPLE_IDX, 0]
    print(f"n={n:4d}  Pearson(OperatorSHAP, reference) = {pearson_r(op, ref):.4f}")
    # Discrete values over the cell measure: the density at the cell centres.
    panels.append({"n": n, "x": dom.coords[:, 0].cpu().numpy(), "op": (op / dom.cell_measures.cpu()).numpy(), "ref": (ref * n).numpy()})

# %%
fig, axes = plt.subplots(1, len(panels), figsize=(9, 3.6), sharey=True)
for ax, p in zip(axes, panels):
    ax.grid(linewidth=0.2)
    ax.bar(p["x"], p["ref"], width=1.0 / p["n"], edgecolor="none", label="RegressionMSR")
    ax.plot(p["x"], p["op"], lw=1.6, color="orange", label="OperatorSHAP")
    ax.set_title(f"n={p['n']}")
axes[-1].legend(loc="lower right")
plt.tight_layout()
save_fig(fig, "Comparison_Shap_Operator", RESULTS / "figures", fmts=("pdf",))
