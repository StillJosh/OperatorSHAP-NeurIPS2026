"""Amortization of OperatorSHAP.

Cumulative backbone evaluations against the number of explained samples, at n = 256 players.

- OperatorSHAP: ``backbone_evals`` of the explainer training run (its ``summary.json``), plus 2 evaluations per sample
  (one explainer forward, one backbone forward for the normalization).
- RegressionMSR, KernelSHAP: the smallest budget of the reference ladder whose median NRMSE vs. the ``gt`` reference
  is at most the median NRMSE of OperatorSHAP (``run_metrics`` CSV), at that budget per sample.
- Medians pool the samples and the three query points. KernelSHAP budgets <= n are skipped.
- Crossover: N = E_train / (b - 2) samples, with E_train the training evaluations and b the budget.

Writes ``results/figures/amortization.pdf``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from operator_shap.figures import MCOLORS, save_fig
from operator_shap.metrics import nrmse
from operator_shap.results import RESULTS, load_metrics, load_phi, load_summary

PANELS = [("burgers1d", "fno", "Burgers 1D"), ("heat2d", "localFNO", "Heat 2D"), ("sw2d", "localFNO", "Shallow Water 2D")]
N_PLAYERS = 256
BUDGETS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
METHODS = [("ps", "RegressionMSR"), ("ks", "KernelSHAP")]
OP_PER_SAMPLE = 2

# %% Training cost and equal-quality budgets
rows = []
for ds, fam, title in PANELS:
    train_evals = int(load_summary(ds, fam, "explainer")["backbone_evals"])
    m = load_metrics(ds, fam, N_PLAYERS)
    op_nrmse = float(m[m.method == "OperatorSHAP"].nrmse.median())
    ref = load_phi(ds, fam, N_PLAYERS, "gt")
    for short, label in METHODS:
        budgets = [b for b in BUDGETS if short != "ks" or b > N_PLAYERS]
        med = {b: float(np.median(nrmse(load_phi(ds, fam, N_PLAYERS, f"{short}_b{b}"), ref).numpy())) for b in budgets}
        budget = min((b for b in budgets if med[b] <= op_nrmse), default=None)
        rows.append({
            "dataset": title, "method": label, "train_evals": train_evals, "op_nrmse": op_nrmse, "budget": budget,
            "nrmse_at_budget": med[budget] if budget else np.nan, "nrmse_max_budget": med[budgets[-1]],
            "crossover": train_evals / (budget - OP_PER_SAMPLE) if budget else np.nan,
        })
amort = pd.DataFrame(rows)
with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 200):
    print(amort.to_string(index=False))

# %% Figure
x_max = max(1e6, 10 * amort.crossover.max())
samples = np.logspace(0, np.log10(x_max), 200)

fig, axes = plt.subplots(1, len(PANELS), figsize=(15, 5), sharey=True, constrained_layout=True)
for ax, (ds, fam, title) in zip(axes, PANELS):
    sub = amort[amort.dataset == title]
    train_evals = sub.train_evals.iloc[0]
    ax.plot(samples, train_evals + OP_PER_SAMPLE * samples, color=MCOLORS["op"], ls="--", lw=1.4,
            label=f"OperatorSHAP (training + {OP_PER_SAMPLE}/sample)")
    for (short, label), row in zip(METHODS, sub.itertuples()):
        if row.budget is None or np.isnan(row.budget):
            continue
        ax.plot(samples, row.budget * samples, color=MCOLORS[short], lw=1.4, label=f"{label} ({row.budget:,.0f}/sample)")
        ax.axvline(row.crossover, color=MCOLORS[short], lw=0.8, ls=":")
        ax.text(row.crossover, 0.03, f"{label}: {row.crossover:,.0f}", transform=ax.get_xaxis_transform(), rotation=90,
                ha="right", va="bottom", fontsize=9, color=MCOLORS[short])
    ax.set(xscale="log", yscale="log", xlabel="number of explained samples", title=f"{title} ($n={N_PLAYERS}$)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
axes[0].set_ylabel("cumulative backbone evaluations")
save_fig(fig, "amortization", RESULTS / "figures", fmts=("pdf",))
