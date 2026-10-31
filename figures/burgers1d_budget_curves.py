"""Burgers 1D: NRMSE vs. coalition budget.

NRMSE of the RegressionMSR and KernelSHAP ladders (budgets 64 to 8192) against the reference, at n in {16, 512, 1024} players.
The reference is the ``exact`` run at n = 16 and the ``gt`` run otherwise. OperatorSHAP and Integrated Gradients are horizontal
lines from the ``run_metrics`` CSVs. Lines are medians, bands the IQR, both pooled over the samples and the three query points.
KernelSHAP budgets <= n are skipped.

Check: the ``RMSR`` and ``KernelSHAP`` rows of each metrics CSV (budget max(2n, 1024)) must equal the ladder NRMSE computed here,
which confirms that the metrics CSV scored against the same reference. Writes ``results/figures/burgers1d_budget_curves.pdf``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from operator_shap.figures import MCOLORS, save_fig
from operator_shap.metrics import nrmse
from operator_shap.results import RESULTS, load_metrics, load_phi

DS, FAM = "burgers1d", "fno"
RESOLUTIONS = [16, 512, 1024]
REFERENCE = {16: "exact", 512: "gt", 1024: "gt"}
BUDGETS = [64, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096, 8192]
METHODS = [("ks", "KernelSHAP", "KernelSHAP"), ("ps", "RegressionMSR", "RMSR")]
FLAT = [("OperatorSHAP", "op", "OperatorSHAP", "--"), ("IG", "ig", "Integrated Gradients", ":")]
BAND_ALPHA = 0.18


# %%
def summary(err: np.ndarray) -> tuple[float, float, float]:
    """Median, 25th and 75th percentile of all entries of ``err``."""
    return float(np.median(err)), *map(float, np.percentile(err, [25, 75]))


# %% NRMSE per (n, method, budget)
err, flat = {}, {}
for n in RESOLUTIONS:
    ref = load_phi(DS, FAM, n, REFERENCE[n])
    m = load_metrics(DS, FAM, n).sort_values(["sample", "query"])
    for short, _, _ in METHODS:
        for b in BUDGETS + [max(2 * n, 1024)]:
            if short != "ks" or b > n:
                err[n, short, b] = nrmse(load_phi(DS, FAM, n, f"{short}_b{b}"), ref).numpy()
    for short, _, metrics_label in METHODS:
        saved = m[m.method == metrics_label].nrmse.to_numpy().reshape(-1, 3)
        np.testing.assert_allclose(saved, err[n, short, max(2 * n, 1024)][: len(saved)], rtol=1e-3, atol=1e-6)
    for method, *_ in FLAT:
        flat[n, method] = summary(m[m.method == method].nrmse.to_numpy())

# %%
rows = [{"n": n, "method": label, "budget": b, "median": summary(err[n, short, b])[0]}
        for n in RESOLUTIONS for short, label, _ in METHODS for b in BUDGETS if short != "ks" or b > n]
curves = pd.DataFrame(rows).pivot_table(index="budget", columns=["n", "method"], values="median")
flat_df = pd.DataFrame({k: dict(zip(["median", "q25", "q75"], v)) for k, v in flat.items()}).T
with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 200):
    print(curves.to_string())
    print(flat_df.to_string())

# %% Crossovers: smallest budget whose median NRMSE is at most the OperatorSHAP median
for n in RESOLUTIONS:
    for short, label, _ in METHODS:
        med = curves[n, label].dropna()
        below = med[med <= flat[n, "OperatorSHAP"][0]]
        print(f"n={n:>4d}  {label:>13s}  " + (f"budget {below.index[0]:>5d} (NRMSE {below.iloc[0]:.4f})" if len(below) else "not reached up to 8192"))

# %% Figure
fig, axes = plt.subplots(1, len(RESOLUTIONS), figsize=(9, 3), sharey=True, sharex=True, constrained_layout=True)
span = [BUDGETS[0], BUDGETS[-1]]
for ax, n in zip(axes, RESOLUTIONS):
    for short, label, _ in METHODS:
        x = np.array([b for b in BUDGETS if short != "ks" or b > n])
        med, lo, hi = np.array([summary(err[n, short, b]) for b in x]).T
        ax.plot(x, med, lw=1.4, color=MCOLORS[short])
        ax.fill_between(x, lo, hi, color=MCOLORS[short], alpha=BAND_ALPHA, lw=0)
    for method, short, _, ls in FLAT:
        med, lo, hi = flat[n, method]
        ax.plot(span, [med] * 2, color=MCOLORS[short], ls=ls, lw=1.2)
        ax.fill_between(span, lo, hi, color=MCOLORS[short], alpha=BAND_ALPHA, lw=0)
    ax.set(xscale="log", yscale="log", xlabel="coalition budget", title=f"$n={n}$")
    ax.grid(True, which="both", alpha=0.25)
axes[0].set_ylabel("NRMSE vs. reference\n(median, IQR)")
handles = [Line2D([], [], color=MCOLORS[short], lw=1.4, label=label) for short, label, _ in METHODS]
handles += [Line2D([], [], color=MCOLORS[short], ls=ls, lw=1.2, label=label) for _, short, label, ls in FLAT]
axes[-1].legend(handles=handles, loc="best", frameon=False, fontsize=8)
save_fig(fig, "burgers1d_budget_curves", RESULTS / "figures", fmts=("pdf",))
