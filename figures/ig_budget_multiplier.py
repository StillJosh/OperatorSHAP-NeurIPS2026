"""Methods at matched cost.

Faithfulness R^2 and CPU runtime per explained sample against a budget multiplier m, at n = 256 players.

- Integrated Gradients uses m path steps (base 1 step).
- RegressionMSR and KernelSHAP use 128 m coalitions (base 128, the smallest budget of the n = 256 reference ladders), so
  m = 1, ..., 256 spans the ladder budgets 128 to 32768. KernelSHAP budgets <= n (m <= 2) are skipped.
- OperatorSHAP does not depend on m: one explainer forward and one backbone forward for the normalization. It is a horizontal line.
- R^2: ``run_metrics`` rows (30 test samples, three query points each, 10000 Shapley-kernel coalitions); lines are medians. The
  spread over samples is wider than the gaps between methods, so no bands are drawn; the quartiles are printed. The dotted grey
  line is the R^2 of the reference (300k-coalition RegressionMSR). The R^2 axes span the IG, RegressionMSR, OperatorSHAP and
  reference medians; KernelSHAP at m = 4 (512 coalitions) falls below the axis on three panels.
- Runtime: ``run_timing`` rows on one CPU thread (10 validation samples, three query points each); lines are medians, bands the
  IQR. OperatorSHAP pools its timings over all m.

Writes ``results/figures/ig_budget_multiplier.pdf`` and ``results/tables/numbers_matched_cost.tex``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from operator_shap.figures import MCOLORS, save_fig
from operator_shap.results import RESULTS, load_metrics, load_timing, write_numbers

PANELS = [("burgers1d", "fno", "Burgers 1D"), ("heat2d", "localFNO", "Heat 2D"), ("rxndiff2d", "localFNO", "Reaction-Diffusion 2D"),
          ("sw2d", "localFNO", "Shallow Water 2D")]
N_PLAYERS = 256
MULTIPLIERS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CLASSICAL_BASE = 128
METHODS = [("IG", "ig", "Integrated Gradients"), ("RMSR", "ps", "RegressionMSR"), ("KernelSHAP", "ks", "KernelSHAP")]
BAND_ALPHA = 0.18

# %%
metrics = pd.concat([load_metrics(ds, fam, N_PLAYERS, f"_m{m}").assign(m=m) for ds, fam, _ in PANELS for m in MULTIPLIERS], ignore_index=True)
timing = pd.concat([load_timing(ds, fam, N_PLAYERS, f"_m{m}").assign(m=m) for ds, fam, _ in PANELS for m in MULTIPLIERS], ignore_index=True)


# %% R^2 and runtime per multiplier
def quartiles(s: pd.Series) -> pd.Series:
    """Median and quartiles of ``s``."""
    return pd.Series({"med": s.median(), "q25": s.quantile(0.25), "q75": s.quantile(0.75)})


keep = lambda df: df[(df.method != "KernelSHAP") | (df.m * CLASSICAL_BASE > N_PLAYERS)]
r2 = keep(metrics).groupby(["pde", "method", "m"]).r2.apply(quartiles).unstack()
sec = keep(timing).groupby(["pde", "method", "m"]).seconds.apply(quartiles).unstack()
# OperatorSHAP and Reference lines from the main metrics runs, so they follow the evaluated explainer
main = pd.concat([load_metrics(ds, fam, N_PLAYERS) for ds, fam, _ in PANELS])
r2_flat = main[main.method.isin(["OperatorSHAP", "Reference"])].groupby(["pde", "method"]).r2.apply(quartiles).unstack()
sec_flat = timing[timing.method == "OperatorSHAP"].groupby("pde").seconds.apply(quartiles).unstack()
with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 200):
    print(pd.concat({"r2": r2.med, "seconds": sec.med}, axis=1).unstack("method").to_string())
    print(r2[["q25", "q75"]].unstack("method").to_string())
    print(r2_flat.to_string())
    print(sec_flat.to_string())

# %% Matched quality: smallest multiplier whose median R^2 reaches the OperatorSHAP median, its runtime, and the ratio to OperatorSHAP
rows = []
for ds, _, title in PANELS:
    op_r2, op_sec = r2_flat.loc[(ds, "OperatorSHAP"), "med"], sec_flat.loc[ds, "med"]
    for method, _, label in METHODS:
        med = r2.loc[(ds, method), "med"]
        hit = med[med >= op_r2]
        m = hit.index[0] if len(hit) else None
        seconds = sec.med.get((ds, method, m), np.nan)
        rows.append({"pde": title, "method": label, "OS r2": op_r2, "best r2": med.max(), "m": m, "seconds": seconds, "seconds / OS": seconds / op_sec})
with pd.option_context("display.float_format", "{:.4g}".format, "display.width", 200):
    print(pd.DataFrame(rows).to_string(index=False))
# Runtime of IG over OperatorSHAP at the first multiplier where IG reaches the OperatorSHAP R2, over the PDEs where it does
reached = pd.DataFrame(rows).dropna(subset=["m"])
ratio = reached[reached.method == "Integrated Gradients"]["seconds / OS"]
write_numbers(RESULTS / "tables" / "numbers_matched_cost.tex", {"matched/ig_ratio_min": ratio.min(), "matched/ig_ratio_max": ratio.max()}, digits=1)

# %% Figure
fig, axes = plt.subplots(2, len(PANELS), figsize=(13, 5.2), sharex=True, constrained_layout=True)
ms = sorted(metrics.m.unique())
span = [ms[0], ms[-1]]
for col, (ds, _, title) in enumerate(PANELS):
    ax_t, ax_q = axes[:, col]
    for method, short, _ in METHODS:
        s = sec.loc[(ds, method)]
        ax_t.plot(s.index, s.med, color=MCOLORS[short], lw=1.4, marker="o", ms=3)
        ax_t.fill_between(s.index, s.q25, s.q75, color=MCOLORS[short], alpha=BAND_ALPHA, lw=0)
        ax_q.plot(r2.loc[(ds, method)].index, r2.loc[(ds, method)].med, color=MCOLORS[short], lw=1.4, marker="o", ms=3)
    ax_t.plot(span, [sec_flat.loc[ds, "med"]] * 2, color=MCOLORS["op"], ls="--", lw=1.2)
    ax_t.fill_between(span, sec_flat.loc[ds, "q25"], sec_flat.loc[ds, "q75"], color=MCOLORS["op"], alpha=BAND_ALPHA, lw=0)
    ax_q.axhline(r2_flat.loc[(ds, "OperatorSHAP"), "med"], color=MCOLORS["op"], ls="--", lw=1.2)
    ax_q.axhline(r2_flat.loc[(ds, "Reference"), "med"], color="grey", ls=":", lw=1.0)
    shown = pd.concat([r2.loc[(ds, "IG"), "med"], r2.loc[(ds, "RMSR"), "med"], r2_flat.loc[ds, "med"]])
    pad = 0.1 * (shown.max() - shown.min())
    ax_q.set_ylim(shown.min() - pad, shown.max() + pad)
    ax_t.set(xscale="log", yscale="log", title=title)
    ax_q.set(xscale="log", xlabel="multiplier $m$")
    ax_q.set_xticks(ms, [str(m) for m in ms])
    ax_q.minorticks_off()
    for ax in (ax_t, ax_q):
        ax.grid(True, which="both", alpha=0.25)
axes[0, 0].set_ylabel("seconds per explained sample\n(1 CPU thread)")
axes[1, 0].set_ylabel("faithfulness $R^2$")
handles = [Line2D([], [], color=MCOLORS[short], lw=1.4, marker="o", ms=3, label=f"{label} ({'$m$ steps' if short == 'ig' else '$128m$ coalitions'})")
           for _, short, label in METHODS]
handles += [Line2D([], [], color=MCOLORS["op"], ls="--", lw=1.2, label="OperatorSHAP"), Line2D([], [], color="grey", ls=":", lw=1.0, label="reference")]
fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
save_fig(fig, "ig_budget_multiplier", RESULTS / "figures", fmts=("pdf",))
