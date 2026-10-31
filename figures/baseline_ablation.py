"""Baseline ablation: zero vs. per-sample mean.

One Darcy 2D test sample at 32x32 players and the domain centre: the permeability u_0, and for each baseline the reference (``gt``
run, 3e5 evaluations) and OperatorSHAP (the explainer trained for that baseline, hard-normalised). The zero baseline uses the
``darcy2d`` explainer, the mean baseline the ``darcy2dmean`` explainer; both explain the same backbone. The explainer metrics per
baseline are in ``mean_baseline.tex`` of ``make_tables.py``. For the caption, the script writes the Pearson correlation between the
two references (median over test inputs and query points) for Darcy 2D and diffusion--sorption 1D to
``results/tables/numbers_baseline.tex``. Writes ``results/figures/baseline_ablation.pdf``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from seaborn.palettes import color_palette

from operator_shap.backbone import load_pair
from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains, explain
from operator_shap.figures import panels, save_fig
from operator_shap.metrics import pearson_r
from operator_shap.results import RESULTS, compose_config, load_phi, write_numbers

# %%
N_PLAYERS = 1024
SAMPLE_IDX = 0
DEVICE = "cpu"
# (row title, Hydra dataset, dataset_short, model family)
BASELINES = [("zero baseline", "darcy_2d", "darcy2d", "localFNO"), ("mean baseline", "darcy_2d_mean", "darcy2dmean", "localFNO")]
CROSS = [("Darcy 2D", "darcy2d", "darcy2dmean", "localFNO"), ("Diff.--Sorp. 1D", "diffsorp1d", "diffsorp1dmean", "fno")]

CMAP_INPUT = color_palette("icefire", as_cmap=True)
CMAP_SHAP = color_palette("RdBu_r", as_cmap=True)

# %%
rows = []
for title, dataset, ds, fam in BASELINES:
    cfg = compose_config("run_metrics", [f"dataset={dataset}", f"attribution.n_players={N_PLAYERS}", f"device={DEVICE}"])
    backbone, explainer, baseline_mode, query_encoding = load_pair(cfg)
    test_ds, test_loader, _ = build_test_loader(cfg, baseline_mode)
    domain_of_batch = build_domains(cfg, test_loader, query_encoding=query_encoding)
    for i, batch in enumerate(test_loader):
        if i == SAMPLE_IDX:
            dom = domain_of_batch(batch, eval_mode=True)
            op = explain(explainer, backbone, dom)[0, 0].cpu().numpy()
            break
    ref = load_phi(ds, fam, N_PLAYERS, "gt")[SAMPLE_IDX, 0].numpy()
    rows.append({"title": title, "op": op, "ref": ref})

native = test_ds[(test_ds.spatial_res[0], SAMPLE_IDX)]
u0_full = native["u0"][0].numpy()
x_star = native["coords"][native["x_star_node"][0]].numpy()

# %%
# Pearson between the zero- and mean-baseline references, median over test inputs and query points.
numbers = {}
for label, ds_zero, ds_mean, fam in CROSS:
    for n in (64, 1024):
        a, b = (load_phi(ds, fam, n, "gt") for ds in (ds_zero, ds_mean))
        r = pearson_r(a.flatten(0, 1), b.flatten(0, 1))
        print(f"{label:>16s}  n={n:4d}  reference zero vs. mean: Pearson median {r.median():+.3f}  [{r.quantile(0.25):+.3f}, {r.quantile(0.75):+.3f}]")
        numbers[f"baseline/{ds_zero}/{n}/reference_pearson"] = float(r.median())
write_numbers(RESULTS / "tables/numbers_baseline.tex", numbers, digits=2)

# %%
extent = (0.0, 1.0, 0.0, 1.0)
npd = int(round(N_PLAYERS ** 0.5))
fig = plt.figure(figsize=(10.5, 5.4), constrained_layout=True)
gs = fig.add_gridspec(4, 5, width_ratios=[1, 0.05, 1, 1, 0.05])

ax_u0 = fig.add_subplot(gs[1:3, 0])
im_in = panels.field_2d(ax_u0, u0_full, extent=extent, cmap=CMAP_INPUT)
ax_u0.set_title(r"$u_0$ (permeability)")
fig.colorbar(im_in, cax=fig.add_subplot(gs[1:3, 1]))
content_axes = [ax_u0]

for r, rd in enumerate(rows):
    fields = [rd["ref"].reshape(npd, npd), rd["op"].reshape(npd, npd)]
    lim = float(max(np.abs(f).max() for f in fields))
    rs = slice(2 * r, 2 * (r + 1))
    axes = [fig.add_subplot(gs[rs, c]) for c in (2, 3)]
    for ax, field in zip(axes, fields):
        im = panels.field_2d(ax, field, extent=extent, cmap=CMAP_SHAP, vmin=-lim, vmax=lim)
        content_axes.append(ax)
    axes[0].set_title(f"RegressionMSR, {rd['title']}")
    axes[1].set_title(f"OperatorSHAP, {rd['title']}")
    fig.colorbar(im, cax=fig.add_subplot(gs[rs, 4]))

# Array axis 0 is plotted vertically, so the query node (c0, c1) sits at (x, y) = (c1, c0).
for ax in content_axes:
    ax.scatter([x_star[1]], [x_star[0]], color="red", s=20, zorder=5)
    ax.set(xlabel=r"$x$", ylabel=r"$y$")

save_fig(fig, "baseline_ablation", RESULTS / "figures", fmts=("pdf",))
