"""Heat 2D triptych.

Input and label of one Heat 2D test sample, and the explanation of the centre query at 8x8 and 16x16 players: OperatorSHAP
(heat2d/localFNO explainer, hard-normalised), the reference (``gt`` run, 3e5 evaluations), and their difference. Writes
``results/figures/heat2d_triptych_heatmap.pdf``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from seaborn.palettes import color_palette

from operator_shap.backbone import load_pair
from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains, explain
from operator_shap.figures import panels, save_fig
from operator_shap.metrics import pearson_r
from operator_shap.results import RESULTS, compose_config, load_phi

# %%
DATASET, DS, FAM = "heat_2d", "heat2d", "localFNO"
N_PLAYERS = (64, 256)
SAMPLE_IDX = 0
DEVICE = "cpu"

CMAP_INPUT = color_palette("icefire", as_cmap=True)
CMAP_SHAP = color_palette("RdBu_r", as_cmap=True)

# %%
cfgs = [compose_config("run_metrics", [f"dataset={DATASET}", f"attribution.n_players={n}", f"device={DEVICE}"]) for n in N_PLAYERS]
backbone, explainer, baseline_mode, query_encoding = load_pair(cfgs[0])

rows = []
for n, cfg in zip(N_PLAYERS, cfgs):
    test_ds, test_loader, n_eval = build_test_loader(cfg, baseline_mode)
    domain_of_batch = build_domains(cfg, test_loader, query_encoding=query_encoding)
    ref = load_phi(DS, FAM, n, "gt")[:, 0]
    op = []
    for batch in test_loader:
        dom = domain_of_batch(batch, eval_mode=True)
        op.append(explain(explainer, backbone, dom)[0, 0].cpu())
    op = torch.stack(op)
    pear = pearson_r(op, ref[:n_eval])
    rows.append({"n": n, "op": op[SAMPLE_IDX], "ref": ref[SAMPLE_IDX]})
    print(f"n={n:4d}  Pearson sample {SAMPLE_IDX}: {pear[SAMPLE_IDX]:.4f}  median over {n_eval}: {pear.median():.4f}")

native = test_ds[(test_ds.spatial_res[0], SAMPLE_IDX)]
u0_full, y_true_full = native["u0"][0].numpy(), native["y_true"][0].numpy()
# Strided grids keep the native query node, so its native coordinate marks the query in every panel.
x_star = native["coords"][native["x_star_node"][0]].numpy()

# %%
extent = (0.0, 1.0, 0.0, 1.0)
nrows = len(rows)
fig = plt.figure(figsize=(13.5, 2.7 * nrows), constrained_layout=True)
gs = fig.add_gridspec(2 * nrows, 6, width_ratios=[1, 0.05, 1, 1, 1, 0.05])

u_lim = (float(min(u0_full.min(), y_true_full.min())), float(max(u0_full.max(), y_true_full.max())))
ax_u0 = fig.add_subplot(gs[0:nrows, 0])
ax_uT = fig.add_subplot(gs[nrows : 2 * nrows, 0])
im_in = panels.field_2d(ax_u0, u0_full, extent=extent, cmap=CMAP_INPUT, vmin=u_lim[0], vmax=u_lim[1])
panels.field_2d(ax_uT, y_true_full, extent=extent, cmap=CMAP_INPUT, vmin=u_lim[0], vmax=u_lim[1])
ax_u0.set_title(r"$u_0$")
ax_uT.set_title(r"$u(t^*)$")
fig.colorbar(im_in, cax=fig.add_subplot(gs[:, 1]))
content_axes = [ax_u0, ax_uT]

for r, rd in enumerate(rows):
    npd = int(round(rd["n"] ** 0.5))
    op, ref = rd["op"].reshape(npd, npd).numpy(), rd["ref"].reshape(npd, npd).numpy()
    diff = op - ref
    lim = float(max(np.abs(op).max(), np.abs(ref).max(), np.abs(diff).max()))
    rs = slice(2 * r, 2 * (r + 1))
    axes = [fig.add_subplot(gs[rs, c]) for c in (2, 3, 4)]
    for ax, field in zip(axes, (op, ref, diff)):
        im = panels.field_2d(ax, field, extent=extent, cmap=CMAP_SHAP, vmin=-lim, vmax=lim)
        content_axes.append(ax)
    axes[0].set_title(rf"OperatorSHAP ($n={rd['n']}$)")
    axes[1].set_title(rf"RegressionMSR ($n={rd['n']}$)")
    axes[2].set_title(r"$\varphi_{\mathrm{OperatorSHAP}} - \varphi_{\mathrm{RegressionMSR}}$")
    fig.colorbar(im, cax=fig.add_subplot(gs[rs, 5]))

# Array axis 0 is plotted vertically, so the query node (c0, c1) sits at (x, y) = (c1, c0).
for ax in content_axes:
    ax.scatter([x_star[1]], [x_star[0]], color="red", s=20, zorder=5)
    ax.set(xlabel=r"$x$", ylabel=r"$y$")

save_fig(fig, "heat2d_triptych_heatmap", RESULTS / "figures", fmts=("pdf",))
