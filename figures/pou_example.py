"""Grid-supported partition of unity.

Visualises the construction of App. "Construction of a partition of unity on a discrete grid": smooth masks eta_i built from the
smooth step sigma_eps around the cell boundaries m_i, so eta_i = 1 at every grid point of cell B_i and 0 at all others. Top row:
one grid point per mask; the smooth mask of a coalition S, m_S = sum_{i in S} eta_i, sampled at the grid points is the binary grid
mask. Bottom row: the same masks on a grid with two points per mask, and a coalition that holds one of the two grid points of cell
B_2, so eta_2 is half activated. Writes ``results/figures/pou_example.pdf``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from operator_shap.figures.style import PALETTE, save_fig
from operator_shap.results import RESULTS

P = 5                                    # masks
EPS = 0.04                               # transition half-width (exaggerated for visibility, below 1/(4P))
COALITIONS = [[1, 2, 4], [2, 4, 5, 8, 9]]  # coalition per row, as grid-point indices
BOUNDS = np.arange(1, P) / P             # cell boundaries m_i
X = np.linspace(0.0, 1.0, 2001)
GRIDS = [(np.arange(k * P) + 0.5) / (k * P) for k in (1, 2)]   # cell-centred grids with 1 and 2 points per mask


# %%
def g(t: np.ndarray) -> np.ndarray:
    return np.where(t > 0, np.exp(-1.0 / np.where(t > 0, t, 1.0)), 0.0)


def sigma(t: np.ndarray, eps: float) -> np.ndarray:
    """Smooth step: 0 for t <= -eps, 1 for t >= eps."""
    up, down = g(t + eps), g(eps - t)
    return up / (up + down)


def masks(x: np.ndarray, bounds: np.ndarray, eps: float) -> np.ndarray:
    """(len(bounds) + 1, len(x)) masks eta_i switching at the cell boundaries."""
    steps = np.stack([sigma(x - m, eps) for m in bounds])         # (p-1, len(x))
    upper = np.concatenate([np.ones((1, x.size)), steps])          # s_{i-1}, with s_0 = 1
    lower = np.concatenate([steps, np.zeros((1, x.size))])         # s_i, with s_p = 0
    return upper - lower


ETA = masks(X, BOUNDS, EPS)
assert np.allclose(ETA.sum(0), 1.0)
for grid in GRIDS:
    assert np.allclose(masks(grid, BOUNDS, EPS), np.repeat(np.eye(P), grid.size // P, axis=1))

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharey=True)

for (ax_l, ax_r), grid, s in zip(axes, GRIDS, COALITIONS):
    mu = np.isin(np.arange(grid.size), s).reshape(P, -1).mean(1)   # share of each cell's grid points in S
    for i in range(P):
        ax_l.plot(X, ETA[i], color=PALETTE[i % len(PALETTE)], lw=1.5, label=rf"$\eta_{{{i + 1}}}$")
        at = grid[(grid > i / P) & (grid < (i + 1) / P)]
        ax_l.plot(at, np.ones_like(at), "o", color=PALETTE[i % len(PALETTE)], ms=4)
    for ax in (ax_l, ax_r):
        for m in BOUNDS:
            ax.axvline(m, color="0.8", lw=0.6, ls=":")
        ax.set_xticks(grid, [rf"$x_{{{j + 1}}}$" for j in range(grid.size)])
        ax.set_ylim(-0.05, 1.15)
    ax_l.set_ylabel(r"$\eta_i(x)$")
    ax_l.set_title(rf"partition of unity, ${grid.size}$ grid points, $\varepsilon={EPS}$")
    ax_r.plot(X, mu @ ETA, color="black", lw=1.5, label=r"$\sum_i\mu_i\eta_i$")
    ax_r.plot(grid, mu @ masks(grid, BOUNDS, EPS), "o", color=PALETTE[3], ms=6, label="values at the grid points")
    ax_r.set_title(r"$S=\{" + ",".join(f"x_{{{j + 1}}}" for j in s) + r"\}$")
axes[1, 0].set_xlabel("$x$")
axes[1, 1].set_xlabel("$x$")
axes[1, 0].legend(ncol=5, frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.28))
axes[1, 1].legend(ncol=2, frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.28))
fig.tight_layout()
save_fig(fig, "pou_example", RESULTS / "figures", fmts=("pdf",))
