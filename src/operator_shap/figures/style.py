"""Matplotlib style and helpers for paper figures.

Importing this module applies the rcParams as a side effect: the figure scripts
do ``from operator_shap.figures import ...`` and the style is on.
"""

from __future__ import annotations

from pathlib import Path
from seaborn.palettes import color_palette
import matplotlib as mpl
from matplotlib.figure import Figure

PALETTE: list[str] = color_palette('colorblind', as_cmap=True)

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})

MCOLORS = {
    "op": PALETTE[0],
    "ks": PALETTE[1],
    "ps": PALETTE[2],
    "ig": PALETTE[3]
}

def save_fig(
    fig: Figure,
    name: str,
    outdir: Path | str,
    fmts: tuple[str, ...] = ("pdf", "png"),
) -> list[Path]:
    """Save ``fig`` to ``outdir/{name}.{fmt}`` for each requested format."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in fmts:
        p = outdir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt, dpi=500)
        paths.append(p)
    return paths
