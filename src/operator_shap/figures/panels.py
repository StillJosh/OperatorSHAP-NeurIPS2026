"""Reusable plotting atoms for paper figures.

Each function takes an existing ``ax`` and only handles the visual
payload; titles, labels, limits and legends stay in the figure script.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage


def field_2d(
    ax: Axes,
    field: np.ndarray,
    *,
    extent=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> AxesImage:
    """Plot a 2D field as a heatmap with ``origin='lower'``."""
    return ax.imshow(
        field,
        extent=extent, origin="lower", cmap=cmap,
        vmin=vmin, vmax=vmax, aspect="equal",
    )
