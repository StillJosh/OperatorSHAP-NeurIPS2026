"""MGN cylinder flow and airfoil: velocity and OperatorSHAP attribution.

One stacked figure per dataset: the input velocity magnitude |u_0| of one test sample on top and the OperatorSHAP density phi(x) of
the first query point below, both over a Delaunay triangulation of the mesh nodes with the obstacle in white. The sample and its
query point come from the test loader of ``run_metrics.py``. Writes ``results/figures/mgn_stacked_{dataset}_{sample}.pdf``; the
cylinder figure is Figure 1 of the paper.
"""

# %%
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path as MPath
from scipy.spatial import ConvexHull

import operator_shap.figures.style 
from operator_shap.backbone import load_pair
from operator_shap.dataset import build_test_loader
from operator_shap.domain import build_domains
from operator_shap.results import RESULTS, compose_config

# %%
DEVICE = "cpu"

# view_cutout (xmin, xmax, ymin, ymax) sets only the axes limits; panel_w is the panel width in inches.
DATASETS = {
    "airfoil": {"dataset": "mgn_airfoil", "sample_idx": 1, "view_cutout": (-1.0, 2.0, -1.0, 1.0), "obstacle_kind": "polygon", "panel_w": 4.5},
    "cylinder_flow": {"dataset": "mgn_cylinder_flow", "sample_idx": 1, "view_cutout": (0.0, 1.6, 0.0, 0.41), "obstacle_kind": "circle", "panel_w": 10.0},
}

CMAP_FIELD = "viridis"
CMAP_PHI = "RdBu_r"
MESH_COLOR = "0.15"
MESH_LW = 0.1
MESH_ALPHA = 1
MESH_NODE_S = 1.5         # scatter marker area (pt^2) for mesh nodes
MESH_NODE_ALPHA = 0.9

# Figure layout in inches.
_L_MARGIN = 0.45     # left:  ylabel + yticklabels
_R_MARGIN = 0.7      # right: cbar label
_T_MARGIN = 0.05     # top
_B_MARGIN = 0.5      # bottom: xlabel + xticklabels
_GAP = 0.08          # vertical gap between rows
_CBAR_W = 0.12       # vertical colorbar width
_CBAR_PAD = 0.08     # gap between image and colorbar

CBAR_LABEL_V = r"$|u_0|$"
CBAR_LABEL_PHI = r"$\varphi(x)$"

# At higher dpi the lw=0.1 pt mesh lines become wider than a pixel and dominate the heatmap.
_SAVE_DPI = 200


def cutout_aspect(cutout: tuple[float, float, float, float]) -> float:
    """Width / height of a ``(xmin, xmax, ymin, ymax)`` cutout."""
    xmin, xmax, ymin, ymax = cutout
    return (xmax - xmin) / (ymax - ymin)


# %%
def airfoil_polygon(mesh_pos: np.ndarray, node_type: np.ndarray) -> np.ndarray:
    """Convex hull of the AIRFOIL-typed nodes, in CCW vertex order.

    NACA-like shapes are essentially convex (sharp trailing edge,
    rounded leading edge), so the hull tightly traces the boundary.
    """
    pts = mesh_pos[node_type == 2]
    hull = ConvexHull(pts)
    return pts[hull.vertices]


def cylinder_disk(
    mesh_pos: np.ndarray, node_type: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Locate the cylinder by stripping channel-wall nodes from WALL_BOUNDARY.

    MGN's ``cylinder_flow`` tags the channel top/bottom walls with the
    same ``WALL_BOUNDARY`` (6) type as the cylinder boundary. The two
    sets are easy to separate: channel walls sit on the bounding box,
    the cylinder doesn't. Cylinder centre and radius are then the mean
    of the remaining nodes and their mean distance to that mean.
    """
    wall = mesh_pos[node_type == 6]
    bbox = (
        mesh_pos[:, 0].min(), mesh_pos[:, 0].max(),
        mesh_pos[:, 1].min(), mesh_pos[:, 1].max(),
    )
    span = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    eps = 1e-3 * span
    on_domain = (
        (np.abs(wall[:, 0] - bbox[0]) < eps)
        | (np.abs(wall[:, 0] - bbox[1]) < eps)
        | (np.abs(wall[:, 1] - bbox[2]) < eps)
        | (np.abs(wall[:, 1] - bbox[3]) < eps)
    )
    cyl = wall[~on_domain]
    if cyl.size == 0:
        raise RuntimeError(
            "No interior wall nodes found — cannot locate cylinder",
        )
    center = cyl.mean(axis=0)
    radius = float(np.linalg.norm(cyl - center, axis=1).mean())
    return center, radius


# %%
def render_field(
    ax,
    mesh_pos: np.ndarray,
    node_type: np.ndarray,
    field: np.ndarray,
    cfg: dict,
    *,
    cmap: str = CMAP_FIELD,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Render any per-node scalar field over the (cropped) mesh.

    Pipeline (shared between velocity-magnitude and GINO-attribution
    panels):
      1. Delaunay-triangulate ``mesh_pos``;
      2. mask any triangle whose centroid lies inside the obstacle so
         neither colour nor mesh lines bleed across the white overlay;
      3. ``tripcolor`` (Gouraud-shaded) + ``triplot`` for mesh edges
         + ``scatter`` for mesh nodes;
      4. white-filled obstacle patch on top;
      5. ``set_aspect('equal')`` and clip to ``cfg['view_cutout']``.
    """
    triang = mtri.Triangulation(mesh_pos[:, 0], mesh_pos[:, 1])
    centroids = mesh_pos[triang.triangles].mean(axis=1)

    if cfg["obstacle_kind"] == "polygon":
        poly_xy = airfoil_polygon(mesh_pos, node_type)
        mask = MPath(poly_xy).contains_points(centroids)
        node_in_obstacle = MPath(poly_xy).contains_points(mesh_pos)
    elif cfg["obstacle_kind"] == "circle":
        center, radius = cylinder_disk(mesh_pos, node_type)
        mask = np.linalg.norm(centroids - center, axis=1) < radius
        node_in_obstacle = np.linalg.norm(mesh_pos - center, axis=1) < radius
    else:
        raise ValueError(f"unknown obstacle_kind {cfg['obstacle_kind']!r}")
    triang.set_mask(mask)

    pc = ax.tripcolor(
        triang, field, shading="gouraud",
        cmap=cmap, vmin=vmin, vmax=vmax,
    )
    ax.triplot(triang, color=MESH_COLOR, lw=MESH_LW, alpha=MESH_ALPHA, rasterized=True)
    visible_nodes = mesh_pos[~node_in_obstacle]
    ax.scatter(
        visible_nodes[:, 0], visible_nodes[:, 1],
        s=MESH_NODE_S, c=MESH_COLOR, alpha=MESH_NODE_ALPHA,
        linewidths=0, rasterized=True,
    )

    if cfg["obstacle_kind"] == "polygon":
        ax.add_patch(Polygon(
            poly_xy, closed=True, facecolor="white",
            edgecolor="white", linewidth=0, zorder=5,
        ))
    else:
        ax.add_patch(Circle(
            center, radius, facecolor="white",
            edgecolor="white", linewidth=0, zorder=5,
        ))

    xmin, xmax, ymin, ymax = cfg["view_cutout"]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    return pc


# %%
@torch.no_grad()
def explained_sample(cfg: dict) -> dict:
    """Mesh, ``|u0|``, query node, and OperatorSHAP density per node of test sample ``cfg['sample_idx']`` at its first query point."""
    run_cfg = compose_config("run_metrics", [f"dataset={cfg['dataset']}", "attribution=eval_mgn", f"device={DEVICE}"])
    backbone, explainer, baseline_mode, query_encoding = load_pair(run_cfg)
    _, test_loader, _ = build_test_loader(run_cfg, baseline_mode)
    data = next(b for i, b in enumerate(test_loader) if i == cfg["sample_idx"])
    dom = build_domains(run_cfg, test_loader, query_encoding=query_encoding)(data, eval_mode=True)
    raw = explainer(dom.explainer_input()).reshape(dom.n_samples, dom.n_queries, dom.n_points)[0, 0]
    v_grand, v_empty = dom.game_values(backbone)
    w = dom.node_weights.reshape(-1)
    # Hard normalisation of the density: spread the efficiency residual uniformly over the domain measure.
    phi = raw + ((v_grand - v_empty)[0, 0] - (raw * w).sum()) / w.sum()
    mesh_pos = data["mesh_pos_raw"][0].numpy()
    return {
        "mesh_pos": mesh_pos,
        "node_type": data["node_type"][0].numpy(),
        "speed": data["u0"][0].norm(dim=-1).numpy(),
        "phi": phi.cpu().numpy(),
        "x_star": mesh_pos[int(dom.x_star[0, 0])],
    }


# %%
def make_stacked_figure(cfg: dict, s: dict):
    """Velocity magnitude on top and OperatorSHAP density below, each with a colorbar of the panel height."""
    panel_w = cfg["panel_w"]
    panel_h = panel_w / cutout_aspect(cfg["view_cutout"])
    fig_w = _L_MARGIN + panel_w + _CBAR_PAD + _CBAR_W + _R_MARGIN
    fig_h = _T_MARGIN + 2 * panel_h + _GAP + _B_MARGIN
    fig, (ax_v, ax_e) = plt.subplots(2, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=_L_MARGIN / fig_w, right=(_L_MARGIN + panel_w) / fig_w, top=1 - _T_MARGIN / fig_h, bottom=_B_MARGIN / fig_h, hspace=_GAP / panel_h
    )
    vmax = float(np.abs(s["phi"]).max())
    pc_v = render_field(ax_v, s["mesh_pos"], s["node_type"], s["speed"], cfg, cmap=CMAP_FIELD)
    pc_e = render_field(ax_e, s["mesh_pos"], s["node_type"], s["phi"], cfg, cmap=CMAP_PHI, vmin=-vmax, vmax=vmax)
    ax_v.set_xticks([])
    ax_v.set_ylabel(r"$y$")
    ax_e.set_ylabel(r"$y$")
    ax_e.set_xlabel(r"$x$")
    for ax, pc, label in [(ax_v, pc_v, CBAR_LABEL_V), (ax_e, pc_e, CBAR_LABEL_PHI)]:
        ax.scatter(*s["x_star"], s=80, marker="*", c="red", edgecolors="black", linewidths=0.6, zorder=10)
        bbox = ax.get_position()
        fig.colorbar(pc, cax=fig.add_axes([(_L_MARGIN + panel_w + _CBAR_PAD) / fig_w, bbox.y0, _CBAR_W / fig_w, bbox.height]), label=label)
    return fig


(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
for name, cfg in DATASETS.items():
    fig = make_stacked_figure(cfg, explained_sample(cfg))
    fig.savefig(RESULTS / "figures" / f"mgn_stacked_{name}_{cfg['sample_idx']}.pdf", bbox_inches="tight", dpi=_SAVE_DPI)
