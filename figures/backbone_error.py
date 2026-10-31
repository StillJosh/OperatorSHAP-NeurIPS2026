"""Backbone error per grid resolution.

Relative L2 test error ||u_hat - u||_2 / ||u||_2 of every grid backbone, per sample over all channels, median over the 30 test
samples, at every per-axis resolution of its explainer ladder (``dataset.explainer_resolutions``). Inputs and targets are strided
from the native grid by the dataset. The MGN backbones are not evaluated, so their rows are ``--``.
Writes ``results/tables/backbone_error.tex``.
"""

# %%
import hydra
import pandas as pd
import torch

from operator_shap.backbone import load_frozen
from operator_shap.dataset import build_eval_loader
from operator_shap.results import RESULTS, compose_config

DEVICE = "cpu"
TABLE = RESULTS / "tables" / "backbone_error.tex"

# (row label, dataset config, model override or None)
ROWS = [
    ("Heat 1D", "heat_1d", None),
    ("Burgers 1D", "burgers_1d", None),
    ("Burgers 1D (ReLU)", "burgers_1d", "fno1d_relu"),
    ("Allen--Cahn 1D", "allen_cahn_1d", None),
    ("Diff.--Sorp. 1D", "diffusion_sorption_1d", None),
    ("Advection 1D", "advection_1d", None),
    ("Heat 2D", "heat_2d", None),
    ("Darcy 2D", "darcy_2d", None),
    ("Reac.--Diff. 2D", "reaction_diffusion_2d", None),
    ("Shallow Water 2D", "shallow_water_2d", None),
    ("Heat 3D", "heat_3d", None),
]
MGN_ROWS = ["MGN Cylinder", "MGN Airfoil"]

# %%
records = []
for label, dataset, model in ROWS:
    cfg = compose_config("run_metrics", [f"dataset={dataset}", f"device={DEVICE}"] + ([f"model={model}"] if model else []))
    backbone, _ = load_frozen(cfg.backbone_checkpoint, DEVICE)
    test_ds = hydra.utils.instantiate(cfg.dataset.class_config, split="test", max_samples=None)
    for res in cfg.dataset.explainer_resolutions:
        err = []
        for batch in build_eval_loader(test_ds, len(test_ds), int(res)):
            with torch.no_grad():
                pred = backbone({"u0": batch["u0"].to(DEVICE)}).cpu()
            err.append(float((pred - batch["y_true"]).norm() / batch["y_true"].norm()))
        err = torch.tensor(err)
        records.append({"label": label, "n_1d": int(res), "rel_l2_med": float(err.median()),
                        "rel_l2_q25": float(err.quantile(0.25)), "rel_l2_q75": float(err.quantile(0.75)), "n_test": len(err)})
errors = pd.DataFrame(records)

# %%
cols = sorted(errors.n_1d.unique())
med = errors.pivot_table(index="label", columns="n_1d", values="rel_l2_med")
lines = [
    r"\begin{tabular}{l" + "c" * len(cols) + "}",
    r"\toprule",
    r"PDE $\backslash\ n_{\mathrm{1D}}$ & " + " & ".join(str(c) for c in cols) + r" \\",
    r"\midrule",
]
for label, *_ in ROWS:
    cells = [f"{med.loc[label, c]:.3f}" if pd.notna(med.loc[label, c]) else "--" for c in cols]
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
for label in MGN_ROWS:
    lines.append(f"{label} & " + " & ".join(["--"] * len(cols)) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
TABLE.parent.mkdir(parents=True, exist_ok=True)
TABLE.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
