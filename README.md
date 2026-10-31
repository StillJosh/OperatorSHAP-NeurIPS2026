# OperatorSHAP

Supplementary code for the NeurIPS 2026 paper

> **OperatorSHAP: Fast and Accurate Shapley Value Estimation for Neural Operators**
> Joshua Stiller, Santo M. A. R. Thies, Felix Czaja, Eyke Hüllermeier

The code regenerates every table and figure of the paper from scratch: data, backbones, explainers, reference Shapley
values, metrics, timings, tables, and figures.

## Setup

```bash
uv sync                  # Linux with CUDA 12.6
uv sync --extra mgn      # adds tensorflow, needed to convert the MeshGraphNets data
source .venv/bin/activate
python download_mwt.py   # code of the MWT backbone, see Third-party code
```

Python 3.12. `torch-scatter` (GINO on the meshes) and `torch-harmonics` (LocalNO on the 2D grids) are Linux-only; on
macOS `uv sync` installs the rest, which runs the 1D and 3D cases. The figure scripts render text with LaTeX, so they
need `latex` and `dvipng` on the `PATH`.

## Reproduction

```bash
./reproduce.sh
```

`reproduce.sh` lists every run in dependency order: data, backbones, explainers, classical references, metrics and
timings, tables and figures. Run outputs go to `outputs/{dataset}/{model}/{run}/`, tables and figures to `results/`.

- The backbones, explainers, and references need a GPU. A reference at $3\cdot10^5$ coalitions takes hours per dataset and
  player count; the MGN Airfoil reference at $5\cdot10^4$ coalitions takes about 30 h on one GPU.
- Shallow Water 2D needs about 100 GB of host memory to load. The MeshGraphNets download is 77 GB.
- Data generation is bitwise reproducible from `seed: 42`. GPU training is not bitwise deterministic, so rerun numbers are
  close to, not identical with, the paper's.
- RegressionMSR is shapiq's `ProxySHAP` (`attribution.approximator=proxyshap`), labelled `RMSR` in the metrics CSVs.

| Paper output | Script |
|---|---|
| `all_pde_metrics_{short,full_top,full_bot}.tex`, `holdout.tex`, `mean_baseline.tex`, `relu.tex`, `midpoint.tex`, `numbers.tex` | `make_tables.py` |
| `Comparison_Shap_Operator.pdf` | `figures/method_comparison.py` |
| `amortization.pdf` | `figures/amortization.py` |
| `analytic_attributions.pdf`, `numbers_analytic.tex` | `figures/analytic_attributions.py` |
| `baseline_ablation.pdf`, `numbers_baseline.tex` | `figures/baseline_ablation.py` |
| `burgers1d_budget_curves.pdf` | `figures/burgers1d_budget_curves.py` |
| `heat2d_triptych_heatmap.pdf` | `figures/heat2d_triptych.py` |
| `ig_budget_multiplier.pdf`, `numbers_matched_cost.tex` | `figures/ig_budget_multiplier.py` |
| `mgn_stacked_airfoil_1.pdf`, `mgn_stacked_cylinder_flow_1.pdf` | `figures/mgn_samples.py` |
| `pou_example.pdf` | `figures/pou_example.py` |
| `backbone_error.tex` | `figures/backbone_error.py` |
| `ssim.tex` | `figures/ssim.py` |

## Single runs

Every entry point is a Hydra script; `configs/` holds one config per dataset, model, and method.

```bash
python generate_data.py datagen=burgers_1d
python train_backbone.py dataset=burgers_1d
python run_attribution.py dataset=burgers_1d                                  # OperatorSHAP explainer
python run_attribution.py dataset=burgers_1d +experiment=budget_runs attribution.n_players=64 attribution.budget=300000
python run_attribution.py dataset=burgers_1d +experiment=cheap_reference_sweep attribution.n_players=64 attribution.approximator=proxyshap
python run_attribution.py dataset=burgers_1d +experiment=cheap_reference_sweep attribution.n_players=64 attribution.approximator=kernelshap
python run_metrics.py dataset=burgers_1d attribution.n_players=64
```

`mode=debug` runs the training, attribution, and metrics scripts on the CPU with 12 samples and a small model and
writes to `outputs_debug/`:

```bash
python generate_data.py datagen=heat_1d n_samples=150
python train_backbone.py dataset=heat_1d mode=debug
python run_attribution.py dataset=heat_1d mode=debug
python run_attribution.py dataset=heat_1d mode=debug +experiment=budget_runs attribution.n_players=8 attribution.budget=300000
python run_attribution.py dataset=heat_1d mode=debug +experiment=cheap_reference_sweep attribution.n_players=8 attribution.approximator=proxyshap
python run_attribution.py dataset=heat_1d mode=debug +experiment=cheap_reference_sweep attribution.n_players=8 attribution.approximator=kernelshap
python run_metrics.py dataset=heat_1d mode=debug attribution.n_players=8
```

## Third-party code

- `src/operator_shap/models/gino.py`: GINO from neuraloperator 2.0.0 (MIT License, Copyright (c) 2023 NeuralOperator
  developers), with the latent FNO stack kept in float32 under autocast.
- MWT: the multiwavelet operator of Gupta et al. (NeurIPS 2021), github.com/gaurav71531/mwt-operator. That repository has
  no license, so its code is not included; `download_mwt.py` fetches it at a fixed commit.
- MeshGraphNets data: DeepMind, github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets.

## License

MIT, see `LICENSE`. The third-party code above keeps the terms of its source.

## Citation

```bibtex
@inproceedings{stiller2026operatorshap,
  title     = {{OperatorSHAP}: Fast and Accurate Shapley Value Estimation for Neural Operators},
  author    = {Stiller, Joshua and Thies, Santo M. A. R. and Czaja, Felix and H{\"u}llermeier, Eyke},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
