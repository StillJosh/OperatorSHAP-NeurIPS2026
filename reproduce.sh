#!/usr/bin/env bash
# Every run behind the paper's tables and figures, in dependency order. Run outputs go to outputs/, tables and figures to results/.
set -euo pipefail
cd "$(dirname "$0")"

GRID_1D=heat_1d,advection_1d,burgers_1d,allen_cahn_1d,diffusion_sorption_1d
GRID_2D=heat_2d,darcy_2d,reaction_diffusion_2d,shallow_water_2d
MID_1D=heat_1d_mid,advection_1d_mid,burgers_1d_mid,allen_cahn_1d_mid,diffusion_sorption_1d_mid
MID_2D=heat_2d_mid,darcy_2d_mid,reaction_diffusion_2d_mid,shallow_water_2d_mid
MGN=mgn_airfoil,mgn_cylinder_flow
MGN_MID=mgn_airfoil_mid,mgn_cylinder_flow_mid
LADDER=128,256,512,1024,2048,4096,8192,16384,32768

# Data
python generate_data.py -m datagen=$GRID_1D,$GRID_2D,heat_3d
python download_mgn.py
python -m operator_shap.datagen.compute_baseline_stats

# Backbones
python train_backbone.py -m +experiment=backbone_sweep
python train_backbone.py -m +experiment=mgn_backbone
python train_backbone.py -m +experiment=mwt_sweep
python train_backbone.py -m +experiment=relu_sweep

# Explainers
for exp in attribution_sweep mgn mwt_sweep relu_sweep holdout_sweep mean_baseline_sweep midpoint_sweep; do
  python run_attribution.py -m +experiment=$exp
done
python run_attribution.py -m +experiment=mgn dataset=$MGN_MID

# Classical references: RegressionMSR at 3e5 coalitions (reference_sweep), RegressionMSR and KernelSHAP at max(2n, 1024) (cheap_reference_sweep)
python run_attribution.py -m +experiment=budget_runs dataset=burgers_1d attribution.n_players=16 attribution.approximator=exact
python run_attribution.py -m +experiment=cheap_reference_sweep dataset=burgers_1d attribution.n_players=16
for sweep in reference_sweep cheap_reference_sweep; do
  python run_attribution.py -m +experiment=$sweep dataset=$GRID_1D attribution.n_players=32,64,128,256,512,1024
  python run_attribution.py -m +experiment=$sweep dataset=$GRID_2D attribution.n_players=64,256,1024
  python run_attribution.py -m +experiment=$sweep dataset=heat_3d,heat_3d_mid attribution.n_players=512
  python run_attribution.py -m +experiment=$sweep dataset=$MID_1D,$MID_2D attribution.n_players=16,64,256,1024
  python run_attribution.py -m +experiment=$sweep dataset=darcy_2d_mean,diffusion_sorption_1d_mean attribution.n_players=64,1024
  python run_attribution.py -m +experiment=$sweep dataset=burgers_1d model=fno1d_relu attribution.n_players=64,256
done
python run_attribution.py -m +experiment=mgn attribution=shap_mgn attribution.n_players=64 attribution.approximator=proxyshap attribution.budget=300000 dataset=mgn_cylinder_flow,mgn_cylinder_flow_mid
python run_attribution.py -m +experiment=mgn attribution=shap_mgn attribution.n_players=64 attribution.approximator=proxyshap attribution.budget=50000 dataset=mgn_airfoil,mgn_airfoil_mid
python run_attribution.py -m +experiment=mgn attribution=shap_mgn attribution.n_players=64 attribution.approximator=proxyshap,kernelshap attribution.budget=1024 dataset=$MGN,$MGN_MID

# Budget ladders of the budget-curve, amortization, and matched-cost figures
python run_attribution.py -m +experiment=budget_runs dataset=burgers_1d,heat_2d,reaction_diffusion_2d,shallow_water_2d attribution.n_players=256 attribution.approximator=proxyshap,kernelshap attribution.budget=$LADDER
python run_attribution.py -m +experiment=budget_runs dataset=burgers_1d attribution.n_players=16,512,1024 attribution.approximator=proxyshap attribution.budget=64,128,192,256,384,512,768,1024,2048,4096,8192
python run_attribution.py -m +experiment=budget_runs dataset=burgers_1d attribution.n_players=16 attribution.approximator=kernelshap attribution.budget=64,128,192,256,384,512,768,1024,2048,4096,8192
python run_attribution.py -m +experiment=budget_runs dataset=burgers_1d attribution.n_players=512 attribution.approximator=kernelshap attribution.budget=768,1024,2048,4096,8192
python run_attribution.py -m +experiment=budget_runs dataset=burgers_1d attribution.n_players=1024 attribution.approximator=kernelshap attribution.budget=2048,4096,8192

# Metrics and timings
python run_metrics.py -m dataset=burgers_1d attribution.n_players=16 dataset.reference_segment=exact
python run_metrics.py -m dataset=$GRID_1D attribution.n_players=32,64,128,256,512,1024
python run_metrics.py -m dataset=$GRID_2D attribution.n_players=64,256,1024
python run_metrics.py -m dataset=heat_3d,heat_3d_mid attribution.n_players=512
python run_metrics.py -m dataset=$MID_1D,$MID_2D attribution.n_players=16,64,256,1024
python run_metrics.py -m dataset=burgers_1d_ho256,heat_2d_ho256 attribution.n_players=256
python run_metrics.py -m dataset=burgers_1d_ho1024,heat_2d_ho1024 attribution.n_players=1024
python run_metrics.py -m dataset=darcy_2d_mean,diffusion_sorption_1d_mean attribution.n_players=64,1024
python run_metrics.py -m dataset=burgers_1d model=fno1d_relu attribution.n_players=64,256
python run_metrics.py -m dataset=$MGN,$MGN_MID attribution=eval_mgn attribution.n_players=64 metrics.ig_batch_size=8 attribution.max_coalition_batch=8
python run_metrics.py -m +experiment=ig_budget_multiplier
python run_timing.py -m +experiment=ig_budget_multiplier device=cpu attribution.n_samples=10 timing.n_warmup=1

# Tables and figures
python make_tables.py
for script in figures/*.py; do python "$script"; done
