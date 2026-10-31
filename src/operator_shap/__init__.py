"""operator_shap: amortised Aumann-Shapley attribution for neural operators.

- backbone:   Model construction and checkpoint loading.
- trainer:    Training loop with early stopping.
- domain:     The coalition game of a batch for grids and meshes.
- utils:      Run setup, baselines, coalition sampling, query encoding, hard normalisation.
- results:    Run output paths and loaders for the tables and figure scripts.
- dataset:    Grid and mesh datasets, resolution sampler, loader factory.
- games:      shapiq game over a Domain.
- methods:    FastSHAP, classical SHAP, Integrated Gradients.
- metrics:    Agreement and faithfulness.
- models:     Grid wrapper, GINO, MWT.
- figures:    Paper figure style and plot atoms.
- datagen:    PDE solvers, initial conditions, writers, MGN converter.
"""
