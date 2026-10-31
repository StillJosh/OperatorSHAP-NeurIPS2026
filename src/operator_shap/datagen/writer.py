"""NPZ writer with reproducible train/val/test splits."""

from pathlib import Path

import numpy as np


def write_splits(
    data: dict[str, np.ndarray],
    output_dir: Path,
    pde_name: str,
    val_frac: float = 0.15,
    n_test: int = 30,
    seed: int = 42,
) -> None:
    """Split data into train/val/test and write ``train.npz``, ``val.npz``, ``test.npz`` to ``output_dir / pde_name``.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Arrays to save. The first axis of each array that has more than 1D
        is assumed to be the sample axis.  Scalar/1D arrays (like ``x``,
        ``t``) are stored as-is in every split.
    output_dir : Path
        Root output directory.
    pde_name : str
        Subdirectory name.
    val_frac : float
        Fraction of samples for validation.
    n_test : int
        Number of held-out test samples (paper convention: 30).
    seed : int
        RNG seed for the split.
    """
    out = Path(output_dir) / pde_name
    out.mkdir(parents=True, exist_ok=True)

    # Identify which keys are per-sample (first axis is sample dim)
    sample_keys = []
    meta_keys = []
    n_samples = None
    for k, v in data.items():
        if v.ndim >= 2:
            sample_keys.append(k)
            if n_samples is None:
                n_samples = v.shape[0]
        else:
            meta_keys.append(k)

    # Deterministic split: test first, then val, remainder train.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = min(n_test, max(0, n_samples - 2))
    n_val = max(1, int(n_samples * val_frac))
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        split_data = {k: data[k][idx] for k in sample_keys}
        split_data.update({k: data[k] for k in meta_keys})
        np.savez(out / f"{split_name}.npz", **split_data)

    print(f"  Wrote {pde_name}: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test → {out}")
