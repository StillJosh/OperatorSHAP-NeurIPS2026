"""Convert DeepMind MeshGraphNets tfrecord splits to per-trajectory npz files.

Each tfrecord (``train.tfrecord``, ``valid.tfrecord``, ``test.tfrecord``)
is processed independently and unpacked into its own output folder::

    out_dir/
      train/train_0000.npz, train_0001.npz, …      (from train.tfrecord)
      val/val_0000.npz,     val_0001.npz,   …      (from valid.tfrecord)
      test/test_0000.npz,   test_0001.npz,  …      (from test.tfrecord)

Tfrecord splits are named ``train`` / ``valid`` / ``test``; the on-disk
folders use ``train`` / ``val`` / ``test`` so they match the dataset
class's ``split=`` argument directly. Filenames carry the same prefix
so each file is uniquely identifiable when copied/inspected outside its
folder. Numbering is 0-based per split.

Each npz holds:

  mesh_pos   (n_nodes, 2)       float32   — static; first-timestep slice
  node_type  (n_nodes,)         int32     — static; first-timestep slice, squeezed
  cells      (n_cells, 3)       int32     — static; first-timestep slice
  velocity   (n_timesteps, n_nodes, 2)    float32   — dynamic (cylinder_flow / airfoil)
  density    (n_timesteps, n_nodes, 1)    float32   — dynamic (airfoil only)
  pressure   (n_timesteps, n_nodes, 1)    float32   — dynamic (airfoil only)

Static fields are stored **once** (the tfrecord redundantly stores them
T times even though they don't change), which is why the npz total is
much smaller than the source tfrecord.

Requires ``tensorflow``. ``download_mgn.py`` at the repository root downloads the
tfrecords and calls :func:`convert`; to convert by hand::

  python -m operator_shap.datagen.mgn_convert \\
      --tfrecord-dir data/raw/airfoil \\
      --out-dir data/mgn_airfoil
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# tfrecord split name -> on-disk folder and filename prefix matching the dataset ``split=`` argument.
_FOLDER = {"train": "train", "valid": "val", "test": "test"}


def _parse_proto(proto, meta):
    """Decode one trajectory record into a dict of tensors."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        out[key] = data
    return out


def convert_split(tfrecord_dir: Path, out_dir: Path, split: str) -> int:
    """Extract every trajectory in ``{tfrecord_dir}/{split}.tfrecord``
    into ``{out_dir}/{folder}/{folder}_NNNN.npz`` with 0-based per-split
    indexing. Idempotent: skips outputs that already exist with non-zero
    size. Returns the number of trajectories written (or already on disk).
    """
    tfrecord = tfrecord_dir / f"{split}.tfrecord"
    if not tfrecord.exists():
        raise FileNotFoundError(tfrecord)
    with open(tfrecord_dir / "meta.json") as f:
        meta = json.load(f)

    folder_name = _FOLDER[split]
    folder = out_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    ds = tf.data.TFRecordDataset(str(tfrecord))
    ds = ds.map(lambda p: _parse_proto(p, meta))

    n = 0
    for i, traj in enumerate(ds):
        out_npz = folder / f"{folder_name}_{i:04d}.npz"
        if out_npz.exists() and out_npz.stat().st_size > 0:
            n += 1
            if (i + 1) % 100 == 0:
                print(f"  {split} {i+1:>4d}  (skip — already on disk)", flush=True)
            continue
        d: dict[str, np.ndarray] = {}
        for key, field in meta["features"].items():
            arr = traj[key].numpy()
            if field["type"] == "static":
                arr = arr[0]
                if key == "node_type":
                    arr = arr.squeeze(-1)
            d[key] = arr
        np.savez(out_npz, **d)
        n += 1
        if (i + 1) % 50 == 0 or i < 3:
            print(f"  {split} {i+1:>4d} -> {folder_name}/{out_npz.name}", flush=True)
    return n


def convert(tfrecord_dir: Path, out_dir: Path) -> None:
    """Convert every ``{train,valid,test}.tfrecord`` present in ``tfrecord_dir`` into ``out_dir/{train,val,test}/``."""
    for s in ("train", "valid", "test"):
        if not (tfrecord_dir / f"{s}.tfrecord").exists():
            print(f"skipping {s}: tfrecord absent")
            continue
        print(f"=== converting {s} ===")
        n = convert_split(tfrecord_dir, out_dir, s)
        print(f"  -> {n} trajectories in {_FOLDER[s]}/")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord-dir", type=Path, required=True,
                   help="dir containing meta.json + {train,valid,test}.tfrecord")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="dir to write {train,val,test}/*.npz into")
    args = p.parse_args()
    convert(args.tfrecord_dir, args.out_dir)


if __name__ == "__main__":
    main()
