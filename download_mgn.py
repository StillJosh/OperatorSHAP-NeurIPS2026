"""Download DeepMind's MeshGraphNets ``airfoil`` and ``cylinder_flow`` datasets and convert them to per-trajectory npz files.

For each dataset the tfrecords and ``meta.json`` go to ``<data_root>/raw/<dataset>/``, the converted trajectories to
``<data_root>/mgn_<dataset>/{train,val,test}/``; the train-split baseline statistics are computed last. Finished
downloads and converted trajectories are skipped on a rerun. The conversion needs ``tensorflow``.

Source: https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets

Usage
-----
    python download_mgn.py
    python download_mgn.py --data-root /path/to/data
"""

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

from operator_shap.datagen.mgn_convert import convert

BASE_URL = "https://storage.googleapis.com/dm-meshgraphnets"
DATASETS = ("airfoil", "cylinder_flow")
FILES = ("meta.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord")


def download(url: str, dst: Path) -> None:
    """Stream ``url`` to ``dst``; skip it when ``dst`` already has the remote size."""
    with urllib.request.urlopen(url) as response:
        size = int(response.headers["Content-Length"])
        if dst.exists() and dst.stat().st_size == size:
            print(f"skipping {dst}: already downloaded")
            return
        print(f"downloading {url} ({size / 1e9:.1f} GB) -> {dst}")
        part = dst.with_name(dst.name + ".part")
        with part.open("wb") as f:
            shutil.copyfileobj(response, f, length=1 << 24)
    part.rename(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert the MeshGraphNets airfoil and cylinder_flow datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="data root of the repository configs (default: data)")
    args = parser.parse_args()

    for name in DATASETS:
        raw_dir = args.data_root / "raw" / name
        raw_dir.mkdir(parents=True, exist_ok=True)
        for file in FILES:
            download(f"{BASE_URL}/{name}/{file}", raw_dir / file)
        convert(raw_dir, args.data_root / f"mgn_{name}")

    datasets = ",".join(f"mgn_{name}" for name in DATASETS)
    subprocess.run(
        [sys.executable, "-m", "operator_shap.datagen.compute_baseline_stats", f"data_root={args.data_root}", f"datasets=[{datasets}]"],
        check=True,
    )


if __name__ == "__main__":
    main()
