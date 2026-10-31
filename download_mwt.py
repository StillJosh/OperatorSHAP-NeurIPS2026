"""Download the multiwavelet operator code of Gupta et al. (NeurIPS 2021) into ``operator_shap.models.mwt_gupta``.

The upstream repository has no license, so its files are not part of this repository. This script fetches
``models/models.py`` and ``models/utils.py`` at a fixed commit and makes the ``get_filter`` import package-relative.
Only the MWT backbones (``model=mwt_gupta_1d``) need it.

Source: https://github.com/gaurav71531/mwt-operator

Usage
-----
    python download_mwt.py
"""

import urllib.request
from pathlib import Path

COMMIT = "4b0770c0987a325b5a18775a0bdb63ab6d69435e"
BASE_URL = f"https://raw.githubusercontent.com/gaurav71531/mwt-operator/{COMMIT}/models"
DST = Path(__file__).resolve().parent / "src" / "operator_shap" / "models" / "mwt_gupta"


def main() -> None:
    for name in ("models.py", "utils.py"):
        with urllib.request.urlopen(f"{BASE_URL}/{name}") as response:
            source = response.read().decode()
        (DST / name).write_text(source.replace("from utils import get_filter", "from .utils import get_filter"))
        print(f"wrote {DST / name}")


if __name__ == "__main__":
    main()
