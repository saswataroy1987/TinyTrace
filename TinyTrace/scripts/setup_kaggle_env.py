from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOBILECLIP_PIP_SPEC = "mobileclip @ git+https://github.com/apple/ml-mobileclip.git@aecfb5453d022e9deff12f81a150ea8f35194baa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install the minimum TinyTrace Kaggle runtime dependencies. If the "
            "uploaded dataset bundle contains a local MobileCLIP wheel or source "
            "archive under vendor/, that offline package is preferred."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional Kaggle dataset root containing vendor/mobileclip*.whl or source archives.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def _find_local_mobileclip_package(dataset_root: Path | None) -> Path | None:
    if dataset_root is None:
        return None
    vendor = dataset_root / "vendor"
    if not vendor.is_dir():
        return None
    patterns = (
        "mobileclip*.whl",
        "mobileclip*.tar.gz",
        "mobileclip*.zip",
        "ml_mobileclip*.whl",
        "ml_mobileclip*.tar.gz",
        "ml_mobileclip*.zip",
    )
    for pattern in patterns:
        matches = sorted(vendor.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> None:
    args = parse_args()
    python = sys.executable
    _run([python, "-m", "pip", "install", "numpy", "torch", "torchvision", "imageio-ffmpeg>=0.6"])

    local_mobileclip = _find_local_mobileclip_package(
        args.dataset_root.resolve() if args.dataset_root is not None else None
    )
    if local_mobileclip is not None:
        _run([python, "-m", "pip", "install", str(local_mobileclip)])
        return

    _run([python, "-m", "pip", "install", MOBILECLIP_PIP_SPEC])


if __name__ == "__main__":
    main()
