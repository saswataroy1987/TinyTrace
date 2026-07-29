from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


MOBILECLIP_S0_URL = (
    "https://docs-assets.developer.apple.com/ml-research/datasets/"
    "mobileclip/mobileclip_s0.pt"
)
MOBILECLIP_S0_SHA256 = "809b408eff74f8058843e86a1f92967097d42ba782450e85b8f4867b7f0ca0b7"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download and verify Apple MobileCLIP-S0.")
    parser.add_argument(
        "--destination",
        type=Path,
        default=project_root / "checkpoints" / "mobileclip_s0.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    if destination.is_file():
        digest = sha256_file(destination)
        if digest == MOBILECLIP_S0_SHA256:
            print(f"MobileCLIP-S0 already verified: {destination}")
            return
        raise ValueError(
            f"Existing checkpoint has SHA-256 {digest}, expected {MOBILECLIP_S0_SHA256}."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    try:
        urllib.request.urlretrieve(MOBILECLIP_S0_URL, temporary)
        digest = sha256_file(temporary)
        if digest != MOBILECLIP_S0_SHA256:
            raise ValueError(
                f"Downloaded checkpoint has SHA-256 {digest}, expected {MOBILECLIP_S0_SHA256}."
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Downloaded and verified MobileCLIP-S0: {destination}")


if __name__ == "__main__":
    main()
