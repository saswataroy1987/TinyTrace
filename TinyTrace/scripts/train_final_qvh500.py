from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "configs" / "final_train_qvh500.json"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Final TinyTrace training is GPU-only, but CUDA is not available.")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_training_profile.py"),
        "--profile",
        str(PROFILE_PATH),
    ]
    print("Running final QVHighlights TinyTrace training with profile:")
    print(PROFILE_PATH)
    print()
    print(json.dumps(json.loads(PROFILE_PATH.read_text(encoding="utf-8")), indent=2))
    print()
    print("Command:")
    print(" ".join(command))
    print()
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
