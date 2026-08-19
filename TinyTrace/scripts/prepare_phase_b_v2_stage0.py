from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2 import Stage0Config, prepare_stage0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the TinyTrace Phase B v2 Stage 0 data contract."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--cache-mapping", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "phase_b_activitynet_v2_run",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_stage0(
        train_annotations=args.train_json,
        val_annotations=args.val_json,
        cache_mapping=args.cache_mapping,
        cache_root=args.cache_root,
        output_root=args.output_root,
        config=Stage0Config.from_json(args.config),
        repository_root=REPOSITORY_ROOT,
        overwrite=args.overwrite,
    )
    report = result["dataset_validation"]
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["validation_passed"]:
        raise SystemExit(2)
    if not report["ready_for_training"]:
        print("Stage 0 subset validation passed, but the manifest is not ready for training.")


if __name__ == "__main__":
    main()
