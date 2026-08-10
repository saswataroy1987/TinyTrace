from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an upload-ready Kaggle dataset bundle for TinyTrace Phase A. "
            "It can package either a fully prepared dense dataset or a raw-source "
            "bundle, plus MobileCLIP and an optional warm-start checkpoint."
        )
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mobileclip-checkpoint", type=Path, required=True)
    parser.add_argument("--bootstrap-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--prepared-train-json",
        type=Path,
        default=None,
        help="Final dense train JSON (tinytrace_phase_a_v4_train.json).",
    )
    parser.add_argument(
        "--prepared-val-json",
        type=Path,
        default=None,
        help="Final dense val JSON (tinytrace_phase_a_v4_val.json).",
    )
    parser.add_argument("--prepared-manifest-json", type=Path, default=None)
    parser.add_argument("--prepared-exclusions-json", type=Path, default=None)
    parser.add_argument(
        "--prepared-videos-root",
        type=Path,
        default=None,
        help="Directory containing train/ and val/ subdirectories for the final dense dataset.",
    )
    parser.add_argument("--raw-root", type=Path, default=None, help="Existing final_qvhighlights_tinytrace root.")
    parser.add_argument("--vendor-mobileclip", type=Path, default=None)
    return parser.parse_args()


def _assert_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def _assert_dir(path: Path | None, label: str) -> Path:
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    mobileclip = _assert_file(args.mobileclip_checkpoint, "MobileCLIP checkpoint")
    _copy_file(mobileclip, output_root / "checkpoints" / "mobileclip_s0.pt")

    if args.bootstrap_checkpoint is not None:
        bootstrap = _assert_file(args.bootstrap_checkpoint, "Bootstrap checkpoint")
        _copy_file(bootstrap, output_root / "bootstrap" / "phase_a_v3_best_primary_metric.pt")

    if args.vendor_mobileclip is not None:
        vendor = _assert_file(args.vendor_mobileclip, "Vendor MobileCLIP package")
        _copy_file(vendor, output_root / "vendor" / vendor.name)

    packaged_modes: list[str] = []

    prepared_inputs = [
        args.prepared_train_json,
        args.prepared_val_json,
        args.prepared_manifest_json,
        args.prepared_exclusions_json,
        args.prepared_videos_root,
    ]
    if any(value is not None for value in prepared_inputs):
        prepared_train = _assert_file(args.prepared_train_json, "Prepared train JSON")
        prepared_val = _assert_file(args.prepared_val_json, "Prepared val JSON")
        prepared_manifest = _assert_file(args.prepared_manifest_json, "Prepared manifest JSON")
        prepared_exclusions = _assert_file(args.prepared_exclusions_json, "Prepared exclusions JSON")
        prepared_videos_root = _assert_dir(args.prepared_videos_root, "Prepared videos root")
        _assert_dir(prepared_videos_root / "train", "Prepared train videos")
        _assert_dir(prepared_videos_root / "val", "Prepared val videos")

        phase_root = output_root / "final_phase_a_v4"
        _copy_file(prepared_train, phase_root / "annotations" / "tinytrace_phase_a_v4_train.json")
        _copy_file(prepared_val, phase_root / "annotations" / "tinytrace_phase_a_v4_val.json")
        _copy_file(prepared_manifest, phase_root / "annotations" / "phase_a_v4_manifest.json")
        _copy_file(prepared_exclusions, phase_root / "annotations" / "phase_a_v4_exclusions.json")
        _copy_tree(prepared_videos_root / "train", phase_root / "videos" / "train")
        _copy_tree(prepared_videos_root / "val", phase_root / "videos" / "val")
        packaged_modes.append("prepared_final_phase_a_v4")

    if args.raw_root is not None:
        raw_root = _assert_dir(args.raw_root, "Raw final_qvhighlights_tinytrace root")
        _assert_file(raw_root / "annotations" / "qvh_raw_valid.json", "Raw qvh_raw_valid.json")
        _assert_file(raw_root / "annotations" / "tinytrace_train.json", "Raw tinytrace_train.json")
        _assert_file(raw_root / "annotations" / "tinytrace_val.json", "Raw tinytrace_val.json")
        _assert_dir(raw_root / "videos" / "train", "Raw train videos")
        _assert_dir(raw_root / "videos" / "val", "Raw val videos")
        _copy_tree(raw_root, output_root / "final_qvhighlights_tinytrace")
        packaged_modes.append("raw_source_bundle")

    if not packaged_modes:
        raise ValueError("Provide either prepared final-phase inputs, raw-root, or both.")

    _write_manifest(
        output_root / "bundle_manifest.json",
        {
            "packaged_modes": packaged_modes,
            "mobileclip_checkpoint": "checkpoints/mobileclip_s0.pt",
            "bootstrap_checkpoint": (
                "bootstrap/phase_a_v3_best_primary_metric.pt"
                if args.bootstrap_checkpoint is not None
                else None
            ),
        },
    )
    print(f"Created TinyTrace Kaggle bundle at {output_root}")


if __name__ == "__main__":
    main()
