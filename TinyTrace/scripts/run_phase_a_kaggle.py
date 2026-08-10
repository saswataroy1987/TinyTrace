from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_phase_a_qvhighlights import prepare_phase_a_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kaggle-ready Phase A launcher. It supports either a fully prepared "
            "final dense dataset bundle or a raw cleaned source bundle that "
            "must be rebuilt into dense Phase-A annotations first."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root of the uploaded Kaggle dataset bundle.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=PROJECT_ROOT / ".kaggle_phase_a_v4",
        help="Writable working directory for generated annotations, caches, and outputs.",
    )
    parser.add_argument("--device", default="cuda", help="Training device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--use-bootstrap-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Initialize the new Phase A run from the prior best v3 checkpoint when "
            "the uploaded dataset bundle provides it."
        ),
    )
    parser.add_argument(
        "--bootstrap-checkpoint",
        type=Path,
        default=None,
        help="Optional explicit bootstrap checkpoint path inside the uploaded dataset bundle.",
    )
    parser.add_argument(
        "--skip-feature-cache",
        action="store_true",
        help="Skip MobileCLIP feature precomputation when a complete cache already exists.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Stop after preparing configs/annotations for training.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Stop after dataset inspection/audit.",
    )
    return parser.parse_args()


def _assert_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path.resolve()


def _assert_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path.resolve()


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, cwd=WORKSPACE_ROOT)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _ensure_work_videos_link(work_root: Path, dataset_videos_root: Path) -> Path:
    """Expose uploaded videos under work_root/videos for annotation portability.

    The dense train/validation JSONs use repository-relative paths such as
    ``videos/train/...``.  Training reads copied/generated annotations from the
    writable work directory, so we must make those same relative paths resolve
    there as well.
    """

    target = dataset_videos_root.resolve()
    link_path = work_root / "videos"
    if link_path.is_symlink():
        if link_path.resolve() == target:
            return link_path
        link_path.unlink()
    elif link_path.exists():
        if link_path.resolve() == target:
            return link_path
        raise FileExistsError(
            f"Cannot create Kaggle work videos link because {link_path} already exists "
            f"and does not point to {target}."
        )
    os.symlink(target, link_path, target_is_directory=True)
    return link_path


def _find_bootstrap_checkpoint(dataset_root: Path, explicit: Path | None) -> Path | None:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            dataset_root / "bootstrap" / "phase_a_v3_best_primary_metric.pt",
            dataset_root / "bootstrap" / "best-primary-metric.pt",
            dataset_root / "outputs-qvh-phase-a-v3-full" / "checkpoints" / "best-primary-metric.pt",
            dataset_root / "outputs-qvh-phase-a-v3-full" / "checkpoints" / "best.pt",
            dataset_root / "outputs-qvh-phase-a-v3-full" / "tinytrace.pt",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _prepared_dataset_paths(dataset_root: Path) -> dict[str, Path] | None:
    prepared_root = dataset_root / "final_phase_a_v4"
    train_json = prepared_root / "annotations" / "tinytrace_phase_a_v4_train.json"
    val_json = prepared_root / "annotations" / "tinytrace_phase_a_v4_val.json"
    manifest_json = prepared_root / "annotations" / "phase_a_v4_manifest.json"
    exclusions_json = prepared_root / "annotations" / "phase_a_v4_exclusions.json"
    videos_root = prepared_root / "videos"
    if all(path.is_file() for path in (train_json, val_json, manifest_json, exclusions_json)) and videos_root.is_dir():
        return {
            "prepared_root": prepared_root.resolve(),
            "train_json": train_json.resolve(),
            "val_json": val_json.resolve(),
            "manifest_json": manifest_json.resolve(),
            "exclusions_json": exclusions_json.resolve(),
            "videos_root": videos_root.resolve(),
        }
    return None


def _raw_dataset_paths(dataset_root: Path) -> dict[str, Path] | None:
    raw_root = dataset_root / "final_qvhighlights_tinytrace"
    annotations = raw_root / "annotations"
    videos = raw_root / "videos"
    raw_json = annotations / "qvh_raw_valid.json"
    train_json = annotations / "tinytrace_train.json"
    val_json = annotations / "tinytrace_val.json"
    if all(path.is_file() for path in (raw_json, train_json, val_json)) and videos.is_dir():
        return {
            "raw_root": raw_root.resolve(),
            "annotations": annotations.resolve(),
            "videos_root": videos.resolve(),
            "raw_json": raw_json.resolve(),
            "train_json": train_json.resolve(),
            "val_json": val_json.resolve(),
        }
    return None


def _run_audit(train_json: Path, val_json: Path, raw_json: Path | None, show_samples: int = 4) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "audit_phase_a_dataset.py"),
        "--train-json",
        str(train_json),
        "--val-json",
        str(val_json),
        "--show-samples",
        str(show_samples),
    ]
    if raw_json is not None:
        command.extend(["--raw-json", str(raw_json)])
    _run(command)


def _copy_prepared_annotations(prepared: dict[str, Path], work_root: Path) -> tuple[Path, Path, Path, Path]:
    generated_annotations = work_root / "annotations"
    generated_annotations.mkdir(parents=True, exist_ok=True)
    output_train_json = generated_annotations / "tinytrace_phase_a_v4_train.json"
    output_val_json = generated_annotations / "tinytrace_phase_a_v4_val.json"
    exclusions_json = generated_annotations / "phase_a_v4_exclusions.json"
    manifest_json = generated_annotations / "phase_a_v4_manifest.json"
    shutil.copy2(prepared["train_json"], output_train_json)
    shutil.copy2(prepared["val_json"], output_val_json)
    shutil.copy2(prepared["exclusions_json"], exclusions_json)
    shutil.copy2(prepared["manifest_json"], manifest_json)
    return output_train_json, output_val_json, exclusions_json, manifest_json


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    work_root = args.work_root.resolve()

    mobileclip_checkpoint = _assert_file(
        dataset_root / "checkpoints" / "mobileclip_s0.pt",
        "MobileCLIP checkpoint",
    )

    bootstrap_checkpoint = None
    if args.use_bootstrap_checkpoint:
        bootstrap_checkpoint = _find_bootstrap_checkpoint(dataset_root, args.bootstrap_checkpoint)
        if bootstrap_checkpoint is None:
            raise FileNotFoundError(
                "Bootstrap checkpoint requested, but none was found. Upload one of:\n"
                "- bootstrap/phase_a_v3_best_primary_metric.pt\n"
                "- outputs-qvh-phase-a-v3-full/checkpoints/best-primary-metric.pt\n"
                "- outputs-qvh-phase-a-v3-full/tinytrace.pt"
            )

    work_root.mkdir(parents=True, exist_ok=True)

    prepared = _prepared_dataset_paths(dataset_root)
    raw = _raw_dataset_paths(dataset_root)

    if prepared is None and raw is None:
        raise FileNotFoundError(
            "Dataset root does not contain a supported TinyTrace Phase A dataset.\n"
            "Provide either:\n"
            "- final_phase_a_v4/annotations/tinytrace_phase_a_v4_{train,val}.json and videos/\n"
            "or:\n"
            "- final_qvhighlights_tinytrace/annotations/{qvh_raw_valid,tinytrace_train,tinytrace_val}.json and videos/"
        )

    if prepared is not None:
        _ensure_work_videos_link(work_root, prepared["videos_root"])
        output_train_json, output_val_json, exclusions_json, manifest_json = _copy_prepared_annotations(
            prepared, work_root
        )
        raw_json_for_audit = (dataset_root / "optional_source_audit" / "final_qvhighlights_tinytrace" / "annotations" / "qvh_raw_valid.json")
        _run_audit(
            output_train_json,
            output_val_json,
            raw_json_for_audit if raw_json_for_audit.is_file() else None,
        )
    else:
        _ensure_work_videos_link(work_root, raw["videos_root"])
        generated_annotations = work_root / "annotations"
        output_train_json = generated_annotations / "tinytrace_phase_a_v4_train.json"
        output_val_json = generated_annotations / "tinytrace_phase_a_v4_val.json"
        exclusions_json = generated_annotations / "phase_a_v4_exclusions.json"
        manifest_json = generated_annotations / "phase_a_v4_manifest.json"

        for stale in (output_train_json, output_val_json, exclusions_json, manifest_json):
            stale.unlink(missing_ok=True)

        prepare_phase_a_dataset(
            raw_json=raw["raw_json"],
            train_json=raw["train_json"],
            val_json=raw["val_json"],
            output_train_json=output_train_json,
            output_val_json=output_val_json,
            exclusions_json=exclusions_json,
            manifest_json=manifest_json,
        )
        _run_audit(output_train_json, output_val_json, raw["raw_json"])

    model_config_path = PROJECT_ROOT / "configs" / "tinytrace_qvhighlights_phase_a_v4.json"
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    model_config["mobileclip_checkpoint"] = str(mobileclip_checkpoint)
    generated_model_config = work_root / "configs" / "tinytrace_qvhighlights_phase_a_v4_kaggle.json"
    _write_json(generated_model_config, model_config)

    profile = {
        "train_script": str(PROJECT_ROOT / "scripts" / "train_tinytrace.py"),
        "model_config": str(generated_model_config),
        "train_dataset_json": str(output_train_json),
        "val_dataset_json": str(output_val_json),
        "init_checkpoint": str(bootstrap_checkpoint) if bootstrap_checkpoint is not None else "",
        "output_dir": str(work_root / "outputs-qvh-phase-a-v4-warmstart"),
        "frame_cache_dir": str(work_root / "cache" / "frames_qvh-phase-a-v4-128-uint8"),
        "visual_feature_cache_dir": str(work_root / "cache" / "mobileclip_qvh-phase-a-v4-128-fp16"),
        "require_visual_feature_cache": True,
        "device": args.device,
        "epochs": 12,
        "batch_size": 1,
        "dataset_size": 128,
        "lr": 0.00005,
        "weight_decay": 0.01,
        "gradient_clip": 5.0,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.1,
        "amp": "auto",
        "accumulation_steps": 8,
        "early_stopping_patience": 4,
        "early_stopping_min_delta": 0.0,
        "early_stopping_min_epochs": 4,
        "monitor": "qvh_mean_score_proxy_Good_mAP",
        "monitor_mode": "max",
        "save_every": 1,
        "checkpoint_keep": 3,
        "prediction_every": 1,
        "prediction_samples": 8,
        "metrics_every": 1,
        "num_workers": 2,
        "log_every": 50,
        "max_steps_per_epoch": 0,
        "max_optimizer_steps": 0,
        "stage2_start_epoch": 0,
        "stage2_visual_lr_scale": 0.05,
        "stage2_unfreeze_strategy": "conv_exp",
        "seed": 7,
        "deterministic": True,
        "allow_random_frames": False,
        "resume": "",
    }
    generated_profile = work_root / "configs" / "train_qvhighlights_phase_a_v4_kaggle.json"
    _write_json(generated_profile, profile)
    print(f"\nGenerated Kaggle profile: {generated_profile}", flush=True)

    if args.prepare_only or args.audit_only:
        return

    if not args.skip_feature_cache:
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "precompute_visual_features.py"),
                "--config",
                str(generated_model_config),
                "--dataset-json",
                str(output_train_json),
                "--dataset-json",
                str(output_val_json),
                "--frame-cache-dir",
                str(work_root / "cache" / "frames_qvh-phase-a-v4-128-uint8"),
                "--visual-feature-cache-dir",
                str(work_root / "cache" / "mobileclip_qvh-phase-a-v4-128-fp16"),
                "--device",
                args.device,
                "--progress-every",
                "50",
            ]
        )

    output_dir = Path(profile["output_dir"])
    if output_dir.exists():
        shutil.rmtree(output_dir)

    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_training_profile.py"),
            "--profile",
            str(generated_profile),
        ]
    )


if __name__ == "__main__":
    main()
