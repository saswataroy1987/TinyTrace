from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and launch TinyTrace Phase B on ActivityNet Captions with "
            "Phase-A-v3 warm-start."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=PROJECT_ROOT / ".phase_b_activitynet_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bootstrap-checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "final_next_phase_assets" / "phase_a_bootstrap" / "phase_a_v3_best_primary_metric.pt",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-feature-cache", action="store_true")
    parser.add_argument("--persist-output-root", type=Path, default=None)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True, cwd=WORKSPACE_ROOT)


def _find_annotations(dataset_root: Path) -> tuple[Path, Path]:
    candidates = [
        dataset_root / "train.json",
        dataset_root / "annotations" / "train.json",
    ]
    train_json = next((path for path in candidates if path.is_file()), None)
    if train_json is None:
        raise FileNotFoundError("Could not find ActivityNet train.json under dataset-root.")
    val_candidates = [
        dataset_root / "val_1.json",
        dataset_root / "annotations" / "val_1.json",
    ]
    val_json = next((path for path in val_candidates if path.is_file()), None)
    if val_json is None:
        raise FileNotFoundError("Could not find ActivityNet val_1.json under dataset-root.")
    return train_json.resolve(), val_json.resolve()


def _find_videos_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "videos",
        dataset_root / "video",
        dataset_root,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not find a videos directory under dataset-root.")


def _ensure_work_videos_link(work_root: Path, dataset_videos_root: Path) -> Path:
    link_path = work_root / "videos"
    if link_path.is_symlink():
        if link_path.resolve() == dataset_videos_root:
            return link_path
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"{link_path} already exists and is not the expected videos link.")
    os.symlink(dataset_videos_root, link_path, target_is_directory=True)
    return link_path


def _resume_checkpoint(output_dir: Path) -> Path | None:
    """Return a checkpoint, or preserve an incomplete pre-checkpoint run aside."""
    latest = output_dir / "checkpoints" / "latest.pt"
    if latest.is_file():
        return latest.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived = output_dir.with_name(f"{output_dir.name}-precheckpoint-{timestamp}")
        output_dir.rename(archived)
        print(
            "Archived incomplete pre-checkpoint Phase-B artifacts without deleting them: "
            f"{archived}",
            flush=True,
        )
    return None


def _export_artifacts(
    *,
    work_root: Path,
    generated_model_config: Path,
    generated_profile: Path,
    output_dir: Path,
    persist_output_root: Path | None,
) -> None:
    if persist_output_root is None:
        return
    persist_output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(generated_model_config, persist_output_root / generated_model_config.name)
    shutil.copy2(generated_profile, persist_output_root / generated_profile.name)
    exported_annotations = persist_output_root / "annotations"
    shutil.rmtree(exported_annotations, ignore_errors=True)
    shutil.copytree(work_root / "annotations", exported_annotations)
    if output_dir.is_dir():
        exported_outputs = persist_output_root / output_dir.name
        shutil.rmtree(exported_outputs, ignore_errors=True)
        shutil.copytree(output_dir, exported_outputs)
    _write_json(
        persist_output_root / "export_summary.json",
        {
            "work_root": str(work_root),
            "output_dir": str(output_dir),
            "generated_model_config": str(generated_model_config),
            "generated_profile": str(generated_profile),
        },
    )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    train_json, val_json = _find_annotations(dataset_root)
    videos_root = _find_videos_root(dataset_root)
    bootstrap_checkpoint = args.bootstrap_checkpoint.resolve()
    if not bootstrap_checkpoint.is_file():
        raise FileNotFoundError(f"Bootstrap checkpoint not found: {bootstrap_checkpoint}")

    _ensure_work_videos_link(work_root, videos_root)

    generated_annotations = work_root / "annotations"
    output_train_json = generated_annotations / "tinytrace_activitynet_phase_b_train.json"
    output_val_json = generated_annotations / "tinytrace_activitynet_phase_b_val.json"
    verified_train_json = generated_annotations / "tinytrace_activitynet_phase_b_train_cache_verified.json"
    verified_val_json = generated_annotations / "tinytrace_activitynet_phase_b_val_cache_verified.json"
    manifest_json = generated_annotations / "activitynet_phase_b_manifest.json"
    cache_failure_manifest = generated_annotations / "activitynet_phase_b_cache_failures.json"

    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "prepare_phase_b_activitynet.py"),
            "--train-json",
            str(train_json),
            "--val-json",
            str(val_json),
            "--videos-root",
            str(videos_root),
            "--output-train-json",
            str(output_train_json),
            "--output-val-json",
            str(output_val_json),
                "--manifest-json",
                str(manifest_json),
                "--media-validation-workers",
                "8",
        ]
    )

    model_config = json.loads(
        (PROJECT_ROOT / "configs" / "tinytrace_activitynet_phase_b_v1.json").read_text(encoding="utf-8")
    )
    model_config["mobileclip_checkpoint"] = str((PROJECT_ROOT / "checkpoints" / "mobileclip_s0.pt").resolve())
    generated_model_config = work_root / "configs" / "tinytrace_activitynet_phase_b_v1_local.json"
    _write_json(generated_model_config, model_config)

    profile = json.loads(
        (PROJECT_ROOT / "configs" / "train_activitynet_phase_b_v1_warmstart.json").read_text(encoding="utf-8")
    )
    profile["model_config"] = str(generated_model_config)
    profile["train_dataset_json"] = str(verified_train_json)
    profile["val_dataset_json"] = str(verified_val_json)
    profile["init_checkpoint"] = str(bootstrap_checkpoint)
    profile["output_dir"] = str(work_root / "outputs-activitynet-phase-b-v1")
    profile["frame_cache_dir"] = str(work_root / "cache" / "frames_activitynet_phase_b_v1")
    profile["visual_feature_cache_dir"] = str(work_root / "cache" / "mobileclip_activitynet_phase_b_v1")
    output_dir = Path(profile["output_dir"])
    resume_checkpoint = _resume_checkpoint(output_dir)
    if resume_checkpoint is not None:
        profile["resume"] = str(resume_checkpoint)
        profile["init_checkpoint"] = ""
        print(f"\nResuming Phase B from: {resume_checkpoint}", flush=True)
    generated_profile = work_root / "configs" / "train_activitynet_phase_b_v1_local.json"
    _write_json(generated_profile, profile)
    print(f"\nGenerated Phase B profile: {generated_profile}", flush=True)

    if args.prepare_only:
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
                "--verified-dataset-json",
                str(verified_train_json),
                "--verified-dataset-json",
                str(verified_val_json),
                "--failure-manifest-json",
                str(cache_failure_manifest),
                "--frame-cache-dir",
                str(work_root / "cache" / "frames_activitynet_phase_b_v1"),
                "--visual-feature-cache-dir",
                str(work_root / "cache" / "mobileclip_activitynet_phase_b_v1"),
                "--device",
                args.device,
                "--progress-every",
                "50",
            ]
        )

    _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_training_profile.py"),
            "--profile",
            str(generated_profile),
        ]
    )
    _export_artifacts(
        work_root=work_root,
        generated_model_config=generated_model_config,
        generated_profile=generated_profile,
        output_dir=output_dir,
        persist_output_root=args.persist_output_root.resolve() if args.persist_output_root else None,
    )


if __name__ == "__main__":
    main()
