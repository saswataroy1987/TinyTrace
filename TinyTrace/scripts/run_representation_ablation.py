from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.ablation import (
    decide_quality_efficiency_tradeoff,
    summarize_run_artifacts,
    validate_representation_ablation,
)
from tinytrace.config import TinyTraceConfig
from tinytrace.representation import FRAME_COUNT_LADDER
from tinytrace.training import TrainingProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or sequentially execute a controlled Priority 4 ablation."
    )
    parser.add_argument("--training-profile", required=True)
    parser.add_argument("--kind", choices=("frame", "caption"), required=True)
    parser.add_argument("--baseline-frames", type=int, default=8)
    parser.add_argument("--caption-candidate", type=int, choices=(48, 64), default=48)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--benchmark-samples", type=int, default=2)
    parser.add_argument("--benchmark-warmup", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _model_paths(args: argparse.Namespace) -> tuple[Path, Path, str, str]:
    if args.kind == "frame":
        try:
            index = FRAME_COUNT_LADDER.index(args.baseline_frames)
        except ValueError as exc:
            raise ValueError(f"baseline-frames must be one of {FRAME_COUNT_LADDER}.") from exc
        if index + 1 >= len(FRAME_COUNT_LADDER):
            raise ValueError("32 frames has no next sequential candidate.")
        candidate = FRAME_COUNT_LADDER[index + 1]
        return (
            PROJECT_ROOT / "configs" / f"tinytrace_frames_{args.baseline_frames:02d}.json",
            PROJECT_ROOT / "configs" / f"tinytrace_frames_{candidate:02d}.json",
            f"frames-{args.baseline_frames:02d}",
            f"frames-{candidate:02d}",
        )
    return (
        PROJECT_ROOT / "configs" / "tinytrace_caption_020.json",
        PROJECT_ROOT / "configs" / f"tinytrace_caption_{args.caption_candidate:03d}.json",
        "caption-020",
        f"caption-{args.caption_candidate:03d}",
    )


def _derived_profile(
    base: TrainingProfile,
    source_profile_path: Path,
    model_config: Path,
    output_dir: Path,
) -> TrainingProfile:
    values = base.to_dict()
    workspace_root = PROJECT_ROOT.parent
    for key in (
        "train_script",
        "train_dataset_json",
        "val_dataset_json",
        "frame_cache_dir",
    ):
        value = values[key]
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            if path.parts and path.parts[0] in {
                "TinyTrace",
                "final_qvhighlights_tinytrace",
                "dataset",
            }:
                path = workspace_root / path
            else:
                path = source_profile_path.parent / path
        values[key] = str(path.resolve())
    values.update(
        {
            "model_config": str(model_config.resolve()),
            "output_dir": str(output_dir.resolve()),
            "resume": "",
        }
    )
    return TrainingProfile.from_dict(values)


def _run(command: list[str]) -> None:
    print(" ".join(command))
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}.")


def main() -> None:
    args = parse_args()
    if args.benchmark_samples < 1 or args.benchmark_warmup < 0 or args.benchmark_repeats < 1:
        raise ValueError("Benchmark sample/repeat counts must be positive and warmup non-negative.")
    output_root = Path(args.output_root).resolve()
    source_profile_path = Path(args.training_profile).resolve()
    base_training = TrainingProfile.from_json(source_profile_path)
    baseline_model_path, candidate_model_path, baseline_name, candidate_name = _model_paths(args)
    baseline_model = TinyTraceConfig.from_json(baseline_model_path)
    candidate_model = TinyTraceConfig.from_json(candidate_model_path)
    differences = validate_representation_ablation(baseline_model, candidate_model, args.kind)

    experiments = []
    for role, name, model_path in (
        ("baseline", baseline_name, baseline_model_path),
        ("candidate", candidate_name, candidate_model_path),
    ):
        run_dir = output_root / name
        profile = _derived_profile(base_training, source_profile_path, model_path, run_dir)
        profile_path = output_root / "profiles" / f"{name}.json"
        _atomic_json(profile_path, profile.to_dict())
        experiments.append(
            {
                "role": role,
                "name": name,
                "model_config": str(model_path),
                "training_profile": str(profile_path),
                "run_dir": str(run_dir),
            }
        )

    manifest = {
        "priority": 4,
        "kind": args.kind,
        "execution_order": [baseline_name, candidate_name],
        "single_variable_differences": {
            key: {"baseline": values[0], "candidate": values[1]}
            for key, values in differences.items()
        },
        "held_constant": [
            key
            for key in base_training.to_dict()
            if key not in {"model_config", "output_dir", "resume"}
        ],
        "experiments": experiments,
        "automatic_default_change": False,
        "status": "planned",
    }
    manifest_path = output_root / "ablation_manifest.json"
    _atomic_json(manifest_path, manifest)

    if args.execute:
        for experiment in experiments:
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_training_profile.py"),
                    "--profile",
                    str(experiment["training_profile"]),
                ]
            )
            profile = TrainingProfile.from_json(experiment["training_profile"])
            run_dir = Path(experiment["run_dir"])
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "benchmark_representation.py"),
                    "--config",
                    str(experiment["model_config"]),
                    "--checkpoint",
                    str(run_dir / "checkpoints" / "best-primary-metric.pt"),
                    "--dataset-json",
                    profile.val_dataset_json,
                    "--frame-cache-dir",
                    profile.frame_cache_dir,
                    "--output",
                    str(run_dir / "representation_benchmark.json"),
                    "--device",
                    profile.device,
                    "--samples",
                    str(args.benchmark_samples),
                    "--warmup",
                    str(args.benchmark_warmup),
                    "--repeats",
                    str(args.benchmark_repeats),
                ]
            )

        baseline_summary = summarize_run_artifacts(experiments[0]["run_dir"])
        candidate_summary = summarize_run_artifacts(experiments[1]["run_dir"])
        report = {
            "manifest": manifest,
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "decision": decide_quality_efficiency_tradeoff(baseline_summary, candidate_summary),
        }
        _atomic_json(output_root / "ablation_report.json", report)
        manifest["status"] = "completed"
        _atomic_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
