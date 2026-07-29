from __future__ import annotations

"""Run the fail-closed TinyTrace QVHighlights Phase-A training gates.

This is deliberately an orchestration script, not another training
implementation.  It prepares (or verifies) the immutable Phase-A-v3 data,
precomputes frozen MobileCLIP features for four clips, proves that those real
examples can be overfit and that both query and video inputs affect the result,
then completes the feature cache and runs an exact
100-optimizer-step real-data smoke test, and only then launches the fresh full
profile.

Every child process is invoked as an argument list (never through a shell).
Gate artifacts are written atomically and existing non-empty run directories
are refused so an old checkpoint cannot accidentally satisfy a new gate.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from tinytrace import JsonTinyTraceDataset, TinyTraceConfig, TinyTraceModel
from tinytrace.metrics import evaluate_event_predictions
from tinytrace.training import TrainingProfile


ANNOTATION_DIR = REPOSITORY_ROOT / "final_qvhighlights_tinytrace" / "annotations"
DEFAULT_MODEL_CONFIG = CODE_ROOT / "configs" / "tinytrace_qvhighlights_phase_a_v3.json"
DEFAULT_FULL_PROFILE = CODE_ROOT / "configs" / "train_qvhighlights_phase_a_v3.json"
DEFAULT_WORK_DIR = CODE_ROOT / "outputs-qvh-phase-a-v3-gates"
PREPARE_SCRIPT = CODE_ROOT / "scripts" / "prepare_phase_a_qvhighlights.py"
PRECOMPUTE_SCRIPT = CODE_ROOT / "scripts" / "precompute_visual_features.py"
TRAIN_SCRIPT = CODE_ROOT / "scripts" / "train_tinytrace.py"
PROFILE_RUNNER = CODE_ROOT / "scripts" / "run_training_profile.py"

TRAIN_JSON = ANNOTATION_DIR / "tinytrace_phase_a_v3_train.json"
VAL_JSON = ANNOTATION_DIR / "tinytrace_phase_a_v3_val.json"
EXCLUSIONS_JSON = ANNOTATION_DIR / "phase_a_v3_exclusions.json"
MANIFEST_JSON = ANNOTATION_DIR / "phase_a_v3_manifest.json"
PHASE_A_OUTPUTS = (TRAIN_JSON, VAL_JSON, EXCLUSIONS_JSON, MANIFEST_JSON)

OVERFIT_SOURCE_IDS = ("1738", "147", "4246", "2941")
EXPECTED_INPUT_COUNTS = {"train": 1218, "val": 136}
EXPECTED_OUTPUT_COUNTS = {"train": 1155, "val": 132}
EXPECTED_EXCLUSIONS = 67
EXPECTED_BINS = 75
EXPECTED_FRAMES = 128
EXPECTED_DURATION_SECONDS = 150.0
MINIMUM_VALID_DURATION_SECONDS = 149.5
SMOKE_OPTIMIZER_STEPS = 100
DETERMINISTIC_CUBLAS_CONFIG = ":4096:8"


class GateFailure(RuntimeError):
    """A declared Phase-A acceptance condition was not met."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and gate TinyTrace Phase A, then launch the full QVHighlights run. "
            "The default path is fail-closed and runs every stage."
        )
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=(
            "Fresh directory for gate subsets, checkpoints, predictions, and reports. "
            "An existing non-empty directory is refused."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Training/extraction device (default: cuda; cuda:N is also accepted).",
    )
    parser.add_argument(
        "--skip-feature-cache",
        action="store_true",
        help="Reuse an already complete feature cache instead of invoking precomputation.",
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help="Stop after all overfit/conditioning/smoke gates pass.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate data, media, configs, checkpoint, GPU, and hashes, then stop.",
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=400,
        help="Exact optimizer-step cap for the four-real-video overfit gate (default: 400).",
    )
    parser.add_argument(
        "--min-overfit-map-gain",
        type=float,
        default=10.0,
        help=(
            "Required Good proxy-mAP gain, in percentage points, over the stronger of "
            "the seeded untrained model and a constant curve (default: 10)."
        ),
    )
    parser.add_argument(
        "--min-overfit-good-map",
        type=float,
        default=75.0,
        help="Minimum absolute Good proxy-mAP for the four-video overfit (default: 75).",
    )
    parser.add_argument(
        "--max-overfit-loss-ratio",
        type=float,
        default=0.35,
        help="Maximum final/first-epoch training-loss ratio (default: 0.35).",
    )
    parser.add_argument(
        "--max-overfit-mae-ratio",
        type=float,
        default=0.65,
        help="Maximum trained/untrained dense score-MAE ratio (default: 0.65).",
    )
    parser.add_argument(
        "--min-conditioning-delta",
        type=float,
        default=1e-4,
        help=(
            "Minimum mean absolute saliency-logit change for cyclic query swap, cyclic "
            "video swap, and zero-visual ablations (default: 1e-4)."
        ),
    )
    parser.add_argument(
        "--min-conditioning-score-delta",
        type=float,
        default=0.01,
        help=(
            "Minimum mean absolute saliency-score change on the 0-4 scale for cyclic "
            "query, cyclic video, and zero-visual counterfactuals (default: 0.01)."
        ),
    )
    parser.add_argument(
        "--min-conditioning-map-drop",
        type=float,
        default=1.0,
        help=(
            "Minimum Good proxy-mAP drop, in percentage points, for cyclic-video "
            "and zero-visual counterfactuals (default: 1). Query swaps are checked "
            "with score/logit deltas because AP is invariant to score calibration."
        ),
    )
    parser.add_argument(
        "--max-smoke-clipping-rate",
        type=float,
        default=0.5,
        help=(
            "Exclusive upper bound on the fraction of clipped optimizer updates "
            "in the 100-step smoke gate (default: 0.5)."
        ),
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateFailure(f"Required JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateFailure(f"Invalid JSON in {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateFailure(f"{context} must be numeric, received {value!r}.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise GateFailure(f"{context} must be finite, received {parsed!r}.")
    return parsed


def _resolve_profile_path(profile_path: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    repository_candidate = REPOSITORY_ROOT / candidate
    if repository_candidate.exists() or (
        candidate.parts
        and candidate.parts[0] in {"TinyTrace", "final_qvhighlights_tinytrace", "dataset"}
    ):
        return repository_candidate.resolve(strict=False)
    return (profile_path.parent / candidate).resolve(strict=False)


def _fresh_directory(path: Path, label: str) -> Path:
    path = path.expanduser().resolve(strict=False)
    if path.exists() and not path.is_dir():
        raise GateFailure(f"{label} must be a directory, but a file exists at {path}.")
    if path.is_dir() and any(path.iterdir()):
        raise GateFailure(
            f"Refusing non-empty {label}: {path}. Use a new --work-dir/output directory; "
            "old gate artifacts cannot be reused implicitly."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_command(command: list[str], *, stage: str, work_dir: Path) -> None:
    print(f"\n[{stage}]", flush=True)
    print(shlex.join(command), flush=True)
    started = time.time()
    child_environment = os.environ.copy()
    child_environment["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_CONFIG
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        env=child_environment,
    )
    report = {
        "stage": stage,
        "command": command,
        "returncode": result.returncode,
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "cublas_workspace_config": child_environment["CUBLAS_WORKSPACE_CONFIG"],
        "status": "passed" if result.returncode == 0 else "failed",
    }
    _atomic_json(work_dir / f"stage-{stage}.json", report)
    if result.returncode != 0:
        raise GateFailure(f"Stage {stage!r} failed with exit status {result.returncode}.")


def _resolve_device(value: str) -> tuple[torch.device, dict[str, object]]:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise GateFailure(f"Invalid --device value {value!r}.") from exc
    report: dict[str, object] = {"requested": value, "resolved": str(device)}
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise GateFailure("CUDA was requested, but torch.cuda.is_available() is false.")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if not 0 <= index < torch.cuda.device_count():
            raise GateFailure(
                f"CUDA device index {index} is unavailable; detected {torch.cuda.device_count()} GPU(s)."
            )
        properties = torch.cuda.get_device_properties(index)
        report.update(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": [properties.major, properties.minor],
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return device, report


def _ensure_phase_a_data(work_dir: Path) -> None:
    present = [path.exists() for path in PHASE_A_OUTPUTS]
    if any(present) and not all(present):
        missing = [str(path) for path, exists in zip(PHASE_A_OUTPUTS, present) if not exists]
        raise GateFailure(
            "Phase-A-v3 publication is incomplete; immutable outputs are all-or-nothing. "
            f"Missing: {missing}"
        )
    if all(present):
        print("[prepare-data] Reusing the complete immutable Phase-A-v3 publication.", flush=True)
        _atomic_json(
            work_dir / "stage-prepare-data.json",
            {"stage": "prepare-data", "status": "reused", "outputs": [str(p) for p in PHASE_A_OUTPUTS]},
        )
        return
    _run_command([sys.executable, str(PREPARE_SCRIPT)], stage="prepare-data", work_dir=work_dir)


def _validate_manifest() -> dict[str, object]:
    manifest = _read_json(MANIFEST_JSON)
    if not isinstance(manifest, dict):
        raise GateFailure("Phase-A manifest must be a JSON object.")
    if manifest.get("schema_version") != "tinytrace.qvhighlights.phase-a.v3":
        raise GateFailure(f"Unexpected Phase-A schema version: {manifest.get('schema_version')!r}.")
    if manifest.get("immutable") is not True:
        raise GateFailure("Phase-A manifest is not marked immutable.")

    verified_hashes: dict[str, str] = {}
    for section in ("source_files", "outputs"):
        records = manifest.get(section)
        if not isinstance(records, dict) or not records:
            raise GateFailure(f"Manifest section {section!r} is missing or empty.")
        for name, record in records.items():
            if not isinstance(record, dict):
                raise GateFailure(f"Manifest {section}.{name} must be an object.")
            path = Path(str(record.get("path", "")))
            expected = str(record.get("sha256", "")).lower()
            if not path.is_file():
                raise GateFailure(f"Manifest-referenced file is missing: {path}")
            actual = _sha256(path)
            if not expected or actual != expected:
                raise GateFailure(
                    f"SHA-256 mismatch for {path}: expected {expected or '<missing>'}, got {actual}."
                )
            verified_hashes[str(path)] = actual

    contract = manifest.get("target_contract", {})
    if contract.get("bin_count") != EXPECTED_BINS or float(contract.get("bin_size_seconds", -1)) != 2.0:
        raise GateFailure("Phase-A manifest must declare exactly 75 direct two-second bins.")
    media_validation = manifest.get("media_validation", {})
    if float(media_validation.get("expected_duration_seconds", -1)) != EXPECTED_DURATION_SECONDS:
        raise GateFailure("Phase-A manifest must require a 150-second media window.")
    if (
        float(media_validation.get("minimum_valid_duration_seconds", -1))
        != MINIMUM_VALID_DURATION_SECONDS
    ):
        raise GateFailure("Phase-A manifest must reject media shorter than 149.5 seconds.")
    counts = manifest.get("counts", {})
    expected_counts = {
        "input_train": EXPECTED_INPUT_COUNTS["train"],
        "input_val": EXPECTED_INPUT_COUNTS["val"],
        "output_train": EXPECTED_OUTPUT_COUNTS["train"],
        "output_val": EXPECTED_OUTPUT_COUNTS["val"],
        "excluded_total": EXPECTED_EXCLUSIONS,
    }
    mismatches = {
        key: {"expected": expected, "actual": counts.get(key)}
        for key, expected in expected_counts.items()
        if counts.get(key) != expected
    }
    if mismatches:
        raise GateFailure(f"Unexpected Phase-A dataset counts: {mismatches}")
    return {"hashes": verified_hashes, "counts": counts, "target_contract": contract}


def _resolve_annotation_media(annotation_path: Path, raw_path: str) -> Path:
    source = Path(raw_path)
    if source.is_absolute():
        return source.resolve(strict=False)
    roots = (
        annotation_path.parent,
        annotation_path.parent.parent,
        annotation_path.parent.parent.parent,
        REPOSITORY_ROOT,
    )
    for root in roots:
        candidate = root / source
        if candidate.is_file():
            return candidate.resolve()
    return (annotation_path.parent.parent / source).resolve(strict=False)


def _validate_split(path: Path, split: str, expected_count: int) -> tuple[list[dict], dict[str, object]]:
    payload = _read_json(path)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise GateFailure(f"{split} Phase-A JSON must be a list of objects: {path}")
    if len(payload) != expected_count:
        raise GateFailure(f"{split} count mismatch: expected {expected_count}, received {len(payload)}.")

    source_ids: set[str] = set()
    media_paths: set[str] = set()
    positive_bins = 0
    nonzero_bins = 0
    current_durations: list[float] = []
    for index, item in enumerate(payload):
        source_id = str(item.get("source_id", "")).strip()
        if not source_id or source_id in source_ids:
            raise GateFailure(f"{split}[{index}] has a missing or duplicate source_id {source_id!r}.")
        source_ids.add(source_id)
        if item.get("task_mode") != "highlight":
            raise GateFailure(f"{split}[{index}] is not a highlight sample.")
        query = item.get("query")
        instruction = item.get("instruction")
        if not isinstance(query, str) or not query.strip():
            raise GateFailure(f"{split}[{index}] has an empty query.")
        if not isinstance(instruction, str) or query not in instruction:
            raise GateFailure(
                f"{split}[{index}] instruction must contain the query so query conditioning is trained."
            )
        scores = item.get("dense_saliency_scores")
        if not isinstance(scores, list) or len(scores) != EXPECTED_BINS:
            raise GateFailure(f"{split}[{index}] must contain exactly {EXPECTED_BINS} saliency scores.")
        parsed_scores = [_finite_number(value, f"{split}[{index}].dense_saliency_scores") for value in scores]
        if any(value < 0.0 or value > 4.0 for value in parsed_scores):
            raise GateFailure(f"{split}[{index}] contains saliency outside [0, 4].")
        nonzero_bins += sum(value > 0.0 for value in parsed_scores)
        positive_bins += sum(value >= 3.0 for value in parsed_scores)
        if item.get("saliency_bin_count") != EXPECTED_BINS:
            raise GateFailure(f"{split}[{index}] saliency_bin_count is not 75.")
        if float(item.get("saliency_bin_size_seconds", -1)) != 2.0:
            raise GateFailure(f"{split}[{index}] saliency_bin_size_seconds is not 2.0.")
        duration = _finite_number(item.get("duration_seconds"), f"{split}[{index}].duration_seconds")
        if duration < MINIMUM_VALID_DURATION_SECONDS:
            raise GateFailure(
                f"{split}[{index}] media is truncated: duration={duration:g}, "
                f"required>={MINIMUM_VALID_DURATION_SECONDS:g}."
            )
        raw_video = item.get("video_path")
        if not isinstance(raw_video, str) or not raw_video.strip():
            raise GateFailure(f"{split}[{index}] has no video_path.")
        video = _resolve_annotation_media(path, raw_video)
        if not video.is_file() or video.stat().st_size <= 0:
            raise GateFailure(f"{split}[{index}] media is missing or empty: {video}")
        try:
            current_duration = JsonTinyTraceDataset._probe_video_duration(str(video))
        except Exception as exc:
            raise GateFailure(
                f"{split}[{index}] current media cannot be probed: {video}: {exc}"
            ) from exc
        if current_duration < MINIMUM_VALID_DURATION_SECONDS:
            raise GateFailure(
                f"{split}[{index}] current media became truncated after publication: "
                f"duration={current_duration:g}, required>={MINIMUM_VALID_DURATION_SECONDS:g}."
            )
        if abs(current_duration - duration) > 0.5:
            raise GateFailure(
                f"{split}[{index}] media duration changed after publication: "
                f"recorded={duration:g}, current={current_duration:g}."
            )
        current_durations.append(current_duration)
        canonical_video = str(video)
        if canonical_video in media_paths:
            raise GateFailure(f"{split} contains duplicate media path {canonical_video}.")
        media_paths.add(canonical_video)

    if positive_bins == 0 or nonzero_bins == 0:
        raise GateFailure(f"{split} contains no positive/nonzero saliency supervision.")
    return payload, {
        "samples": len(payload),
        "source_ids": source_ids,
        "media_paths": media_paths,
        "nonzero_bins": nonzero_bins,
        "good_bins": positive_bins,
        "negative_bins": len(payload) * EXPECTED_BINS - positive_bins,
        "minimum_current_duration_seconds": min(current_durations),
        "maximum_current_duration_seconds": max(current_durations),
    }


def _validate_model_and_profile(device_value: str) -> tuple[TinyTraceConfig, dict, dict[str, Path]]:
    config = TinyTraceConfig.from_json(DEFAULT_MODEL_CONFIG)
    if not config.phase_a_dense_saliency:
        raise GateFailure("Phase-A-v3 config must enable phase_a_dense_saliency.")
    if config.max_frames != EXPECTED_FRAMES:
        raise GateFailure(f"Phase-A-v3 must use {EXPECTED_FRAMES} frames, got {config.max_frames}.")
    if config.phase_a_bin_count != EXPECTED_BINS or config.phase_a_bin_size_seconds != 2.0:
        raise GateFailure("Phase-A-v3 config must predict 75 direct two-second bins.")
    if not config.freeze_visual_encoder:
        raise GateFailure("The Phase-A edge baseline requires a frozen visual encoder.")
    if any(
        value != 0.0
        for value in (
            config.time_loss_weight,
            config.score_loss_weight,
            config.caption_loss_weight,
            config.sync_loss_weight,
            config.boundary_loss_weight,
        )
    ):
        raise GateFailure("Autoregressive timestamp/score/caption losses must be disabled in dense Phase A.")

    profile_path = DEFAULT_FULL_PROFILE.resolve()
    profile = TrainingProfile.from_json(profile_path).to_dict()
    if not math.isclose(float(profile["lr"]), 1e-4, rel_tol=0.0, abs_tol=1e-12):
        raise GateFailure(f"Full Phase-A learning rate must be 0.0001, got {profile['lr']!r}.")
    if not math.isclose(float(profile["gradient_clip"]), 5.0, rel_tol=0.0, abs_tol=1e-12):
        raise GateFailure(
            "Full Phase-A gradient clip must be 5.0 after the measured real-batch "
            f"gradient audit, got {profile['gradient_clip']!r}."
        )
    if int(profile.get("max_optimizer_steps", 0)) != 0:
        raise GateFailure("Full Phase-A profile must not have an optimizer-step debug cap.")
    if int(profile.get("stage2_start_epoch", 0)) != 0:
        raise GateFailure("Full Phase-A profile must retain the frozen MobileCLIP baseline.")
    profile_model = _resolve_profile_path(profile_path, str(profile["model_config"]))
    if profile_model != DEFAULT_MODEL_CONFIG.resolve():
        raise GateFailure(f"Full profile does not use the Phase-A-v3 model config: {profile_model}")
    paths = {
        "train": _resolve_profile_path(profile_path, str(profile["train_dataset_json"])),
        "val": _resolve_profile_path(profile_path, str(profile["val_dataset_json"])),
        "output": _resolve_profile_path(profile_path, str(profile["output_dir"])),
        "frame_cache": _resolve_profile_path(profile_path, str(profile["frame_cache_dir"])),
        "feature_cache": _resolve_profile_path(profile_path, str(profile["visual_feature_cache_dir"])),
    }
    if paths["train"] != TRAIN_JSON.resolve() or paths["val"] != VAL_JSON.resolve():
        raise GateFailure("Full profile is not bound to the immutable Phase-A-v3 train/validation files.")
    if not bool(profile.get("require_visual_feature_cache")):
        raise GateFailure("Full Phase-A profile must require the frozen visual-feature cache.")

    checkpoint = Path(config.mobileclip_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = CODE_ROOT / checkpoint
    if not checkpoint.is_file():
        raise GateFailure(f"Pinned MobileCLIP checkpoint is missing: {checkpoint}")
    actual_checkpoint_sha = _sha256(checkpoint)
    if actual_checkpoint_sha != config.mobileclip_checkpoint_sha256.lower():
        raise GateFailure(
            "Pinned MobileCLIP checkpoint SHA mismatch: "
            f"expected {config.mobileclip_checkpoint_sha256}, got {actual_checkpoint_sha}."
        )
    profile["device"] = device_value
    return config, profile, paths


def _preflight(args: argparse.Namespace, work_dir: Path) -> tuple[TinyTraceConfig, dict, dict[str, Path], list[dict]]:
    device, device_report = _resolve_device(args.device)
    manifest_report = _validate_manifest()
    config, profile, paths = _validate_model_and_profile(args.device)
    train_rows, train_report = _validate_split(TRAIN_JSON, "train", EXPECTED_OUTPUT_COUNTS["train"])
    _, val_report = _validate_split(VAL_JSON, "validation", EXPECTED_OUTPUT_COUNTS["val"])
    expected_positive_weight = train_report["negative_bins"] / train_report["good_bins"]
    if not math.isclose(
        config.saliency_positive_weight,
        expected_positive_weight,
        rel_tol=0.0,
        abs_tol=0.01,
    ):
        raise GateFailure(
            "saliency_positive_weight must match the cleaned training split's "
            f"negative/positive-bin ratio ({expected_positive_weight:.6f}), got "
            f"{config.saliency_positive_weight:.6f}."
        )
    train_report["expected_saliency_positive_weight"] = expected_positive_weight
    source_overlap = train_report["source_ids"].intersection(val_report["source_ids"])
    media_overlap = train_report["media_paths"].intersection(val_report["media_paths"])
    if source_overlap or media_overlap:
        raise GateFailure(
            "Train/validation disjointness failed: "
            f"source_overlap={sorted(source_overlap)[:8]} media_overlap={sorted(media_overlap)[:8]}"
        )

    report = {
        "status": "passed",
        "device": device_report,
        "model_config": str(DEFAULT_MODEL_CONFIG.resolve()),
        "model_config_sha256": _sha256(DEFAULT_MODEL_CONFIG),
        "full_profile": str(DEFAULT_FULL_PROFILE.resolve()),
        "full_profile_sha256": _sha256(DEFAULT_FULL_PROFILE),
        "manifest": manifest_report,
        "train": {key: value for key, value in train_report.items() if not isinstance(value, set)},
        "validation": {key: value for key, value in val_report.items() if not isinstance(value, set)},
        "source_id_overlap": 0,
        "media_path_overlap": 0,
        "resolved_paths": {key: str(value) for key, value in paths.items()},
        "config_contract": {
            "max_frames": config.max_frames,
            "phase_a_bin_count": config.phase_a_bin_count,
            "phase_a_bin_size_seconds": config.phase_a_bin_size_seconds,
            "freeze_visual_encoder": config.freeze_visual_encoder,
            "learning_rate": profile["lr"],
        },
        "torch_version": torch.__version__,
        "device_type": device.type,
    }
    _atomic_json(work_dir / "preflight.json", report)
    return config, profile, paths, train_rows


def _precompute_features(
    args: argparse.Namespace,
    work_dir: Path,
    paths: dict[str, Path],
    *,
    dataset_paths: tuple[Path, ...],
    stage: str,
    expected_entries: int,
) -> None:
    command = [
        sys.executable,
        str(PRECOMPUTE_SCRIPT),
        "--config",
        str(DEFAULT_MODEL_CONFIG.resolve()),
    ]
    for dataset_path in dataset_paths:
        command.extend(["--dataset-json", str(dataset_path.resolve())])
    command.extend(
        [
            "--frame-cache-dir",
            str(paths["frame_cache"]),
            "--visual-feature-cache-dir",
            str(paths["feature_cache"]),
            "--device",
            args.device,
            "--amp",
            "auto",
            "--progress-every",
            "25",
        ]
    )
    if args.skip_feature_cache:
        print(
            f"[{stage}] Precomputation skipped by request; validating cache coverage.",
            flush=True,
        )
        _atomic_json(
            work_dir / f"stage-{stage}.json",
            {"stage": stage, "status": "skipped_requested"},
        )
    else:
        _run_command(command, stage=stage, work_dir=work_dir)

    missing: list[str] = []
    invalid: list[str] = []
    visited = 0
    for dataset_path in dataset_paths:
        dataset = JsonTinyTraceDataset(
            dataset_path,
            config=TinyTraceConfig.from_json(DEFAULT_MODEL_CONFIG),
            frame_cache_dir=paths["frame_cache"],
            allow_random_frames=False,
            validate_videos_on_init=False,
            strict_media_validation=True,
            visual_feature_cache_dir=paths["feature_cache"],
            require_visual_feature_cache=args.skip_feature_cache,
        )
        for index in range(len(dataset)):
            visited += 1
            cache_path = dataset.visual_feature_cache_path(index)
            if not cache_path.is_file() or cache_path.stat().st_size <= 0:
                missing.append(f"source_id={dataset.items[index].get('source_id')} cache={cache_path}")
                continue
            if args.skip_feature_cache:
                # A skipped precompute must prove more than file existence. The
                # canonical strict reader checks format version, frame/patch
                # dimensions, alignment, and finite values for every entry.
                try:
                    sample = dataset[index]
                    if "visual_patch_features" not in sample:
                        raise ValueError("cache was not loaded")
                except Exception as exc:
                    invalid.append(
                        f"source_id={dataset.items[index].get('source_id')} "
                        f"cache={cache_path}: {exc}"
                    )
    if visited != expected_entries:
        raise GateFailure(
            f"{stage} dataset count mismatch: expected {expected_entries}, visited {visited}."
        )
    if missing or invalid:
        raise GateFailure(
            "Frozen visual-feature cache validation failed "
            f"({len(missing)} missing, {len(invalid)} invalid); "
            f"examples: {(missing + invalid)[:5]}"
        )
    _atomic_json(
        work_dir / f"{stage}-validation.json",
        {
            "status": "passed",
            "stage": stage,
            "expected_entries": expected_entries,
            "missing_entries": 0,
            "invalid_entries": 0,
            "content_validation": "full" if args.skip_feature_cache else "precompute-reader",
            "cache_dir": str(paths["feature_cache"]),
        },
    )


def _write_overfit_subset(train_rows: list[dict], work_dir: Path) -> Path:
    by_id = {str(row.get("source_id")): row for row in train_rows}
    missing = [source_id for source_id in OVERFIT_SOURCE_IDS if source_id not in by_id]
    if missing:
        raise GateFailure(f"Declared real overfit source ids are absent after media repair: {missing}")
    subset: list[dict] = []
    for source_id in OVERFIT_SOURCE_IDS:
        row = dict(by_id[source_id])
        row["video_path"] = str(_resolve_annotation_media(TRAIN_JSON, str(row["video_path"])))
        subset.append(row)
    subset_path = work_dir / "overfit-four-real.json"
    _atomic_json(subset_path, subset)
    return subset_path


def _training_command(
    *,
    dataset_json: Path,
    val_json: Path,
    output_dir: Path,
    frame_cache: Path,
    feature_cache: Path,
    device: str,
    epochs: int,
    max_optimizer_steps: int,
    accumulation_steps: int,
    monitor: str,
    monitor_mode: str,
    prediction_every: int,
    prediction_samples: int,
    metrics_every: int,
    num_workers: int,
    log_every: int,
) -> list[str]:
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(DEFAULT_MODEL_CONFIG.resolve()),
        "--dataset-json",
        str(dataset_json.resolve()),
        "--val-dataset-json",
        str(val_json.resolve()),
        "--output-dir",
        str(output_dir.resolve(strict=False)),
        "--frame-cache-dir",
        str(frame_cache),
        "--visual-feature-cache-dir",
        str(feature_cache),
        "--require-visual-feature-cache",
        "--device",
        device,
        "--epochs",
        str(epochs),
        "--batch-size",
        "1",
        "--lr",
        "0.0001",
        "--weight-decay",
        "0.01",
        "--gradient-clip",
        "5.0",
        "--warmup-ratio",
        "0.05",
        "--min-lr-ratio",
        "0.1",
        "--amp",
        "auto",
        "--accumulation-steps",
        str(accumulation_steps),
        "--early-stopping-patience",
        "0",
        "--early-stopping-min-delta",
        "0.0",
        "--early-stopping-min-epochs",
        "1",
        "--monitor",
        monitor,
        "--monitor-mode",
        monitor_mode,
        "--save-every",
        "0",
        "--checkpoint-keep",
        "1",
        "--prediction-every",
        str(prediction_every),
        "--prediction-samples",
        str(prediction_samples),
        "--metrics-every",
        str(metrics_every),
        "--num-workers",
        str(num_workers),
        "--log-every",
        str(log_every),
        "--max-steps-per-epoch",
        "0",
        "--max-optimizer-steps",
        str(max_optimizer_steps),
        "--stage2-start-epoch",
        "0",
        "--stage2-visual-lr-scale",
        "0.05",
        "--stage2-unfreeze-strategy",
        "conv_exp",
        "--seed",
        "7",
        "--deterministic",
    ]


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyTraceModel, TinyTraceConfig, dict]:
    if not checkpoint_path.is_file():
        raise GateFailure(f"Expected checkpoint is missing: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise GateFailure(f"Unable to load gate checkpoint {checkpoint_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise GateFailure(f"Checkpoint lacks a model_state object: {checkpoint_path}")
    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        raise GateFailure(f"Checkpoint lacks its model config: {checkpoint_path}")
    config = TinyTraceConfig.from_dict(config_payload)
    model = TinyTraceModel(config, load_pretrained_visual=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, config, payload


def _dense_forward(
    model: TinyTraceModel,
    sample: dict,
    prompt_sample: dict,
    device: torch.device,
    *,
    zero_visual: bool = False,
) -> tuple[list[float], list[float]]:
    prompt_length = int(prompt_sample["prompt_length"])
    prompt_ids = prompt_sample["token_ids"][:prompt_length].unsqueeze(0).to(device)
    patch_features = sample.get("visual_patch_features")
    if patch_features is None:
        raise GateFailure(f"Required visual feature cache was not loaded for {sample.get('source_id')}.")
    patch_features = patch_features.unsqueeze(0).to(device)
    if zero_visual:
        patch_features = torch.zeros_like(patch_features)
    with torch.inference_mode():
        output = model(
            sample["frames"].unsqueeze(0).to(device),
            sample["frame_times"].unsqueeze(0).to(device),
            prompt_ids,
            frame_mask=torch.ones(
                1, sample["frame_times"].numel(), dtype=torch.bool, device=device
            ),
            visual_patch_features=patch_features,
            prompt_lengths=torch.tensor([prompt_length], dtype=torch.long, device=device),
        )
    if output.saliency_logits is None or output.saliency_scores is None:
        raise GateFailure("Dense Phase-A checkpoint did not produce saliency outputs.")
    logits = output.saliency_logits[0].detach().float().cpu().tolist()
    scores = output.saliency_scores[0].detach().float().cpu().tolist()
    return logits, scores


def _dense_events(values: list[float], threshold: float, bin_size: float) -> list[dict]:
    events: list[dict] = []
    start: int | None = None
    for index in range(len(values) + 1):
        active = index < len(values) and values[index] >= threshold
        if active and start is None:
            start = index
        if not active and start is not None:
            events.append(
                {
                    "timestamp": [start * bin_size, index * bin_size],
                    "score": [max(values[start:index])],
                }
            )
            start = None
    return events


def _prediction_artifacts(
    model: TinyTraceModel,
    config: TinyTraceConfig,
    dataset: JsonTinyTraceDataset,
    device: torch.device,
    *,
    checkpoint_identity: str,
) -> list[dict]:
    artifacts: list[dict] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        _, scores = _dense_forward(model, sample, sample, device)
        targets = sample["saliency_targets"].float().tolist()
        artifacts.append(
            {
                "sample_index": index,
                "source_id": sample.get("source_id"),
                "qid": sample.get("source_id"),
                "query": sample.get("query"),
                "video_path": sample.get("video_path"),
                "task_mode": "highlight",
                "checkpoint_identity": checkpoint_identity,
                "ground_truth": _dense_events(
                    targets, config.phase_a_positive_threshold, config.phase_a_bin_size_seconds
                ),
                "predicted": _dense_events(
                    scores, config.phase_a_positive_threshold, config.phase_a_bin_size_seconds
                ),
                "pred_saliency_scores": scores,
                "qvh_mean_score_targets": targets,
            }
        )
    return artifacts


def _dense_fit_report(
    artifacts: Iterable[dict],
    *,
    positive_threshold: float,
) -> dict[str, float | int]:
    absolute_errors: list[float] = []
    positive_errors: list[float] = []
    nonpositive_errors: list[float] = []
    for sample_index, artifact in enumerate(artifacts):
        predictions = artifact.get("pred_saliency_scores")
        targets = artifact.get("qvh_mean_score_targets")
        if not isinstance(predictions, list) or not isinstance(targets, list):
            raise GateFailure(f"Dense fit artifact {sample_index} is missing scores/targets.")
        if len(predictions) != EXPECTED_BINS or len(targets) != EXPECTED_BINS:
            raise GateFailure(f"Dense fit artifact {sample_index} does not contain 75 bins.")
        for bin_index, (prediction, target) in enumerate(zip(predictions, targets)):
            predicted_score = _finite_number(
                prediction, f"dense fit sample {sample_index} prediction {bin_index}"
            )
            target_score = _finite_number(
                target, f"dense fit sample {sample_index} target {bin_index}"
            )
            error = abs(predicted_score - target_score)
            absolute_errors.append(error)
            if target_score >= positive_threshold:
                positive_errors.append(error)
            else:
                nonpositive_errors.append(error)
    if not absolute_errors or not positive_errors or not nonpositive_errors:
        raise GateFailure("Dense fit report requires positive and non-positive target bins.")
    return {
        "bins": len(absolute_errors),
        "positive_bins": len(positive_errors),
        "nonpositive_bins": len(nonpositive_errors),
        "score_mae": sum(absolute_errors) / len(absolute_errors),
        "positive_score_mae": sum(positive_errors) / len(positive_errors),
        "nonpositive_score_mae": sum(nonpositive_errors) / len(nonpositive_errors),
    }


def _seeded_initial_model(config: TinyTraceConfig, device: torch.device) -> TinyTraceModel:
    """Recreate the exact seed-7 task initialization used by the child trainer."""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(7)
            torch.manual_seed(7)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(7)
            model = TinyTraceModel(config, load_pretrained_visual=False)
    finally:
        random.setstate(python_state)
    model.to(device)
    model.eval()
    return model


def _validate_curves(curves: Iterable[list[float]], context: str) -> dict[str, object]:
    parsed = list(curves)
    if not parsed:
        raise GateFailure(f"{context} contains no prediction curves.")
    variances: list[float] = []
    ranges: list[float] = []
    for index, curve in enumerate(parsed):
        if not isinstance(curve, list) or len(curve) != EXPECTED_BINS:
            raise GateFailure(f"{context}[{index}] is not a {EXPECTED_BINS}-bin curve.")
        values = [_finite_number(value, f"{context}[{index}]") for value in curve]
        if any(value < 0.0 or value > 4.0 for value in values):
            raise GateFailure(f"{context}[{index}] contains a score outside [0, 4].")
        mean = sum(values) / len(values)
        variances.append(sum((value - mean) ** 2 for value in values) / len(values))
        ranges.append(max(values) - min(values))
    distinct = len({tuple(round(value, 6) for value in curve) for curve in parsed})
    return {
        "curves": len(parsed),
        "distinct_curves": distinct,
        "mean_within_curve_variance": sum(variances) / len(variances),
        "minimum_curve_range": min(ranges),
    }


def _run_overfit(
    args: argparse.Namespace,
    work_dir: Path,
    subset_path: Path,
    paths: dict[str, Path],
    device: torch.device,
) -> tuple[TinyTraceModel, TinyTraceConfig, JsonTinyTraceDataset, list[dict]]:
    if args.overfit_steps < 4:
        raise GateFailure("--overfit-steps must be at least 4 for a four-video gate.")
    if not math.isfinite(args.min_overfit_map_gain) or args.min_overfit_map_gain <= 0.0:
        raise GateFailure("--min-overfit-map-gain must be finite and positive.")
    if not math.isfinite(args.min_overfit_good_map) or not 0.0 < args.min_overfit_good_map <= 100.0:
        raise GateFailure("--min-overfit-good-map must be finite and in (0, 100].")
    for name in ("max_overfit_loss_ratio", "max_overfit_mae_ratio"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise GateFailure(f"--{name.replace('_', '-')} must be finite and in (0, 1).")
    overfit_dir = work_dir / "overfit"
    if overfit_dir.exists() and any(overfit_dir.iterdir()):
        raise GateFailure(f"Refusing existing non-empty overfit output: {overfit_dir}")
    steps_per_epoch = len(OVERFIT_SOURCE_IDS)
    epochs = math.ceil(args.overfit_steps / steps_per_epoch)
    command = _training_command(
        dataset_json=subset_path,
        val_json=subset_path,
        output_dir=overfit_dir,
        frame_cache=paths["frame_cache"],
        feature_cache=paths["feature_cache"],
        device=args.device,
        epochs=epochs,
        max_optimizer_steps=args.overfit_steps,
        accumulation_steps=1,
        monitor="val_loss",
        monitor_mode="min",
        prediction_every=epochs,
        prediction_samples=len(OVERFIT_SOURCE_IDS),
        metrics_every=epochs,
        num_workers=0,
        log_every=steps_per_epoch,
    )
    _run_command(command, stage="overfit", work_dir=work_dir)
    summary = _read_json(overfit_dir / "run_summary.json")
    history = _read_json(overfit_dir / "history.json")
    if not isinstance(history, list) or not history:
        raise GateFailure("Overfit run did not publish a non-empty epoch history.")
    if summary.get("global_step") != args.overfit_steps:
        raise GateFailure(
            f"Overfit optimizer cap was not exact: expected {args.overfit_steps}, "
            f"got {summary.get('global_step')}."
        )
    model, config, checkpoint = _load_checkpoint_model(overfit_dir / "checkpoints" / "latest.pt", device)
    if checkpoint.get("global_step") != args.overfit_steps:
        raise GateFailure("Latest overfit checkpoint does not correspond to the declared final step.")
    trainer_prediction_path = overfit_dir / "predictions" / f"epoch-{epochs:04d}.json"
    if not trainer_prediction_path.is_file():
        raise GateFailure(
            "The overfit run did not publish its final-epoch prediction artifact: "
            f"{trainer_prediction_path}"
        )
    dataset = JsonTinyTraceDataset(
        subset_path,
        config=config,
        frame_cache_dir=paths["frame_cache"],
        allow_random_frames=False,
        validate_videos_on_init=True,
        strict_media_validation=True,
        visual_feature_cache_dir=paths["feature_cache"],
        require_visual_feature_cache=True,
    )
    initial_model = _seeded_initial_model(config, device)
    initial_artifacts = _prediction_artifacts(
        initial_model,
        config,
        dataset,
        device,
        checkpoint_identity="seed-7-untrained",
    )
    del initial_model
    _atomic_json(work_dir / "overfit-initial-predictions.json", initial_artifacts)
    checkpoint_identity = f"overfit-latest:step-{args.overfit_steps}"
    artifacts = _prediction_artifacts(
        model,
        config,
        dataset,
        device,
        checkpoint_identity=checkpoint_identity,
    )
    _atomic_json(work_dir / "overfit-predictions.json", artifacts)
    metrics = evaluate_event_predictions(artifacts)
    initial_metrics = evaluate_event_predictions(initial_artifacts)
    fit_report = _dense_fit_report(
        artifacts, positive_threshold=config.phase_a_positive_threshold
    )
    initial_fit_report = _dense_fit_report(
        initial_artifacts, positive_threshold=config.phase_a_positive_threshold
    )
    curve_report = _validate_curves(
        [artifact["pred_saliency_scores"] for artifact in artifacts], "overfit predictions"
    )
    metric_name = "qvh_mean_score_proxy_Good_mAP"
    constant_name = "qvh_mean_score_proxy_Good_constant_mAP"
    if metric_name not in metrics or constant_name not in metrics:
        raise GateFailure("Overfit evaluator did not emit the declared Good proxy/constant metrics.")
    trained_map = float(metrics[metric_name])
    initial_map = float(initial_metrics[metric_name])
    constant_map = float(metrics[constant_name])
    reference_map = max(initial_map, constant_map)
    gain = trained_map - reference_map
    first_epoch_loss = _finite_number(history[0].get("train_loss"), "first overfit train_loss")
    final_epoch_loss = _finite_number(history[-1].get("train_loss"), "final overfit train_loss")
    loss_ratio = final_epoch_loss / first_epoch_loss if first_epoch_loss > 0.0 else float("inf")
    mae_ratio = float(fit_report["score_mae"]) / float(initial_fit_report["score_mae"])
    positive_mae_ratio = float(fit_report["positive_score_mae"]) / float(
        initial_fit_report["positive_score_mae"]
    )
    failures = []
    if gain < args.min_overfit_map_gain:
        failures.append(
            f"proxy-mAP gain over the stronger untrained/constant baseline is {gain:.2f}pp, "
            f"below {args.min_overfit_map_gain:.2f}pp"
        )
    if trained_map < args.min_overfit_good_map:
        failures.append(
            f"absolute Good proxy-mAP {trained_map:.2f} is below "
            f"{args.min_overfit_good_map:.2f}"
        )
    if loss_ratio > args.max_overfit_loss_ratio:
        failures.append(
            f"final/first-epoch train-loss ratio {loss_ratio:.4f} exceeds "
            f"{args.max_overfit_loss_ratio:.4f}"
        )
    if mae_ratio > args.max_overfit_mae_ratio:
        failures.append(
            f"trained/untrained dense MAE ratio {mae_ratio:.4f} exceeds "
            f"{args.max_overfit_mae_ratio:.4f}"
        )
    if positive_mae_ratio > args.max_overfit_mae_ratio:
        failures.append(
            f"trained/untrained positive-bin MAE ratio {positive_mae_ratio:.4f} exceeds "
            f"{args.max_overfit_mae_ratio:.4f}"
        )
    if int(curve_report["distinct_curves"]) < 2:
        failures.append("fewer than two distinct curves were produced")
    if float(curve_report["mean_within_curve_variance"]) <= 1e-6:
        failures.append("predicted saliency curves are effectively flat")
    report = {
        "status": "failed" if failures else "passed",
        "source_ids": list(OVERFIT_SOURCE_IDS),
        "optimizer_steps": args.overfit_steps,
        "epochs": epochs,
        "checkpoint": str(overfit_dir / "checkpoints" / "latest.pt"),
        "checkpoint_identity": checkpoint_identity,
        "trainer_prediction_artifact": str(trainer_prediction_path),
        "gate_prediction_artifact": str(work_dir / "overfit-predictions.json"),
        "initial_prediction_artifact": str(work_dir / "overfit-initial-predictions.json"),
        "metrics": metrics,
        "untrained_metrics": initial_metrics,
        "fit": fit_report,
        "untrained_fit": initial_fit_report,
        "first_epoch_train_loss": first_epoch_loss,
        "final_epoch_train_loss": final_epoch_loss,
        "final_to_first_epoch_loss_ratio": loss_ratio,
        "maximum_loss_ratio": args.max_overfit_loss_ratio,
        "trained_to_untrained_mae_ratio": mae_ratio,
        "trained_to_untrained_positive_mae_ratio": positive_mae_ratio,
        "maximum_mae_ratio": args.max_overfit_mae_ratio,
        "trained_good_proxy_map": trained_map,
        "untrained_good_proxy_map": initial_map,
        "constant_good_proxy_map": constant_map,
        "map_reference": "max(untrained, constant)",
        "proxy_map_gain_percentage_points": gain,
        "required_proxy_map_gain_percentage_points": args.min_overfit_map_gain,
        "minimum_absolute_good_proxy_map": args.min_overfit_good_map,
        "curve_diagnostics": curve_report,
        "failures": failures,
    }
    _atomic_json(work_dir / "overfit-gate.json", report)
    if failures:
        raise GateFailure("Four-real-video overfit gate failed: " + "; ".join(failures))
    return model, config, dataset, artifacts


def _mean_absolute_delta(left: list[float], right: list[float]) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        raise GateFailure("Conditioning comparison received incompatible saliency logits.")
    deltas = [abs(a - b) for a, b in zip(left, right)]
    if not all(math.isfinite(value) for value in deltas):
        raise GateFailure("Conditioning comparison produced non-finite deltas.")
    return sum(deltas) / len(deltas), max(deltas)


def _run_conditioning_gate(
    args: argparse.Namespace,
    work_dir: Path,
    model: TinyTraceModel,
    dataset: JsonTinyTraceDataset,
    device: torch.device,
) -> None:
    if not math.isfinite(args.min_conditioning_delta) or args.min_conditioning_delta <= 0.0:
        raise GateFailure("--min-conditioning-delta must be finite and positive.")
    if (
        not math.isfinite(args.min_conditioning_score_delta)
        or args.min_conditioning_score_delta <= 0.0
    ):
        raise GateFailure("--min-conditioning-score-delta must be finite and positive.")
    if (
        not math.isfinite(args.min_conditioning_map_drop)
        or not 0.0 < args.min_conditioning_map_drop <= 100.0
    ):
        raise GateFailure("--min-conditioning-map-drop must be finite and in (0, 100].")
    samples = [dataset[index] for index in range(len(dataset))]
    cases = []
    aggregate: dict[str, list[float]] = {"query_swap": [], "video_swap": [], "zero_visual": []}
    aggregate_score_deltas: dict[str, list[float]] = {
        "query_swap": [],
        "video_swap": [],
        "zero_visual": [],
    }
    evaluation_artifacts: dict[str, list[dict]] = {
        "matched": [],
        "query_swap": [],
        "video_swap": [],
        "zero_visual": [],
    }
    for index, base in enumerate(samples):
        swapped = samples[(index + 1) % len(samples)]
        base_logits, base_scores = _dense_forward(model, base, base, device)
        query_logits, query_scores = _dense_forward(model, base, swapped, device)
        video_logits, video_scores = _dense_forward(model, swapped, base, device)
        zero_logits, zero_scores = _dense_forward(model, base, base, device, zero_visual=True)
        variants = {
            "query_swap": (query_logits, query_scores),
            "video_swap": (video_logits, video_scores),
            "zero_visual": (zero_logits, zero_scores),
        }
        targets = base["saliency_targets"].float().tolist()
        common_artifact = {
            "source_id": base.get("source_id"),
            "qid": base.get("source_id"),
            "task_mode": "highlight",
            "qvh_mean_score_targets": targets,
        }
        evaluation_artifacts["matched"].append(
            {**common_artifact, "pred_saliency_scores": base_scores}
        )
        deltas = {}
        for name, (logits, scores) in variants.items():
            mean_delta, max_delta = _mean_absolute_delta(base_logits, logits)
            mean_score_delta, max_score_delta = _mean_absolute_delta(base_scores, scores)
            aggregate[name].append(mean_delta)
            aggregate_score_deltas[name].append(mean_score_delta)
            evaluation_artifacts[name].append(
                {**common_artifact, "pred_saliency_scores": scores}
            )
            deltas[name] = {
                "mean_absolute_logit_delta": mean_delta,
                "max_absolute_logit_delta": max_delta,
                "mean_absolute_score_delta": mean_score_delta,
                "max_absolute_score_delta": max_score_delta,
            }
        cases.append(
            {
                "source_id": base.get("source_id"),
                "query": base.get("query"),
                "cyclic_source_id": swapped.get("source_id"),
                "cyclic_query": swapped.get("query"),
                "deltas": deltas,
                "curves": {
                    "base": base_scores,
                    "query_swap": query_scores,
                    "video_swap": video_scores,
                    "zero_visual": zero_scores,
                },
            }
        )

    aggregate_means = {name: sum(values) / len(values) for name, values in aggregate.items()}
    aggregate_score_means = {
        name: sum(values) / len(values)
        for name, values in aggregate_score_deltas.items()
    }
    counterfactual_metrics = {
        name: evaluate_event_predictions(artifacts)
        for name, artifacts in evaluation_artifacts.items()
    }
    metric_name = "qvh_mean_score_proxy_Good_mAP"
    matched_map = float(counterfactual_metrics["matched"][metric_name])
    map_drops = {
        name: matched_map - float(counterfactual_metrics[name][metric_name])
        for name in ("video_swap", "zero_visual")
    }
    failures = [
        f"{name} mean logit delta {value:.8g} is below {args.min_conditioning_delta:.8g}"
        for name, value in aggregate_means.items()
        if value < args.min_conditioning_delta
    ]
    failures.extend(
        f"{name} mean score delta {value:.8g} is below "
        f"{args.min_conditioning_score_delta:.8g}"
        for name, value in aggregate_score_means.items()
        if value < args.min_conditioning_score_delta
    )
    failures.extend(
        f"{name} Good proxy-mAP drop {value:.2f}pp is below "
        f"{args.min_conditioning_map_drop:.2f}pp"
        for name, value in map_drops.items()
        if value < args.min_conditioning_map_drop
    )
    report = {
        "status": "failed" if failures else "passed",
        "protocol": {
            "query_swap": "keep video i; use prompt/query (i+1) mod 4",
            "video_swap": "keep prompt/query i; use video/features (i+1) mod 4",
            "zero_visual": "keep query/frame times i; replace cached MobileCLIP features with zeros",
        },
        "minimum_required_mean_absolute_logit_delta": args.min_conditioning_delta,
        "aggregate_mean_absolute_logit_deltas": aggregate_means,
        "minimum_required_mean_absolute_score_delta": args.min_conditioning_score_delta,
        "aggregate_mean_absolute_score_deltas": aggregate_score_means,
        "counterfactual_metrics": counterfactual_metrics,
        "matched_good_proxy_map": matched_map,
        "counterfactual_good_proxy_map_drops": map_drops,
        "minimum_required_good_proxy_map_drop": args.min_conditioning_map_drop,
        "cases": cases,
        "failures": failures,
    }
    _atomic_json(work_dir / "conditioning-gate.json", report)
    if failures:
        raise GateFailure("Query/video conditioning gate failed: " + "; ".join(failures))


def _all_finite(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    return False


def _run_smoke(args: argparse, work_dir: Path, paths: dict[str, Path], profile: dict) -> None:
    if (
        not math.isfinite(args.max_smoke_clipping_rate)
        or not 0.0 <= args.max_smoke_clipping_rate < 1.0
    ):
        raise GateFailure("--max-smoke-clipping-rate must be finite and in [0, 1).")
    smoke_dir = work_dir / "smoke-100"
    if smoke_dir.exists() and any(smoke_dir.iterdir()):
        raise GateFailure(f"Refusing existing non-empty smoke output: {smoke_dir}")
    command = _training_command(
        dataset_json=TRAIN_JSON,
        val_json=VAL_JSON,
        output_dir=smoke_dir,
        frame_cache=paths["frame_cache"],
        feature_cache=paths["feature_cache"],
        device=args.device,
        epochs=1,
        max_optimizer_steps=SMOKE_OPTIMIZER_STEPS,
        accumulation_steps=int(profile["accumulation_steps"]),
        monitor="qvh_mean_score_proxy_Good_mAP",
        monitor_mode="max",
        prediction_every=1,
        prediction_samples=max(8, int(profile["prediction_samples"])),
        metrics_every=1,
        num_workers=int(profile["num_workers"]),
        log_every=50,
    )
    _run_command(command, stage="smoke-100", work_dir=work_dir)
    summary = _read_json(smoke_dir / "run_summary.json")
    history = _read_json(smoke_dir / "history.json")
    predictions = _read_json(smoke_dir / "predictions" / "epoch-0001.json")
    if not isinstance(history, list) or len(history) != 1:
        raise GateFailure("Smoke run must contain exactly one epoch history record.")
    if not isinstance(predictions, list):
        raise GateFailure("Smoke prediction artifact must be a JSON list.")
    train_metrics = history[0].get("train", {})
    optimizer_steps = int(train_metrics.get("optimizer_steps", 0))
    clipped_steps = int(train_metrics.get("clipped_steps", 0))
    clipping_rate = clipped_steps / optimizer_steps if optimizer_steps else 1.0
    curves = [item.get("pred_saliency_scores") for item in predictions]
    curve_report = _validate_curves(curves, "smoke predictions")
    structured = summary.get("final_structured_metrics")
    failures = []
    if summary.get("global_step") != SMOKE_OPTIMIZER_STEPS:
        failures.append(
            f"global_step={summary.get('global_step')} instead of {SMOKE_OPTIMIZER_STEPS}"
        )
    if optimizer_steps != SMOKE_OPTIMIZER_STEPS:
        failures.append(
            f"epoch optimizer_steps={optimizer_steps} instead of {SMOKE_OPTIMIZER_STEPS}"
        )
    if not isinstance(structured, dict) or not structured or not _all_finite(structured):
        failures.append("structured metrics are absent or non-finite")
    if clipping_rate >= args.max_smoke_clipping_rate:
        failures.append(
            f"gradient clipping rate {clipping_rate:.3f} is not below "
            f"{args.max_smoke_clipping_rate:.3f}"
        )
    if float(curve_report["mean_within_curve_variance"]) <= 1e-8:
        failures.append("prediction curves are flat")
    if int(curve_report["distinct_curves"]) < 2:
        failures.append("validation predictions are not distinct across samples")
    diagnostics = summary.get("final_generation_diagnostics") or {}
    if float(diagnostics.get("dense_distinct_curve_ratio", 0.0)) <= 0.0:
        failures.append("full validation diagnostics report no distinct dense curves")
    report = {
        "status": "failed" if failures else "passed",
        "required_optimizer_steps": SMOKE_OPTIMIZER_STEPS,
        "global_step": summary.get("global_step"),
        "optimizer_steps": optimizer_steps,
        "clipped_steps": clipped_steps,
        "clipping_rate": clipping_rate,
        "maximum_clipping_rate_exclusive": args.max_smoke_clipping_rate,
        "structured_metrics": structured,
        "generation_diagnostics": diagnostics,
        "curve_diagnostics": curve_report,
        "failures": failures,
    }
    _atomic_json(work_dir / "smoke-gate.json", report)
    if failures:
        raise GateFailure("100-step real-data smoke gate failed: " + "; ".join(failures))


def _launch_full(args: argparse, work_dir: Path, profile: dict, paths: dict[str, Path]) -> None:
    full_output = paths["output"]
    _fresh_directory(full_output, "full Phase-A output directory")
    # The runner resolves paths relative to a profile's location.  Absolute
    # runtime paths avoid changing that meaning when the audited profile is
    # copied into the gate directory to apply only the explicit device choice.
    runtime_profile = dict(profile)
    runtime_profile.update(
        {
            "train_script": str(TRAIN_SCRIPT.resolve()),
            "model_config": str(DEFAULT_MODEL_CONFIG.resolve()),
            "train_dataset_json": str(TRAIN_JSON.resolve()),
            "val_dataset_json": str(VAL_JSON.resolve()),
            "output_dir": str(full_output),
            "frame_cache_dir": str(paths["frame_cache"]),
            "visual_feature_cache_dir": str(paths["feature_cache"]),
            "device": args.device,
            "resume": "",
        }
    )
    # Revalidate the exact object which the child runner will consume.
    runtime_profile = TrainingProfile.from_dict(runtime_profile).to_dict()
    runtime_path = work_dir / "full-profile.resolved.json"
    _atomic_json(runtime_path, runtime_profile)
    _run_command(
        [sys.executable, str(PROFILE_RUNNER), "--profile", str(runtime_path)],
        stage="full-training",
        work_dir=work_dir,
    )


def main() -> None:
    args = parse_args()
    work_dir = _fresh_directory(args.work_dir, "Phase-A gate work directory")
    pipeline_report: dict[str, object] = {
        "status": "running",
        "started_unix": time.time(),
        "work_dir": str(work_dir),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    }
    _atomic_json(work_dir / "pipeline.json", pipeline_report)
    try:
        _ensure_phase_a_data(work_dir)
        config, profile, paths, train_rows = _preflight(args, work_dir)
        del config
        if args.preflight_only:
            pipeline_report.update(
                {
                    "status": "passed",
                    "finished_unix": time.time(),
                    "full_training": "not_started_preflight_only",
                }
            )
            _atomic_json(work_dir / "pipeline.json", pipeline_report)
            print("\nPhase-A preflight passed; no cache or training stage was started.", flush=True)
            return
        if not args.skip_full:
            full_output = paths["output"]
            if full_output.exists() and any(full_output.iterdir()):
                raise GateFailure(
                    f"Refusing existing non-empty full Phase-A output: {full_output}. "
                    "Move it aside or version the profile output before running the pipeline."
                )
        subset_path = _write_overfit_subset(train_rows, work_dir)
        # Prove learnability and conditioning after extracting only four clips;
        # a broken model should not make the user wait for the entire cache.
        _precompute_features(
            args,
            work_dir,
            paths,
            dataset_paths=(subset_path,),
            stage="feature-cache-overfit",
            expected_entries=len(OVERFIT_SOURCE_IDS),
        )
        device, _ = _resolve_device(args.device)
        model, _, overfit_dataset, _ = _run_overfit(
            args, work_dir, subset_path, paths, device
        )
        _run_conditioning_gate(args, work_dir, model, overfit_dataset, device)
        del model, overfit_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _precompute_features(
            args,
            work_dir,
            paths,
            dataset_paths=(TRAIN_JSON, VAL_JSON),
            stage="feature-cache-full",
            expected_entries=EXPECTED_OUTPUT_COUNTS["train"]
            + EXPECTED_OUTPUT_COUNTS["val"],
        )
        _run_smoke(args, work_dir, paths, profile)
        if args.skip_full:
            print("\nAll Phase-A gates passed. Full training was skipped by request.", flush=True)
        else:
            _launch_full(args, work_dir, profile, paths)
        pipeline_report.update(
            {
                "status": "passed",
                "finished_unix": time.time(),
                "full_training": "skipped" if args.skip_full else "completed",
            }
        )
        _atomic_json(work_dir / "pipeline.json", pipeline_report)
    except BaseException as exc:
        pipeline_report.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "finished_unix": time.time(),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
        _atomic_json(work_dir / "pipeline.json", pipeline_report)
        raise


if __name__ == "__main__":
    main()
