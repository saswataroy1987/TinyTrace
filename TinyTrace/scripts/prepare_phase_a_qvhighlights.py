from __future__ import annotations

"""Prepare immutable, dense Phase-A QVHighlights annotations.

The legacy TinyTrace annotations contain compressed score runs.  Those runs are
useful for compact autoregressive experiments, but they are not the official
QVHighlights target: the highlight task is evaluated as one saliency score per
two-second clip.  This script rebuilds that target directly from the source
0.5-second points and never edits the source annotation files.

The direct mapping is ``clip_index = int(source_time / 2)``.  In particular,
this intentionally does *not* copy TRACE's ``int(generated_time / 2) - 1``
reformatter rule.  That offset belongs to TRACE's generated timestamp
convention; applying it to the source labels would shift every ground-truth
clip one position to the left.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


SCHEMA_VERSION = "tinytrace.qvhighlights.phase-a.v3"
BIN_SIZE_SECONDS = 2.0
BIN_COUNT = 75
EXPECTED_DURATION_SECONDS = BIN_SIZE_SECONDS * BIN_COUNT
SOURCE_POINTS_PER_BIN = 4
SOURCE_POINT_STEP_SECONDS = 0.5
MAX_SALIENCY_SCORE = 4.0
DEFAULT_DURATION_TOLERANCE_SECONDS = 0.5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION_DIR = PROJECT_ROOT / "final_qvhighlights_tinytrace" / "annotations"


class AnnotationValidationError(ValueError):
    """Raised when source annotations cannot define an unambiguous target."""


class MediaProbeError(RuntimeError):
    """Raised for a clip-specific ffprobe failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build immutable Phase-A-v3 QVHighlights splits with 75 direct "
            "two-second saliency bins and an explicit media-exclusion audit."
        )
    )
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR / "qvh_raw_valid.json",
        help="Source TRACE/QVHighlights annotations containing times and scores.",
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR / "tinytrace_train.json",
        help="Existing split metadata; this file is read only.",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR / "tinytrace_val.json",
        help="Existing split metadata; this file is read only.",
    )
    parser.add_argument(
        "--output-train-json",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR / "tinytrace_phase_a_v3_train.json",
    )
    parser.add_argument(
        "--output-val-json",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR / "tinytrace_phase_a_v3_val.json",
    )
    parser.add_argument(
        "--exclusions-json",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR / "phase_a_v3_exclusions.json",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR / "phase_a_v3_manifest.json",
    )
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=DEFAULT_DURATION_TOLERANCE_SECONDS,
        help=(
            "Tolerance below the required 150-second QVHighlights window. A "
            "clip shorter than 150 minus this value is excluded."
        ),
    )
    parser.add_argument(
        "--probe-timeout-seconds",
        type=float,
        default=30.0,
        help="Per-video ffprobe timeout.",
    )
    return parser.parse_args()


def _canonical_source_id(value: object, location: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise AnnotationValidationError(
            f"{location} source id must be a string or integer, received {value!r}."
        )
    canonical = str(value).strip()
    if not canonical:
        raise AnnotationValidationError(f"{location} source id must not be empty.")
    return canonical


def _load_json_list(path: Path, label: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationValidationError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, list):
        raise AnnotationValidationError(f"{label} must contain a JSON list: {path}")
    if not all(isinstance(item, dict) for item in payload):
        raise AnnotationValidationError(f"Every {label} entry must be a JSON object: {path}")
    return payload


def _unique_index(items: list[dict], key: str, label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for index, item in enumerate(items):
        if key not in item:
            raise AnnotationValidationError(f"{label}[{index}] is missing {key!r}.")
        canonical = _canonical_source_id(item[key], f"{label}[{index}]")
        if canonical in indexed:
            raise AnnotationValidationError(
                f"Source-id collision in {label}: {item[key]!r} occurs more than once."
            )
        indexed[canonical] = item
    return indexed


def _single_number(row: object, location: str) -> float:
    if not isinstance(row, list) or len(row) != 1:
        raise AnnotationValidationError(
            f"{location} must be a one-value list, received {row!r}."
        )
    value = row[0]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationValidationError(f"{location} must contain a numeric value, received {row!r}.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AnnotationValidationError(f"{location} must be finite, received {parsed!r}.")
    return parsed


def build_dense_saliency(raw_item: dict, source_id: str) -> tuple[list[float], float, int]:
    """Validate 4-point source blocks and aggregate them into 75 direct bins."""

    times = raw_item.get("times")
    scores = raw_item.get("scores")
    if not isinstance(times, list) or not isinstance(scores, list):
        raise AnnotationValidationError(
            f"Raw source {source_id} must provide list-valued times and scores."
        )
    if not times:
        raise AnnotationValidationError(f"Raw source {source_id} has no saliency source blocks.")
    if len(times) != len(scores):
        raise AnnotationValidationError(
            f"Raw source {source_id} has {len(times)} time rows but {len(scores)} score rows."
        )
    if len(times) % SOURCE_POINTS_PER_BIN != 0:
        raise AnnotationValidationError(
            f"Raw source {source_id} has {len(times)} points; expected complete "
            f"{SOURCE_POINTS_PER_BIN}-point source blocks."
        )

    dense = [0.0] * BIN_COUNT
    occupied_bins: set[int] = set()
    maximum_time = 0.0

    for block_start in range(0, len(times), SOURCE_POINTS_PER_BIN):
        block_number = block_start // SOURCE_POINTS_PER_BIN
        block_times = [
            _single_number(times[block_start + offset], f"raw[{source_id}].times[{block_start + offset}]")
            for offset in range(SOURCE_POINTS_PER_BIN)
        ]
        block_scores = [
            _single_number(scores[block_start + offset], f"raw[{source_id}].scores[{block_start + offset}]")
            for offset in range(SOURCE_POINTS_PER_BIN)
        ]

        first_time = block_times[0]
        if first_time < 0.0:
            raise AnnotationValidationError(
                f"Raw source {source_id} block {block_number} starts before zero: {first_time}."
            )
        clip_index = int(first_time / BIN_SIZE_SECONDS)
        if not 0 <= clip_index < BIN_COUNT:
            raise AnnotationValidationError(
                f"Raw source {source_id} block {block_number} maps to clip {clip_index}; "
                f"valid indices are 0..{BIN_COUNT - 1}."
            )

        expected_start = clip_index * BIN_SIZE_SECONDS
        expected_times = [
            expected_start + offset * SOURCE_POINT_STEP_SECONDS
            for offset in range(SOURCE_POINTS_PER_BIN)
        ]
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
            for actual, expected in zip(block_times, expected_times)
        ):
            raise AnnotationValidationError(
                f"Raw source {source_id} block {block_number} is not an aligned four-point "
                f"2-second source block: times={block_times}, expected={expected_times}."
            )
        if clip_index in occupied_bins:
            raise AnnotationValidationError(
                f"Raw source {source_id} has multiple source blocks mapping to clip {clip_index}."
            )
        if any(score < 0.0 or score > MAX_SALIENCY_SCORE for score in block_scores):
            raise AnnotationValidationError(
                f"Raw source {source_id} block {block_number} has a score outside [0, 4]: "
                f"{block_scores}."
            )
        if any(
            not math.isclose(score, block_scores[0], rel_tol=0.0, abs_tol=1e-6)
            for score in block_scores[1:]
        ):
            raise AnnotationValidationError(
                f"Raw source {source_id} block {block_number} has inconsistent scores across "
                f"its four source points: {block_scores}."
            )

        dense[clip_index] = round(sum(block_scores) / SOURCE_POINTS_PER_BIN, 6)
        occupied_bins.add(clip_index)
        maximum_time = max(maximum_time, block_times[-1])

    return dense, maximum_time, len(occupied_bins)


def _resolve_video_path(video_path: str, split_json_path: Path) -> Path:
    source = Path(video_path).expanduser()
    if source.is_absolute():
        return source.resolve(strict=False)

    annotation_dir = split_json_path.resolve().parent
    candidates = (
        annotation_dir.parent / source,
        annotation_dir / source,
        annotation_dir.parent.parent / source,
        Path.cwd() / source,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    # The first candidate is the dataset-relative contract used by the current
    # final_qvhighlights_tinytrace annotations.  Returning it also makes a
    # missing-media exclusion precise and reproducible.
    return candidates[0].resolve(strict=False)


def probe_media_duration(video_path: Path, timeout_seconds: float = 30.0) -> float:
    """Return the shortest positive stream/format duration reported by ffprobe."""

    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=duration:format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe is required for Phase-A media validation but was not found on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(f"ffprobe timed out after {timeout_seconds:g}s") from exc

    if probe.returncode != 0:
        detail = probe.stderr.strip() or f"ffprobe exited with status {probe.returncode}"
        raise MediaProbeError(detail)
    try:
        payload = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe returned invalid JSON: {exc}") from exc

    candidates: list[float] = []
    for stream in payload.get("streams", []):
        value = stream.get("duration") if isinstance(stream, dict) else None
        if value not in (None, "", "N/A"):
            try:
                candidates.append(float(value))
            except (TypeError, ValueError):
                continue
    format_payload = payload.get("format", {})
    format_value = format_payload.get("duration") if isinstance(format_payload, dict) else None
    if format_value not in (None, "", "N/A"):
        try:
            candidates.append(float(format_value))
        except (TypeError, ValueError):
            pass

    positive = [value for value in candidates if math.isfinite(value) and value > 0.0]
    if not positive:
        raise MediaProbeError("ffprobe did not report a finite positive video duration")
    return min(positive)


def _validate_processed_item(item: dict, split: str, index: int) -> None:
    location = f"{split}[{index}]"
    required_strings = ("video_path", "instruction", "query", "task_mode")
    for field in required_strings:
        value = item.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AnnotationValidationError(
                f"{location}.{field} must be a non-empty string, received {value!r}."
            )
    if item["task_mode"] != "highlight":
        raise AnnotationValidationError(
            f"{location}.task_mode must be 'highlight' for Phase A, received {item['task_mode']!r}."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _write_and_fsync(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_atomically(payloads: dict[Path, bytes], manifest_path: Path) -> None:
    """Publish immutable files, with the manifest linked last as the commit marker."""

    outputs = list(payloads)
    if len({path.resolve(strict=False) for path in outputs}) != len(outputs):
        raise ValueError("Phase-A output paths must be distinct.")
    parents = {path.resolve(strict=False).parent for path in outputs}
    if len(parents) != 1:
        raise ValueError("All Phase-A output files must share one directory for atomic publication.")
    for path in outputs:
        if path.exists():
            raise FileExistsError(
                f"Immutable Phase-A output already exists: {path}. "
                "Choose new versioned output names instead of overwriting it."
            )

    output_dir = next(iter(parents))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".phase-a-v3-stage-", dir=output_dir))
    staged: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for destination, payload in payloads.items():
            stage_path = stage_dir / destination.name
            _write_and_fsync(stage_path, payload)
            staged[destination] = stage_path

        publication_order = [path for path in outputs if path != manifest_path] + [manifest_path]
        for destination in publication_order:
            # Hard-link publication is atomic and refuses a race-created
            # destination instead of replacing it.
            os.link(staged[destination], destination)
            published.append(destination)
        directory_fd = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def prepare_phase_a_dataset(
    *,
    raw_json: Path,
    train_json: Path,
    val_json: Path,
    output_train_json: Path,
    output_val_json: Path,
    exclusions_json: Path,
    manifest_json: Path,
    duration_tolerance_seconds: float = DEFAULT_DURATION_TOLERANCE_SECONDS,
    probe_timeout_seconds: float = 30.0,
    probe_fn: Callable[[Path, float], float] | None = None,
) -> dict:
    if (
        not math.isfinite(duration_tolerance_seconds)
        or duration_tolerance_seconds < 0.0
        or duration_tolerance_seconds >= EXPECTED_DURATION_SECONDS
    ):
        raise ValueError(
            "duration_tolerance_seconds must be finite, non-negative, and below "
            f"{EXPECTED_DURATION_SECONDS:g}."
        )
    if not math.isfinite(probe_timeout_seconds) or probe_timeout_seconds <= 0.0:
        raise ValueError("probe_timeout_seconds must be finite and positive.")

    input_paths = [raw_json, train_json, val_json]
    output_paths = [output_train_json, output_val_json, exclusions_json, manifest_json]
    input_resolved = {path.resolve(strict=False) for path in input_paths}
    if any(path.resolve(strict=False) in input_resolved for path in output_paths):
        raise ValueError("An output path must not replace a source annotation file.")
    # Fail before the expensive media scan when this immutable version already
    # exists.  _publish_atomically repeats the check to close the race window.
    for path in output_paths:
        if path.exists():
            raise FileExistsError(
                f"Immutable Phase-A output already exists: {path}. "
                "Choose new versioned output names instead of overwriting it."
            )

    raw_items = _load_json_list(raw_json, "raw annotations")
    train_items = _load_json_list(train_json, "train split")
    val_items = _load_json_list(val_json, "validation split")

    raw_by_id = _unique_index(raw_items, "id", "raw annotations")
    train_by_id = _unique_index(train_items, "source_id", "train split")
    val_by_id = _unique_index(val_items, "source_id", "validation split")
    overlap = sorted(set(train_by_id).intersection(val_by_id))
    if overlap:
        raise AnnotationValidationError(
            f"Train/validation source-id overlap is forbidden; examples: {overlap[:10]}."
        )
    train_paths = {str(item.get("video_path", "")) for item in train_items}
    val_paths = {str(item.get("video_path", "")) for item in val_items}
    path_overlap = sorted(path for path in train_paths.intersection(val_paths) if path)
    if path_overlap:
        raise AnnotationValidationError(
            f"Train/validation video-path overlap is forbidden; examples: {path_overlap[:10]}."
        )

    split_ids = set(train_by_id).union(val_by_id)
    missing_raw = sorted(split_ids.difference(raw_by_id))
    unused_raw = sorted(set(raw_by_id).difference(split_ids))
    if missing_raw or unused_raw:
        raise AnnotationValidationError(
            "Raw/split source-id sets must match exactly: "
            f"missing_raw={missing_raw[:10]} unused_raw={unused_raw[:10]}."
        )

    # Validate every annotation before probing media so malformed labels never
    # get silently hidden inside the exclusions report.
    target_by_id: dict[str, tuple[list[float], float, int]] = {}
    for source_id, raw_item in raw_by_id.items():
        target_by_id[source_id] = build_dense_saliency(raw_item, source_id)

    probe = probe_fn or probe_media_duration
    prepared: dict[str, list[dict]] = {"train": [], "val": []}
    exclusions: list[dict] = []

    for split, items, split_path in (
        ("train", train_items, train_json),
        ("val", val_items, val_json),
    ):
        for index, item in enumerate(items):
            _validate_processed_item(item, split, index)
            source_id = _canonical_source_id(item["source_id"], f"{split}[{index}]")
            raw_item = raw_by_id[source_id]
            raw_video = raw_item.get("video")
            if not isinstance(raw_video, str) or not raw_video.strip():
                raise AnnotationValidationError(f"Raw source {source_id} has no valid video path.")
            if Path(raw_video).name != Path(item["video_path"]).name:
                raise AnnotationValidationError(
                    f"Video collision/mismatch for source {source_id}: raw={raw_video!r}, "
                    f"split={item['video_path']!r}."
                )

            dense_scores, maximum_source_time, source_block_count = target_by_id[source_id]
            resolved_video = _resolve_video_path(item["video_path"], split_path)
            common_exclusion = {
                "source_id": item["source_id"],
                "split": split,
                "video_path": item["video_path"],
                "resolved_video_path": str(resolved_video),
                "maximum_source_timestamp_seconds": maximum_source_time,
            }
            if not resolved_video.is_file():
                exclusions.append({**common_exclusion, "reason": "missing_media"})
                continue

            try:
                duration = float(probe(resolved_video, probe_timeout_seconds))
            except RuntimeError as exc:
                if isinstance(exc, MediaProbeError):
                    exclusions.append(
                        {**common_exclusion, "reason": "probe_failed", "detail": str(exc)}
                    )
                    continue
                raise
            except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
                exclusions.append(
                    {**common_exclusion, "reason": "probe_failed", "detail": str(exc)}
                )
                continue
            if not math.isfinite(duration) or duration <= 0.0:
                exclusions.append(
                    {
                        **common_exclusion,
                        "reason": "probe_failed",
                        "detail": f"invalid probed duration {duration!r}",
                    }
                )
                continue
            minimum_valid_duration = EXPECTED_DURATION_SECONDS - duration_tolerance_seconds
            if duration < minimum_valid_duration:
                exclusions.append(
                    {
                        **common_exclusion,
                        "reason": "truncated_media",
                        "duration_seconds": round(duration, 6),
                        "expected_duration_seconds": EXPECTED_DURATION_SECONDS,
                        "minimum_valid_duration_seconds": minimum_valid_duration,
                        "duration_tolerance_seconds": duration_tolerance_seconds,
                        "detail": "media is shorter than the required QVHighlights window",
                    }
                )
                continue
            if maximum_source_time > duration + duration_tolerance_seconds:
                exclusions.append(
                    {
                        **common_exclusion,
                        "reason": "annotation_exceeds_media",
                        "duration_seconds": round(duration, 6),
                        "expected_duration_seconds": EXPECTED_DURATION_SECONDS,
                        "duration_tolerance_seconds": duration_tolerance_seconds,
                        "detail": "a source timestamp lies beyond the decoded media duration",
                    }
                )
                continue

            prepared[split].append(
                {
                    "source_id": item["source_id"],
                    "video_path": item["video_path"],
                    "instruction": item["instruction"],
                    "query": item["query"],
                    "task_mode": item["task_mode"],
                    "duration_seconds": round(duration, 6),
                    "saliency_bin_size_seconds": BIN_SIZE_SECONDS,
                    "saliency_bin_count": BIN_COUNT,
                    "dense_saliency_scores": list(dense_scores),
                    "source_saliency_block_count": source_block_count,
                }
            )

    if not prepared["train"] or not prepared["val"]:
        raise RuntimeError(
            "Media validation removed an entire split; no Phase-A outputs were published."
        )

    exclusions_payload = {
        "schema_version": SCHEMA_VERSION,
        "duration_tolerance_seconds": duration_tolerance_seconds,
        "count": len(exclusions),
        "reason_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
        "items": exclusions,
    }
    train_bytes = _json_bytes(prepared["train"])
    val_bytes = _json_bytes(prepared["val"])
    exclusions_bytes = _json_bytes(exclusions_payload)
    output_hashes = {
        output_train_json.name: hashlib.sha256(train_bytes).hexdigest(),
        output_val_json.name: hashlib.sha256(val_bytes).hexdigest(),
        exclusions_json.name: hashlib.sha256(exclusions_bytes).hexdigest(),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable": True,
        "source_files": {
            "raw_json": {"path": str(raw_json.resolve()), "sha256": _sha256(raw_json)},
            "train_json": {"path": str(train_json.resolve()), "sha256": _sha256(train_json)},
            "val_json": {"path": str(val_json.resolve()), "sha256": _sha256(val_json)},
        },
        "target_contract": {
            "task": "QVHighlights Phase A dense saliency",
            "bin_size_seconds": BIN_SIZE_SECONDS,
            "bin_count": BIN_COUNT,
            "duration_seconds": EXPECTED_DURATION_SECONDS,
            "source_points_per_bin": SOURCE_POINTS_PER_BIN,
            "source_point_step_seconds": SOURCE_POINT_STEP_SECONDS,
            "source_to_bin_mapping": "int(source_timestamp_seconds / 2.0)",
            "missing_source_bins": "0.0",
            "trace_offset_warning": (
                "Do not apply TRACE reformat_vhd.py's int(generated_time / 2) - 1 rule "
                "to these source-derived targets. Direct official clip indices have no -1 offset."
            ),
        },
        "media_validation": {
            "probe": "ffprobe stream=duration:format=duration",
            "expected_duration_seconds": EXPECTED_DURATION_SECONDS,
            "minimum_valid_duration_seconds": (
                EXPECTED_DURATION_SECONDS - duration_tolerance_seconds
            ),
            "duration_tolerance_seconds": duration_tolerance_seconds,
            "policy": (
                "Exclude missing/probe-failed media, every clip shorter than the expected "
                "150-second QVHighlights window minus tolerance, and any remaining clip "
                "whose source timestamp exceeds decoded duration plus tolerance."
            ),
        },
        "counts": {
            "raw": len(raw_items),
            "input_train": len(train_items),
            "input_val": len(val_items),
            "output_train": len(prepared["train"]),
            "output_val": len(prepared["val"]),
            "excluded_total": len(exclusions),
            "excluded_by_reason": exclusions_payload["reason_counts"],
        },
        "outputs": {
            output_train_json.name: {
                "path": str(output_train_json.resolve(strict=False)),
                "sha256": output_hashes[output_train_json.name],
            },
            output_val_json.name: {
                "path": str(output_val_json.resolve(strict=False)),
                "sha256": output_hashes[output_val_json.name],
            },
            exclusions_json.name: {
                "path": str(exclusions_json.resolve(strict=False)),
                "sha256": output_hashes[exclusions_json.name],
            },
        },
    }
    manifest_bytes = _json_bytes(manifest)
    _publish_atomically(
        {
            output_train_json: train_bytes,
            output_val_json: val_bytes,
            exclusions_json: exclusions_bytes,
            manifest_json: manifest_bytes,
        },
        manifest_json,
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_phase_a_dataset(
        raw_json=args.raw_json,
        train_json=args.train_json,
        val_json=args.val_json,
        output_train_json=args.output_train_json,
        output_val_json=args.output_val_json,
        exclusions_json=args.exclusions_json,
        manifest_json=args.manifest_json,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
        probe_timeout_seconds=args.probe_timeout_seconds,
    )
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Published immutable Phase-A-v3 manifest: {args.manifest_json}")


if __name__ == "__main__":
    main()
