from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .cache import CacheValidationError, load_and_validate_cache, load_cache_mapping
from .config import Stage0Config


MANIFEST_SCHEMA = "tinytrace.phase-b-v2.activitynet-manifest.v1"


class _DuplicateTrackingDict(dict):
    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__()
        self.duplicate_keys: list[str] = []
        for key, value in pairs:
            if key in self:
                self.duplicate_keys.append(key)
            self[key] = value


def _load_annotation_object(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_DuplicateTrackingDict,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"ActivityNet annotations must be a JSON object: {path}")
    return dict(payload), list(getattr(payload, "duplicate_keys", []))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_feature_path(path: Path, cache_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(cache_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _record(
    records: list[dict[str, object]],
    *,
    scope: str,
    split: str,
    video_id: str,
    reason_code: str,
    detail: str,
    event_index: int | None = None,
) -> None:
    item: dict[str, object] = {
        "scope": scope,
        "split": split,
        "video_id": video_id,
        "reason_code": reason_code,
        "detail": detail,
    }
    if event_index is not None:
        item["event_index"] = event_index
    records.append(item)


def _events_for_entry(
    *,
    split: str,
    video_id: str,
    entry: dict[str, Any],
    duration: float,
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    sentences = entry.get("sentences")
    timestamps = entry.get("timestamps")
    if not isinstance(sentences, list) or not isinstance(timestamps, list):
        _record(
            records,
            scope="sample",
            split=split,
            video_id=video_id,
            reason_code="invalid_event_arrays",
            detail="sentences and timestamps must both be lists",
        )
        return []
    if len(sentences) != len(timestamps):
        for index in range(min(len(sentences), len(timestamps)), max(len(sentences), len(timestamps))):
            _record(
                records,
                scope="event",
                split=split,
                video_id=video_id,
                event_index=index,
                reason_code="unpaired_caption_or_timestamp",
                detail=f"sentences={len(sentences)}, timestamps={len(timestamps)}",
            )

    events: list[dict[str, object]] = []
    for index, (sentence, timestamp) in enumerate(zip(sentences, timestamps)):
        caption = " ".join(str(sentence).strip().split())
        if not caption:
            _record(
                records,
                scope="event",
                split=split,
                video_id=video_id,
                event_index=index,
                reason_code="empty_caption",
                detail="caption is empty after whitespace normalization",
            )
            continue
        if not isinstance(timestamp, list) or len(timestamp) != 2:
            _record(
                records,
                scope="event",
                split=split,
                video_id=video_id,
                event_index=index,
                reason_code="invalid_timestamp_shape",
                detail=f"timestamp={timestamp!r}",
            )
            continue
        try:
            start, end = float(timestamp[0]), float(timestamp[1])
        except (TypeError, ValueError):
            start, end = math.nan, math.nan
        if not math.isfinite(start) or not math.isfinite(end):
            reason = "non_finite_event_boundary"
        elif start < 0:
            reason = "negative_event_start"
        elif end <= start:
            reason = "reversed_or_empty_event"
        elif end > duration:
            reason = "event_exceeds_duration"
        else:
            events.append({"start": start, "end": end, "caption": caption})
            continue
        _record(
            records,
            scope="event",
            split=split,
            video_id=video_id,
            event_index=index,
            reason_code=reason,
            detail=f"start={start!r}, end={end!r}, duration={duration!r}",
        )
    events.sort(key=lambda item: (float(item["start"]), float(item["end"]), str(item["caption"])))
    return events


def _atomic_json(path: Path, payload: object, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Stage 0 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, sort_keys=True)
            destination.write("\n")
        Path(temporary_name).replace(path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _git_revision(repository_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"torch": torch.__version__}
    for package in ("torchvision", "numpy", "pytest"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _dataset_statistics(samples: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for name in ("train", "val", "total"):
        selected = samples if name == "total" else [item for item in samples if item["split"] == name]
        durations = [float(item["duration"]) for item in selected]
        frame_counts = [int(item["frame_count"]) for item in selected]
        event_counts = [len(item["events"]) for item in selected]  # type: ignore[arg-type]
        result[name] = {
            "samples": len(selected),
            "events": sum(event_counts),
            "duration_seconds_mean": sum(durations) / len(durations) if durations else 0.0,
            "duration_seconds_min": min(durations) if durations else 0.0,
            "duration_seconds_max": max(durations) if durations else 0.0,
            "frames_mean": sum(frame_counts) / len(frame_counts) if frame_counts else 0.0,
            "frames_min": min(frame_counts) if frame_counts else 0,
            "frames_max": max(frame_counts) if frame_counts else 0,
            "events_per_video_mean": sum(event_counts) / len(event_counts) if event_counts else 0.0,
            "events_per_video_min": min(event_counts) if event_counts else 0,
            "events_per_video_max": max(event_counts) if event_counts else 0,
        }
    return result


def prepare_stage0(
    *,
    train_annotations: str | Path,
    val_annotations: str | Path,
    cache_mapping: str | Path,
    cache_root: str | Path,
    output_root: str | Path,
    config: Stage0Config,
    repository_root: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build and validate the immutable Stage 0 manifest and reports."""

    train_path = Path(train_annotations).resolve()
    val_path = Path(val_annotations).resolve()
    mapping_path = Path(cache_mapping).resolve()
    root = Path(cache_root).resolve()
    destination = Path(output_root).resolve()
    mapping = load_cache_mapping(mapping_path, root)
    train_payload, train_duplicates = _load_annotation_object(train_path)
    val_payload, val_duplicates = _load_annotation_object(val_path)
    split_payloads = {"train": train_payload, "val": val_payload}
    duplicate_ids = {"train": set(train_duplicates), "val": set(val_duplicates)}
    overlap = sorted(set(train_payload) & set(val_payload))
    overlap_set = set(overlap)

    records: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    cache_eligible_ids: set[str] = set()
    for split, payload in split_payloads.items():
        for video_id, entry in sorted(payload.items()):
            if video_id in duplicate_ids[split]:
                _record(
                    records,
                    scope="sample",
                    split=split,
                    video_id=video_id,
                    reason_code="duplicate_video_id",
                    detail="video ID occurs more than once in the annotation JSON object",
                )
                continue
            if video_id in overlap_set:
                _record(
                    records,
                    scope="sample",
                    split=split,
                    video_id=video_id,
                    reason_code="train_validation_leakage",
                    detail="video ID occurs in both train and validation annotations",
                )
                continue
            if not isinstance(entry, dict):
                _record(
                    records,
                    scope="sample",
                    split=split,
                    video_id=video_id,
                    reason_code="invalid_annotation_entry",
                    detail="annotation entry is not an object",
                )
                continue
            try:
                duration = float(entry.get("duration"))
            except (TypeError, ValueError):
                duration = math.nan
            if not math.isfinite(duration) or duration <= 0:
                _record(
                    records,
                    scope="sample",
                    split=split,
                    video_id=video_id,
                    reason_code="invalid_duration",
                    detail=f"duration={entry.get('duration')!r}",
                )
                continue
            events = _events_for_entry(
                split=split,
                video_id=video_id,
                entry=entry,
                duration=duration,
                records=records,
            )
            if not events:
                if not any(
                    record["scope"] == "sample"
                    and record["split"] == split
                    and record["video_id"] == video_id
                    for record in records
                ):
                    _record(
                        records,
                        scope="sample",
                        split=split,
                        video_id=video_id,
                        reason_code="no_valid_events",
                        detail="no valid timestamped captions remain",
                    )
                continue
            cache_eligible_ids.add(video_id)
            feature_path = mapping.get(video_id)
            if feature_path is None:
                _record(
                    records,
                    scope="sample",
                    split=split,
                    video_id=video_id,
                    reason_code="missing_cache_mapping",
                    detail="video ID has no explicit cache mapping entry",
                )
                continue
            try:
                _, _, cache_report = load_and_validate_cache(
                    feature_path,
                    duration=duration,
                    config=config,
                    compute_statistics=False,
                )
            except CacheValidationError as exc:
                _record(
                    records,
                    scope="sample",
                    split=split,
                    video_id=video_id,
                    reason_code=exc.code,
                    detail=exc.detail,
                )
                continue
            samples.append(
                {
                    "video_id": video_id,
                    "duration": duration,
                    "split": split,
                    "events": events,
                    "visual_feature_path": _portable_feature_path(feature_path, root),
                    "frame_count": cache_report["frame_count"],
                }
            )

    samples.sort(key=lambda item: (str(item["split"]), str(item["video_id"])))
    randomizer = random.Random(config.seed)
    sample_indexes = list(range(len(samples)))
    randomizer.shuffle(sample_indexes)
    representative = []
    for index in sorted(sample_indexes[: config.representative_sample_count]):
        item = samples[index]
        feature_path = Path(str(item["visual_feature_path"]))
        if not feature_path.is_absolute():
            feature_path = root / feature_path
        _, _, cache_report = load_and_validate_cache(
            feature_path,
            duration=float(item["duration"]),
            config=config,
            compute_statistics=True,
        )
        representative.append({"video_id": item["video_id"], "split": item["split"], **cache_report})

    retained_by_split = Counter(str(item["split"]) for item in samples)
    skipped_sample_records = [record for record in records if record["scope"] == "sample"]
    skipped_event_records = [record for record in records if record["scope"] == "event"]
    reason_counts = Counter(str(record["reason_code"]) for record in records)
    mapped_annotation_count = len(cache_eligible_ids & set(mapping))
    has_duplicate_annotation_ids = bool(train_duplicates or val_duplicates)
    validation_passed = not overlap and not has_duplicate_annotation_ids and bool(samples)
    ready_for_training = (
        validation_passed
        and config.validation_scope == "full"
        and retained_by_split["train"] > 0
        and retained_by_split["val"] > 0
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "cache_read_only": True,
        "cache_mapping_schema": "tinytrace.phase-b-v2.cache-map.v1",
        "cache_mapping_semantics": "explicit video_id mapping derived on the original V1 cache PC",
        "samples": samples,
    }
    manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    skipped_report = {
        "schema_version": "tinytrace.phase-b-v2.skipped-samples.v1",
        "record_count": len(records),
        "skipped_sample_count": len(skipped_sample_records),
        "skipped_event_count": len(skipped_event_records),
        "counts_by_reason": dict(sorted(reason_counts.items())),
        "records": records,
    }
    validation_report = {
        "schema_version": "tinytrace.phase-b-v2.dataset-validation.v1",
        "validation_passed": validation_passed,
        "ready_for_training": ready_for_training,
        "validation_scope": config.validation_scope,
        "cache_read_only": True,
        "duration_verification": "annotation duration versus cached frame_times; raw media not required",
        "input_samples": {"train": len(train_payload), "val": len(val_payload)},
        "retained_samples": {
            "train": retained_by_split["train"],
            "val": retained_by_split["val"],
            "total": len(samples),
        },
        "mapping_entries": len(mapping),
        "cache_eligible_annotation_ids": len(cache_eligible_ids),
        "mapped_annotation_ids": mapped_annotation_count,
        "annotation_cache_mapping_coverage_percent": (
            100.0 * mapped_annotation_count / len(cache_eligible_ids) if cache_eligible_ids else 0.0
        ),
        "train_validation_overlap": overlap,
        "duplicate_annotation_ids": {
            "train": sorted(set(train_duplicates)),
            "val": sorted(set(val_duplicates)),
        },
        "skipped_sample_count": len(skipped_sample_records),
        "skipped_event_count": len(skipped_event_records),
        "counts_by_reason": dict(sorted(reason_counts.items())),
        "expected_feature_shape": ["T", config.expected_patch_tokens, config.expected_feature_dim],
        "expected_storage_dtype": config.expected_storage_dtype,
        "validated_cache_entries": len(samples),
        "dataset_statistics": _dataset_statistics(samples),
        "representative_cache_statistics": representative,
    }
    repo = Path(repository_root).resolve() if repository_root else Path.cwd().resolve()
    reproducibility = {
        "schema_version": "tinytrace.phase-b-v2.reproducibility.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(repo),
        "seed": config.seed,
        "manifest_sha256": manifest_sha256,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "inputs": {
            "train_annotations": {"path": str(train_path), "sha256": _sha256(train_path)},
            "val_annotations": {"path": str(val_path), "sha256": _sha256(val_path)},
            "cache_mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path)},
            "cache_root": str(root),
        },
    }
    resolved_config = {
        **config.to_dict(),
        "train_annotations": str(train_path),
        "val_annotations": str(val_path),
        "cache_mapping": str(mapping_path),
        "cache_root": str(root),
        "output_root": str(destination),
    }

    outputs = {
        destination / "manifests" / "activitynet_v2_manifest.json": manifest,
        destination / "reports" / "skipped_samples.json": skipped_report,
        destination / "reports" / "dataset_validation.json": validation_report,
        destination / "metadata" / "reproducibility.json": reproducibility,
        destination / "configs" / "resolved_config.json": resolved_config,
    }
    for path, payload in outputs.items():
        _atomic_json(path, payload, overwrite=overwrite)
    return {
        "manifest": manifest,
        "skipped_samples": skipped_report,
        "dataset_validation": validation_report,
        "reproducibility": reproducibility,
        "resolved_config": resolved_config,
    }
