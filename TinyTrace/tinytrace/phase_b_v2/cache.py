from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from .config import Stage0Config


LEGACY_V1_CACHE_FORMAT_VERSION = 2


class CacheValidationError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def legacy_v1_cache_filename(
    *,
    video_path: str | Path,
    num_frames: int,
    image_size: int,
    mobileclip_model_name: str,
    mobileclip_checkpoint_sha256: str,
    mobileclip_apply_normalization: bool,
) -> str:
    """Reproduce V1's exact path/stat-dependent cache filename contract."""

    source = Path(video_path)
    stat = source.stat() if source.exists() else None
    identity = "|".join(
        [
            str(source.resolve()),
            str(stat.st_size if stat else "missing"),
            str(stat.st_mtime_ns if stat else "missing"),
            str(num_frames),
            str(image_size),
            mobileclip_model_name,
            mobileclip_checkpoint_sha256,
            str(mobileclip_apply_normalization),
            str(LEGACY_V1_CACHE_FORMAT_VERSION),
        ]
    )
    return f"{hashlib.sha256(identity.encode('utf-8')).hexdigest()}.pt"


def _require_tensor(payload: dict, key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise CacheValidationError("missing_or_invalid_tensor", f"{key} is not a tensor")
    return value


def load_and_validate_cache(
    path: str | Path,
    *,
    duration: float,
    config: Stage0Config,
    compute_statistics: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Load one read-only V1 cache entry and enforce the V2 Stage 0 contract."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise CacheValidationError("missing_cache", f"cache does not exist: {cache_path}")
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise CacheValidationError("cache_load_error", str(exc)) from exc
    if not isinstance(payload, dict):
        raise CacheValidationError("invalid_cache_payload", "cache payload is not a dictionary")
    if payload.get("format_version") != config.expected_cache_format_version:
        raise CacheValidationError(
            "cache_format_mismatch",
            f"expected {config.expected_cache_format_version}, got {payload.get('format_version')!r}",
        )

    features = _require_tensor(payload, "patch_features")
    frame_times = _require_tensor(payload, "frame_times")
    if not features.is_floating_point():
        raise CacheValidationError("invalid_feature_dtype", f"features use {features.dtype}")
    if str(features.dtype).removeprefix("torch.") != config.expected_storage_dtype:
        raise CacheValidationError(
            "invalid_feature_dtype",
            f"expected {config.expected_storage_dtype}, got {features.dtype}",
        )
    if features.ndim != 3:
        raise CacheValidationError("invalid_feature_rank", f"shape is {list(features.shape)}")
    expected_tail = (config.expected_patch_tokens, config.expected_feature_dim)
    if tuple(features.shape[1:]) != expected_tail:
        raise CacheValidationError(
            "invalid_feature_shape",
            f"expected [T,{expected_tail[0]},{expected_tail[1]}], got {list(features.shape)}",
        )
    if features.size(0) < 1:
        raise CacheValidationError("empty_feature_sequence", "feature sequence has no frames")
    if not frame_times.is_floating_point():
        raise CacheValidationError("invalid_frame_time_dtype", f"times use {frame_times.dtype}")
    if str(frame_times.dtype).removeprefix("torch.") != config.expected_frame_times_dtype:
        raise CacheValidationError(
            "invalid_frame_time_dtype",
            f"expected {config.expected_frame_times_dtype}, got {frame_times.dtype}",
        )
    if frame_times.shape != (features.size(0),):
        raise CacheValidationError(
            "frame_time_shape_mismatch",
            f"feature frames={features.size(0)}, frame_times shape={list(frame_times.shape)}",
        )
    if not torch.isfinite(features).all():
        raise CacheValidationError("non_finite_features", "features contain NaN or Inf")
    if not torch.isfinite(frame_times).all():
        raise CacheValidationError("non_finite_frame_times", "frame_times contain NaN or Inf")
    if frame_times[0].item() < 0:
        raise CacheValidationError("negative_frame_time", f"first time is {frame_times[0].item()}")
    if frame_times.numel() > 1 and (frame_times[1:] < frame_times[:-1]).any():
        raise CacheValidationError("unordered_frame_times", "frame_times are not nondecreasing")
    if not math.isfinite(duration) or duration <= 0:
        raise CacheValidationError("invalid_duration", f"duration is {duration!r}")
    last_time = float(frame_times[-1].item())
    if last_time > duration + config.duration_tolerance_seconds:
        raise CacheValidationError(
            "frame_time_exceeds_duration",
            f"last time {last_time:.6f}s exceeds duration {duration:.6f}s",
        )
    expected_last_time = max(duration - config.frame_sampling_safety_margin_seconds, 0.0)
    if abs(last_time - expected_last_time) > config.duration_tolerance_seconds:
        raise CacheValidationError(
            "frame_time_duration_mismatch",
            (
                f"last frame time {last_time:.6f}s does not match the V1 sampling endpoint "
                f"{expected_last_time:.6f}s for annotation duration {duration:.6f}s"
            ),
        )

    report: dict[str, object] = {
        "path": str(cache_path),
        "shape": list(features.shape),
        "dtype": str(features.dtype),
        "frame_times_dtype": str(frame_times.dtype),
        "frame_count": int(features.size(0)),
        "first_frame_time": float(frame_times[0].item()),
        "last_frame_time": last_time,
        "expected_last_frame_time": expected_last_time,
        "finite": True,
        "nan_count": 0,
        "inf_count": 0,
    }
    if compute_statistics:
        values = features.float()
        frame_vectors = values.flatten(1)
        norms = torch.linalg.vector_norm(frame_vectors, dim=1)
        report.update(
            {
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "l2_norm_mean": float(norms.mean().item()),
                "l2_norm_min": float(norms.min().item()),
                "l2_norm_max": float(norms.max().item()),
                "all_zero": bool(torch.count_nonzero(values).item() == 0),
                "constant": bool(values.min().item() == values.max().item()),
            }
        )
        if features.size(0) > 1:
            normalized = torch.nn.functional.normalize(frame_vectors, dim=1)
            adjacent = (normalized[:-1] * normalized[1:]).sum(dim=1)
            report["adjacent_cosine_mean"] = float(adjacent.mean().item())
            report["first_last_cosine"] = float((normalized[0] * normalized[-1]).sum().item())

    target_dtype = torch.float16 if config.output_feature_dtype == "float16" else torch.float32
    return features.to(target_dtype), frame_times.float(), report


def load_cache_mapping(path: str | Path, cache_root: str | Path) -> dict[str, Path]:
    """Load the portable, explicit mapping required because cache payloads lack IDs."""

    mapping_path = Path(path)
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "tinytrace.phase-b-v2.cache-map.v1":
        raise ValueError("Cache mapping must use schema tinytrace.phase-b-v2.cache-map.v1.")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Cache mapping entries must be a list.")
    root = Path(cache_root)
    result: dict[str, Path] = {}
    used_paths: dict[Path, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Cache mapping entry {index} is not an object.")
        video_id = entry.get("video_id")
        raw_path = entry.get("visual_feature_path")
        if not isinstance(video_id, str) or not video_id.strip():
            raise ValueError(f"Cache mapping entry {index} has an invalid video_id.")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(f"Cache mapping entry {index} has an invalid path.")
        if video_id in result:
            raise ValueError(f"Duplicate video_id in cache mapping: {video_id}")
        candidate = Path(raw_path)
        resolved = candidate if candidate.is_absolute() else root / candidate
        resolved = resolved.resolve()
        if resolved in used_paths:
            raise ValueError(
                f"Cache path is mapped to both {used_paths[resolved]} and {video_id}: {resolved}"
            )
        result[video_id] = resolved
        used_paths[resolved] = video_id
    return result
