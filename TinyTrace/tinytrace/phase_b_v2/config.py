from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class Stage0Config:
    """Resolved validation contract for the existing MobileCLIP cache."""

    schema_version: str = "tinytrace.phase-b-v2.stage0.config.v1"
    expected_cache_format_version: int = 2
    expected_patch_tokens: int = 64
    expected_feature_dim: int = 1024
    expected_storage_dtype: str = "float16"
    expected_frame_times_dtype: str = "float32"
    duration_tolerance_seconds: float = 0.5
    frame_sampling_safety_margin_seconds: float = 0.25
    representative_sample_count: int = 10
    validation_scope: str = "full"
    seed: int = 7
    output_feature_dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.schema_version != "tinytrace.phase-b-v2.stage0.config.v1":
            raise ValueError(f"Unsupported Stage 0 config schema: {self.schema_version}")
        for name in (
            "expected_cache_format_version",
            "expected_patch_tokens",
            "expected_feature_dim",
            "representative_sample_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            not isinstance(self.duration_tolerance_seconds, (int, float))
            or isinstance(self.duration_tolerance_seconds, bool)
            or not math.isfinite(float(self.duration_tolerance_seconds))
            or self.duration_tolerance_seconds < 0
        ):
            raise ValueError("duration_tolerance_seconds must be finite and non-negative.")
        if (
            not isinstance(self.frame_sampling_safety_margin_seconds, (int, float))
            or isinstance(self.frame_sampling_safety_margin_seconds, bool)
            or not math.isfinite(float(self.frame_sampling_safety_margin_seconds))
            or self.frame_sampling_safety_margin_seconds < 0
        ):
            raise ValueError("frame_sampling_safety_margin_seconds must be finite and non-negative.")
        if self.validation_scope not in {"full", "representative_subset"}:
            raise ValueError("validation_scope must be 'full' or 'representative_subset'.")
        if self.expected_storage_dtype not in {"float16", "float32"}:
            raise ValueError("expected_storage_dtype must be float16 or float32.")
        if self.expected_frame_times_dtype != "float32":
            raise ValueError("expected_frame_times_dtype must be float32 for V1 cache compatibility.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer.")
        if self.output_feature_dtype not in {"float16", "float32"}:
            raise ValueError("output_feature_dtype must be float16 or float32.")

    @classmethod
    def from_json(cls, path: str | Path) -> "Stage0Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Stage 0 config must be a JSON object.")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown Stage 0 config fields: {unknown}")
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
