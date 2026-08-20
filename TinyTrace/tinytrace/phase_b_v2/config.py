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
        # A generated resolved config includes input provenance alongside the
        # validation contract. Allow only those known metadata fields so it can
        # be used directly by the dataset loader without weakening validation.
        resolved_metadata = {
            "train_annotations",
            "val_annotations",
            "cache_mapping",
            "cache_root",
            "output_root",
        }
        unknown = sorted(set(payload) - allowed - resolved_metadata)
        if unknown:
            raise ValueError(f"Unknown Stage 0 config fields: {unknown}")
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseBV2Config:
    """Versioned detector and captioner contract for V2 Stages 1 and 2."""

    schema_version: str = "tinytrace.phase-b-v2.model.config.v1"
    stage: str = "localization"
    feature_dim: int = 1024
    patch_tokens: int = 64
    d_model: int = 256
    temporal_layers: int = 4
    temporal_heads: int = 8
    dropout: float = 0.1
    event_queries: int = 32
    event_decoder_layers: int = 2
    multiscale_kernel_size: int = 3
    matcher_class_cost: float = 1.0
    matcher_l1_cost: float = 5.0
    matcher_giou_cost: float = 2.0
    no_event_weight: float = 0.2
    loss_event_weight: float = 1.0
    loss_l1_weight: float = 5.0
    loss_giou_weight: float = 2.0
    loss_caption_weight: float = 1.0
    joint_ground_truth_segment_ratio: float = 0.5
    flan_model_name: str = "google/flan-t5-small"
    flan_revision: str = "main"
    flan_local_files_only: bool = False
    conditioning_tokens: int = 4
    caption_max_tokens: int = 64
    generation_max_tokens: int = 64
    generation_min_tokens: int = 3
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    train_flan_encoder_layers: int = 1
    train_flan_decoder_layers: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != "tinytrace.phase-b-v2.model.config.v1":
            raise ValueError(f"Unsupported V2 model config schema: {self.schema_version}")
        if self.stage not in {"localization", "caption", "joint"}:
            raise ValueError("stage must be 'localization', 'caption', or 'joint'.")
        for name in ("feature_dim", "patch_tokens", "d_model", "temporal_layers", "temporal_heads", "event_queries", "event_decoder_layers", "multiscale_kernel_size", "conditioning_tokens", "caption_max_tokens", "generation_max_tokens", "generation_min_tokens", "no_repeat_ngram_size", "train_flan_encoder_layers", "train_flan_decoder_layers"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.d_model % self.temporal_heads:
            raise ValueError("d_model must be divisible by temporal_heads.")
        for name in ("dropout", "matcher_class_cost", "matcher_l1_cost", "matcher_giou_cost", "loss_event_weight", "loss_l1_weight", "loss_giou_weight", "loss_caption_weight", "repetition_penalty"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not isinstance(self.no_event_weight, (int, float)) or not math.isfinite(float(self.no_event_weight)) or not 0 < float(self.no_event_weight) <= 1:
            raise ValueError("no_event_weight must be in (0, 1].")
        if not isinstance(self.joint_ground_truth_segment_ratio, (int, float)) or not math.isfinite(float(self.joint_ground_truth_segment_ratio)) or not 0 <= float(self.joint_ground_truth_segment_ratio) <= 1:
            raise ValueError("joint_ground_truth_segment_ratio must be in [0, 1].")
        if self.generation_min_tokens > self.generation_max_tokens:
            raise ValueError("generation_min_tokens cannot exceed generation_max_tokens.")
        if not isinstance(self.flan_model_name, str) or not self.flan_model_name.strip():
            raise ValueError("flan_model_name must be non-empty.")

    @classmethod
    def from_json(cls, path: str | Path) -> "PhaseBV2Config":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("V2 model config must be a JSON object.")
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown V2 model config fields: {unknown}")
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
