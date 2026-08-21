"""Configuration contract for the isolated Stage 2 v3 experiment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class DirectMobileCLIPCaptionConfig:
    """Direct cached-patch conditioning for FLAN-T5 Small.

    MobileCLIP remains frozen: this config describes only the new visual
    resampler/bridge and the deliberately limited FLAN-T5 fine tuning policy.
    """

    schema_version: str = "tinytrace.phase-b-v3.direct-mobileclip.config.v1"
    feature_dim: int = 1024
    patch_tokens: int = 64
    max_event_frames: int = 8
    visual_tokens: int = 16
    visual_heads: int = 8
    visual_dropout: float = 0.1
    flan_model_name: str = "google/flan-t5-small"
    flan_revision: str = "main"
    flan_local_files_only: bool = True
    instruction: str = "Describe the event shown in the video."
    caption_max_tokens: int = 64
    generation_max_tokens: int = 64
    generation_min_tokens: int = 3
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    train_flan_encoder_layers: int = 1
    train_flan_decoder_layers: int = 1
    use_stage1_temporal_context: bool = False
    temporal_context_tokens: int = 2
    stage1_context_dim: int = 256
    train_stage1_temporal_layers: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != "tinytrace.phase-b-v3.direct-mobileclip.config.v1":
            raise ValueError(f"Unsupported Stage 2 v3 config schema: {self.schema_version}")
        for name in (
            "feature_dim", "patch_tokens", "max_event_frames", "visual_tokens", "visual_heads",
            "caption_max_tokens", "generation_max_tokens", "generation_min_tokens",
            "no_repeat_ngram_size", "train_flan_encoder_layers", "train_flan_decoder_layers",
            "temporal_context_tokens", "stage1_context_dim", "train_stage1_temporal_layers",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in ("feature_dim", "patch_tokens", "max_event_frames", "visual_tokens", "visual_heads", "caption_max_tokens", "generation_max_tokens", "generation_min_tokens", "no_repeat_ngram_size", "stage1_context_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.generation_min_tokens > self.generation_max_tokens:
            raise ValueError("generation_min_tokens cannot exceed generation_max_tokens.")
        for name in ("visual_dropout", "repetition_penalty"):
            value = getattr(self, name)
            if not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction must be non-empty.")

    @classmethod
    def from_json(cls, path: str | Path) -> "DirectMobileCLIPCaptionConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Stage 2 v3 config must be a JSON object.")
        unknown = sorted(set(payload) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown Stage 2 v3 config fields: {unknown}")
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
