"""Configuration for the isolated final causal event-modeling run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class FinalCausalConfig:
    schema_version: str = "tinytrace.final-causal.config.v1"
    feature_dim: int = 1024
    patch_tokens: int = 64
    max_video_frames: int = 32
    visual_slots_per_frame: int = 8
    time_tokens_per_frame: int = 6
    visual_heads: int = 8
    visual_dropout: float = 0.1
    time_bins: int = 100
    flan_model_name: str = "google/flan-t5-small"
    flan_revision: str = "main"
    flan_local_files_only: bool = True
    instruction: str = "Describe the chronological events shown in the video."
    target_max_tokens: int = 384
    generation_max_tokens: int = 384
    generation_min_tokens: int = 4
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    train_flan_encoder_layers: int = 1
    train_full_flan_decoder: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != "tinytrace.final-causal.config.v1":
            raise ValueError("Unsupported final causal config schema.")
        for name in ("feature_dim", "patch_tokens", "max_video_frames", "visual_slots_per_frame", "time_tokens_per_frame", "visual_heads", "time_bins", "target_max_tokens", "generation_max_tokens", "generation_min_tokens", "no_repeat_ngram_size", "train_flan_encoder_layers"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if any(getattr(self, name) < 1 for name in ("feature_dim", "patch_tokens", "max_video_frames", "visual_slots_per_frame", "time_tokens_per_frame", "visual_heads", "time_bins", "target_max_tokens", "generation_max_tokens")):
            raise ValueError("Feature, token, frame, and generation dimensions must be positive.")
        if self.generation_min_tokens > self.generation_max_tokens:
            raise ValueError("generation_min_tokens cannot exceed generation_max_tokens.")

    @classmethod
    def from_json(cls, path: str | Path) -> "FinalCausalConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Final causal config must be a JSON object.")
        unknown = sorted(set(payload) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"Unknown final causal config fields: {unknown}")
        return cls(**payload)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
