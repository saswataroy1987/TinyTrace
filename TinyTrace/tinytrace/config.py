from __future__ import annotations
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class TinyTraceConfig:
    MAX_SUPPORTED_FRAMES: ClassVar[int] = 128
    MAX_SUPPORTED_CAPTION_TOKENS: ClassVar[int] = 64
    MAX_SUPPORTED_GENERATED_TOKENS: ClassVar[int] = 512
    image_size: int = 256
    max_frames: int = 8
    visual_hidden_dim: int = 1024
    compressed_visual_tokens: int = 4
    time_tokens_per_frame: int = 6

    mobileclip_model_name: str = "mobileclip_s0"
    mobileclip_checkpoint: str = "checkpoints/mobileclip_s0.pt"
    mobileclip_checkpoint_sha256: str = "809b408eff74f8058843e86a1f92967097d42ba782450e85b8f4867b7f0ca0b7"
    freeze_visual_encoder: bool = True
    # Apple's pinned MobileCLIP v1 S0 transform does not normalize. This flag
    # exists only for explicit compatibility experiments and must remain false
    # in the architecture-aligned baseline.
    mobileclip_apply_normalization: bool = False
    mobileclip_image_mean: tuple[float, float, float] = (
        0.48145466,
        0.4578275,
        0.40821073,
    )
    mobileclip_image_std: tuple[float, float, float] = (
        0.26862954,
        0.26130258,
        0.27577711,
    )

    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.0
    max_position_embeddings: int = 2048
    time_loss_weight: float = 1.0
    score_loss_weight: float = 1.0
    caption_loss_weight: float = 1.0
    sync_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0

    # Phase A (QVHighlights) uses direct, fixed 2-second saliency bins.  This
    # deliberately bypasses the autoregressive timestamp/score protocol used
    # by caption-mode Phase B: a fixed grid has no numeric-string termination
    # or event-boundary decision to collapse.
    phase_a_dense_saliency: bool = False
    phase_a_bin_count: int = 75
    phase_a_bin_size_seconds: float = 2.0
    phase_a_max_score: float = 4.0
    phase_a_positive_threshold: float = 3.0
    saliency_regression_loss_weight: float = 1.0
    saliency_relevance_loss_weight: float = 0.5
    saliency_ranking_loss_weight: float = 0.25
    saliency_positive_weight: float = 4.77
    visual_encoder_chunk_size: int = 32

    text_vocab_size: int = 256
    max_text_len: int = 48
    max_caption_tokens: int = 20
    min_caption_tokens: int = 5
    max_events: int = 3
    max_generated_tokens: int = 128
    timestamp_value_count: int = 2
    score_value_count: int = 1

    time_vocab: tuple[str, ...] = field(
        default_factory=lambda: ("<sync>", "<sep>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".")
    )
    score_vocab: tuple[str, ...] = field(
        default_factory=lambda: ("<sync>", "<sep>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".")
    )

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    video_token_id: int = 3
    instruction_token_offset: int = 4

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Fail early when a configuration violates an executable contract."""
        positive_integer_fields = {
            "image_size": self.image_size,
            "max_frames": self.max_frames,
            "visual_hidden_dim": self.visual_hidden_dim,
            "compressed_visual_tokens": self.compressed_visual_tokens,
            "time_tokens_per_frame": self.time_tokens_per_frame,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "text_vocab_size": self.text_vocab_size,
            "max_events": self.max_events,
            "max_generated_tokens": self.max_generated_tokens,
            "max_position_embeddings": self.max_position_embeddings,
            "phase_a_bin_count": self.phase_a_bin_count,
            "visual_encoder_chunk_size": self.visual_encoder_chunk_size,
        }
        for name, value in positive_integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, received {value!r}.")

        nonnegative_integer_fields = {
            "max_text_len": self.max_text_len,
            "max_caption_tokens": self.max_caption_tokens,
            "min_caption_tokens": self.min_caption_tokens,
            "timestamp_value_count": self.timestamp_value_count,
            "score_value_count": self.score_value_count,
        }
        for name, value in nonnegative_integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, received {value!r}.")

        if self.time_tokens_per_frame != 6:
            raise ValueError(
                "time_tokens_per_frame must be 6 for the fixed-width '0000.0' TRACE encoding."
            )
        if not isinstance(self.phase_a_dense_saliency, bool):
            raise ValueError("phase_a_dense_saliency must be a boolean.")
        for name in (
            "phase_a_bin_size_seconds",
            "phase_a_max_score",
            "phase_a_positive_threshold",
            "saliency_positive_weight",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number.")
        if self.phase_a_positive_threshold > self.phase_a_max_score:
            raise ValueError("phase_a_positive_threshold cannot exceed phase_a_max_score.")
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal positional encoding.")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if not isinstance(self.dropout, (int, float)) or isinstance(self.dropout, bool):
            raise ValueError("dropout must be a number in [0, 1).")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.min_caption_tokens > self.max_caption_tokens:
            raise ValueError("min_caption_tokens cannot exceed max_caption_tokens.")
        if self.max_frames > self.MAX_SUPPORTED_FRAMES:
            raise ValueError(
                f"max_frames exceeds the declared safety limit of {self.MAX_SUPPORTED_FRAMES}."
            )
        if self.max_caption_tokens > self.MAX_SUPPORTED_CAPTION_TOKENS:
            raise ValueError(
                "max_caption_tokens exceeds the declared safety limit of "
                f"{self.MAX_SUPPORTED_CAPTION_TOKENS}."
            )
        if self.max_generated_tokens > self.MAX_SUPPORTED_GENERATED_TOKENS:
            raise ValueError(
                "max_generated_tokens exceeds the declared safety limit of "
                f"{self.MAX_SUPPORTED_GENERATED_TOKENS}."
            )
        for name in (
            "time_loss_weight",
            "score_loss_weight",
            "caption_loss_weight",
            "sync_loss_weight",
            "boundary_loss_weight",
            "saliency_regression_loss_weight",
            "saliency_relevance_loss_weight",
            "saliency_ranking_loss_weight",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number.")

        if not isinstance(self.mobileclip_model_name, str) or not self.mobileclip_model_name.strip():
            raise ValueError("mobileclip_model_name cannot be empty.")
        if not isinstance(self.mobileclip_checkpoint, str) or not self.mobileclip_checkpoint.strip():
            raise ValueError("mobileclip_checkpoint cannot be empty.")
        if not isinstance(self.mobileclip_checkpoint_sha256, str):
            raise ValueError("mobileclip_checkpoint_sha256 must be a string.")
        if self.mobileclip_checkpoint_sha256 and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.mobileclip_checkpoint_sha256
        ):
            raise ValueError("mobileclip_checkpoint_sha256 must be empty or a 64-character hexadecimal digest.")
        if not isinstance(self.freeze_visual_encoder, bool):
            raise ValueError("freeze_visual_encoder must be a boolean.")
        if not isinstance(self.mobileclip_apply_normalization, bool):
            raise ValueError("mobileclip_apply_normalization must be a boolean.")

        for name, values in (
            ("mobileclip_image_mean", self.mobileclip_image_mean),
            ("mobileclip_image_std", self.mobileclip_image_std),
        ):
            if not isinstance(values, (tuple, list)) or len(values) != 3 or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ):
                raise ValueError(f"{name} must contain three numeric channel values.")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain only finite values.")
        if any(float(value) <= 0 for value in self.mobileclip_image_std):
            raise ValueError("mobileclip_image_std values must be greater than zero.")

        required_numeric_vocab = (
            "<sync>",
            "<sep>",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ".",
        )
        if tuple(self.time_vocab) != required_numeric_vocab:
            raise ValueError("time_vocab must preserve the 13-token TRACE numeric vocabulary and ordering.")
        if tuple(self.score_vocab) != required_numeric_vocab:
            raise ValueError("score_vocab must preserve the 13-token TRACE numeric vocabulary and ordering.")

        special_ids = {
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "video_token_id": self.video_token_id,
        }
        if len(set(special_ids.values())) != len(special_ids):
            raise ValueError("Text special-token IDs must be distinct.")
        for name, value in special_ids.items():
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < self.text_vocab_size:
                raise ValueError(f"{name} must be an integer inside the text vocabulary.")
        if not isinstance(self.instruction_token_offset, int) or not 0 <= self.instruction_token_offset < self.text_vocab_size:
            raise ValueError("instruction_token_offset must be inside the text vocabulary.")

        maximum_sequence_length = self.maximum_training_sequence_length
        if maximum_sequence_length > self.max_position_embeddings:
            raise ValueError(
                "Configured maximum sequence length exceeds positional capacity: "
                f"{maximum_sequence_length} > {self.max_position_embeddings}."
            )
        if self.max_generated_tokens < self.required_generation_token_budget:
            raise ValueError(
                "max_generated_tokens cannot represent the configured maximum structured output: "
                f"{self.max_generated_tokens} < {self.required_generation_token_budget}."
            )
        if self.maximum_inference_sequence_length > self.max_position_embeddings:
            raise ValueError(
                "Configured inference sequence length exceeds positional capacity: "
                f"{self.maximum_inference_sequence_length} > {self.max_position_embeddings}."
            )

    @property
    def maximum_event_token_count(self) -> int:
        time_tokens = self.timestamp_value_count * 6 + max(0, self.timestamp_value_count - 1) + 1
        score_tokens = self.score_value_count * 3 + max(0, self.score_value_count - 1) + 1
        caption_tokens = self.max_caption_tokens + 1
        return time_tokens + score_tokens + caption_tokens

    @property
    def maximum_training_sequence_length(self) -> int:
        visual_prefix = self.max_frames * (
            self.compressed_visual_tokens + self.time_tokens_per_frame
        )
        prompt = self.max_text_len + 2
        if self.phase_a_dense_saliency:
            return visual_prefix + prompt
        event_targets = self.max_events * self.maximum_event_token_count + 1
        return visual_prefix + prompt + event_targets

    @property
    def required_generation_token_budget(self) -> int:
        """Tokens required for max_events complete events followed by EOS."""
        if self.phase_a_dense_saliency:
            return 1
        return self.max_events * self.maximum_event_token_count + 1

    @property
    def maximum_inference_sequence_length(self) -> int:
        visual_prefix = self.max_frames * (
            self.compressed_visual_tokens + self.time_tokens_per_frame
        )
        maximum_prompt = self.max_text_len + 2
        if self.phase_a_dense_saliency:
            return visual_prefix + maximum_prompt
        return visual_prefix + maximum_prompt + self.max_generated_tokens

    @property
    def text_token_base(self) -> int:
        return 0

    @property
    def sync_token_id(self) -> int:
        return self.text_vocab_size

    @property
    def time_token_base(self) -> int:
        return self.text_vocab_size + 1

    @property
    def score_token_base(self) -> int:
        return self.time_token_base + len(self.time_vocab)

    @property
    def total_token_vocab(self) -> int:
        return self.score_token_base + len(self.score_vocab)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TinyTraceConfig":
        if not isinstance(values, dict):
            raise ValueError("TinyTrace configuration payload must be an object.")
        # dataclasses.fields excludes ClassVar safety constants, so JSON cannot
        # override executable limits as though they were model parameters.
        known_fields = {config_field.name for config_field in fields(cls)}
        unknown = sorted(set(values) - set(known_fields))
        if unknown:
            raise ValueError(f"Unknown TinyTrace config fields: {', '.join(unknown)}")

        normalized = dict(values)
        # Compatibility migration: the immediately preceding checkpoint format
        # stored mean/std fields while preprocessing normalized unconditionally.
        # Earlier checkpoints without these fields used the official no-normalize
        # path. New checkpoints always store the explicit flag.
        if "mobileclip_apply_normalization" not in normalized and (
            "mobileclip_image_mean" in normalized or "mobileclip_image_std" in normalized
        ):
            normalized["mobileclip_apply_normalization"] = True
        for field_name in ("time_vocab", "score_vocab", "mobileclip_image_mean", "mobileclip_image_std"):
            if field_name in normalized:
                normalized[field_name] = tuple(normalized[field_name])
        return cls(**normalized)

    @classmethod
    def from_json(cls, path: str | Path) -> "TinyTraceConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
