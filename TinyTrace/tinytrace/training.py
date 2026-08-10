from __future__ import annotations

import json
import math
import platform
import random
import subprocess
import sys
from importlib import metadata as importlib_metadata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .model import TinyTraceModel


OPTIMIZER_FORMAT_VERSION = 3
CHECKPOINT_FORMAT_VERSION = 3
SUPPORTED_MONITORS = {
    "val_loss",
    "train_loss",
    "temporal_mean_iou",
    "r1_iou_0.3",
    "r1_iou_0.5",
    "score_mae",
    "caption_exact_match",
    "event_count_mae",
    "qvh_mean_score_proxy_Good_mAP",
    "qvh_mean_score_proxy_Good_Hit1",
    # Read compatibility for the v1 profile/artifacts. New Phase-A profiles
    # must use the explicitly named dense proxy or official metrics.
    "qvh_mAP",
    "qvh_HIT_at_1",
}
MONITOR_DIRECTIONS = {
    "val_loss": "min",
    "train_loss": "min",
    "temporal_mean_iou": "max",
    "r1_iou_0.3": "max",
    "r1_iou_0.5": "max",
    "score_mae": "min",
    "caption_exact_match": "max",
    "event_count_mae": "min",
    "qvh_mean_score_proxy_Good_mAP": "max",
    "qvh_mean_score_proxy_Good_Hit1": "max",
    "qvh_mAP": "max",
    "qvh_HIT_at_1": "max",
}
PARAMETER_GROUP_ORDER = (
    "compression",
    "embeddings",
    "lcem",
    "task_heads",
    "mobileclip",
)


@dataclass
class TrainingConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    visual_lr_scale: float = 0.1
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    amp_mode: str = "off"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 1
    accumulation_steps: int = 1
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    checkpoint_keep: int = 3

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("learning_rate", "weight_decay", "gradient_clip", "visual_lr_scale"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number.")
            if float(value) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")
        for name in ("warmup_ratio", "min_lr_ratio"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number in [0, 1].")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.amp_mode not in {"off", "auto", "fp16", "bf16"}:
            raise ValueError("amp_mode must be one of: off, auto, fp16, bf16.")
        if (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience < 0
        ):
            raise ValueError("early_stopping_patience must be a non-negative integer.")
        if (
            not isinstance(self.early_stopping_min_epochs, int)
            or isinstance(self.early_stopping_min_epochs, bool)
            or self.early_stopping_min_epochs < 1
        ):
            raise ValueError("early_stopping_min_epochs must be a positive integer.")
        if (
            not isinstance(self.early_stopping_min_delta, (int, float))
            or isinstance(self.early_stopping_min_delta, bool)
            or not math.isfinite(float(self.early_stopping_min_delta))
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError("early_stopping_min_delta must be finite and non-negative.")
        for name in ("accumulation_steps", "checkpoint_keep"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.monitor, str) or not self.monitor.strip():
            raise ValueError("monitor must be a non-empty string.")
        if self.monitor not in SUPPORTED_MONITORS:
            raise ValueError(
                f"monitor must be one of: {', '.join(sorted(SUPPORTED_MONITORS))}."
            )
        if self.monitor_mode not in {"min", "max"}:
            raise ValueError("monitor_mode must be either 'min' or 'max'.")
        expected_mode = MONITOR_DIRECTIONS[self.monitor]
        if self.monitor_mode != expected_mode:
            raise ValueError(
                f"monitor_mode for {self.monitor!r} must be {expected_mode!r}."
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainingProfile:
    """Validated, backwards-compatible configuration for a complete run."""

    train_script: str = "scripts/train_tinytrace.py"
    model_config: str = "configs/tinytrace_baseline.json"
    train_dataset_json: str = ""
    val_dataset_json: str = ""
    init_checkpoint: str = ""
    output_dir: str = "outputs"
    frame_cache_dir: str = ".cache/frames"
    visual_feature_cache_dir: str = ""
    require_visual_feature_cache: bool = False
    device: str = "cpu"
    epochs: int = 3
    batch_size: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    amp: str = "off"
    accumulation_steps: int = 1
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    early_stopping_min_epochs: int = 1
    monitor: str = "val_loss"
    monitor_mode: str = "min"
    dataset_size: int = 128
    resume: str = ""
    save_every: int = 5
    checkpoint_keep: int = 3
    prediction_every: int = 1
    prediction_samples: int = 2
    metrics_every: int = 1
    num_workers: int = 0
    log_every: int = 25
    max_steps_per_epoch: int = 0
    max_optimizer_steps: int = 0
    stage2_start_epoch: int = 0
    stage2_visual_lr_scale: float = 0.1
    stage2_unfreeze_strategy: str = "conv_exp"
    seed: int = 7
    deterministic: bool = True
    allow_random_frames: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("train_script", "model_config", "output_dir", "frame_cache_dir", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        for name in (
            "train_dataset_json",
            "val_dataset_json",
            "init_checkpoint",
            "visual_feature_cache_dir",
            "stage2_unfreeze_strategy",
            "resume",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string.")
        for name in ("epochs", "batch_size", "dataset_size", "prediction_samples", "checkpoint_keep"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "save_every",
            "prediction_every",
            "metrics_every",
            "num_workers",
            "log_every",
            "max_steps_per_epoch",
            "max_optimizer_steps",
            "stage2_start_epoch",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer.")
        if not all(
            isinstance(value, bool)
            for value in (
                self.deterministic,
                self.allow_random_frames,
                self.require_visual_feature_cache,
            )
        ):
            raise ValueError(
                "deterministic, allow_random_frames, and require_visual_feature_cache must be booleans."
            )
        if self.require_visual_feature_cache and not self.visual_feature_cache_dir:
            raise ValueError(
                "require_visual_feature_cache=true requires visual_feature_cache_dir."
            )
        if self.require_visual_feature_cache and self.stage2_start_epoch > 0:
            raise ValueError(
                "Frozen visual-feature caches cannot be used with stage-2 visual "
                "encoder unfreezing; set stage2_start_epoch=0."
            )
        TrainingConfig(
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            visual_lr_scale=self.stage2_visual_lr_scale,
            warmup_ratio=self.warmup_ratio,
            min_lr_ratio=self.min_lr_ratio,
            amp_mode=self.amp,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_min_delta=self.early_stopping_min_delta,
            early_stopping_min_epochs=self.early_stopping_min_epochs,
            accumulation_steps=self.accumulation_steps,
            monitor=self.monitor,
            monitor_mode=self.monitor_mode,
            checkpoint_keep=self.checkpoint_keep,
        )
        if self.early_stopping_patience > 0 and not self.val_dataset_json:
            raise ValueError("val_dataset_json is required when early stopping is enabled.")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingProfile":
        if not isinstance(values, dict):
            raise ValueError("Training profile payload must be an object.")
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown training profile fields: {', '.join(unknown)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parameter_category(name: str) -> str:
    if name.startswith("visual_encoder.mobileclip."):
        return "mobileclip"
    if name.startswith("visual_encoder.compressor."):
        return "compression"
    if name.startswith(
        (
            "text_embeddings.",
            "time_embeddings.",
            "score_embeddings.",
            "token_type_embeddings.",
        )
    ) or name == "sync_embedding":
        return "embeddings"
    if name.startswith("blocks.") or name.startswith("final_norm."):
        return "lcem"
    if name.startswith(
        (
            "text_head.",
            "time_head.",
            "score_head.",
            "boundary_head.",
            "phase_a_",
        )
    ):
        return "task_heads"
    raise ValueError(f"Trainable parameter is not assigned to a known optimizer group: {name}")


def build_named_optimizer(
    model: TinyTraceModel,
    config: TrainingConfig,
) -> torch.optim.AdamW:
    """Build deterministic named groups, including currently frozen MobileCLIP.

    Including the visual parameters from the beginning lets staged unfreezing
    preserve Adam and scheduler state without replacing the optimizer.
    """
    grouped: dict[tuple[str, str], list[torch.nn.Parameter]] = {}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        identifier = id(parameter)
        if identifier in seen:
            raise ValueError(f"Parameter appears more than once in model.named_parameters(): {name}")
        seen.add(identifier)
        category = _parameter_category(name)
        decay_kind = "decay" if parameter.ndim >= 2 else "no_decay"
        grouped.setdefault((category, decay_kind), []).append(parameter)

    parameter_groups = []
    for category in PARAMETER_GROUP_ORDER:
        for decay_kind in ("decay", "no_decay"):
            parameters = grouped.get((category, decay_kind), [])
            if not parameters:
                continue
            learning_rate = config.learning_rate
            if category == "mobileclip":
                learning_rate *= config.visual_lr_scale
            parameter_groups.append(
                {
                    "params": parameters,
                    "group_name": f"{category}.{decay_kind}",
                    "lr": learning_rate,
                    "initial_lr": learning_rate,
                    "weight_decay": config.weight_decay if decay_kind == "decay" else 0.0,
                }
            )

    assigned = {id(parameter) for group in parameter_groups for parameter in group["params"]}
    if assigned != seen:
        raise ValueError("Optimizer parameter assignment is incomplete or contains duplicates.")
    return torch.optim.AdamW(parameter_groups)


def optimizer_group_summary(optimizer: torch.optim.Optimizer) -> list[dict]:
    summary = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = group["params"]
        summary.append(
            {
                "name": group.get("group_name", f"group_{index}"),
                "parameter_count": sum(parameter.numel() for parameter in parameters),
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in parameters if parameter.requires_grad
                ),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
            }
        )
    return summary


def _build_legacy_optimizer(
    model: TinyTraceModel,
    learning_rate: float,
    weight_decay: float,
    visual_lr_scale: float,
) -> torch.optim.AdamW:
    visual_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("visual_encoder.mobileclip."):
            visual_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    groups = []
    if other_parameters:
        groups.append({"params": other_parameters, "lr": learning_rate})
    if visual_parameters:
        groups.append({"params": visual_parameters, "lr": learning_rate * visual_lr_scale})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def load_optimizer_state_compat(
    optimizer: torch.optim.Optimizer,
    model: TinyTraceModel,
    state_dict: dict,
    training_config: TrainingConfig,
    optimizer_format_version: int,
) -> None:
    if optimizer_format_version >= OPTIMIZER_FORMAT_VERSION:
        optimizer.load_state_dict(state_dict)
        return

    legacy = _build_legacy_optimizer(
        model,
        training_config.learning_rate,
        training_config.weight_decay,
        training_config.visual_lr_scale,
    )
    legacy.load_state_dict(state_dict)
    for parameter, state in legacy.state.items():
        optimizer.state[parameter] = state


def warmup_cosine_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps < 1:
        raise ValueError("total_steps must be positive.")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps).")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1].")
    clamped_step = min(max(int(step), 0), total_steps)
    if warmup_steps > 0 and clamped_step < warmup_steps:
        return float(clamped_step + 1) / float(warmup_steps)
    decay_steps = total_steps - warmup_steps
    decay_position = (clamped_step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_position))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    multiplier = lambda step: warmup_cosine_multiplier(  # noqa: E731
        step,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def set_scheduler_step(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    completed_steps: int,
) -> None:
    """Position a scheduler when migrating a checkpoint without scheduler state."""
    if completed_steps < 0:
        raise ValueError("completed_steps cannot be negative.")
    scheduler.last_epoch = completed_steps
    learning_rates = [
        base_lr * schedule(completed_steps)
        for base_lr, schedule in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    for group, learning_rate in zip(scheduler.optimizer.param_groups, learning_rates):
        group["lr"] = learning_rate
    scheduler._last_lr = learning_rates
    scheduler._step_count = completed_steps + 1


@dataclass(frozen=True)
class AmpSettings:
    enabled: bool
    dtype: torch.dtype | None
    use_grad_scaler: bool


def resolve_amp_settings(mode: str, device: torch.device) -> AmpSettings:
    if mode == "off":
        return AmpSettings(False, None, False)
    if mode == "auto":
        if device.type != "cuda":
            return AmpSettings(False, None, False)
        mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if mode == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 AMP is supported only on CUDA devices.")
        return AmpSettings(True, torch.float16, True)
    if mode == "bf16":
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("bf16 AMP requires a CPU or CUDA device.")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bf16 AMP.")
        return AmpSettings(True, torch.bfloat16, False)
    raise ValueError(f"Unsupported AMP mode: {mode}")


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    min_epochs: int = 1
    mode: str = "min"
    monitor: str = "val_loss"
    best: float = float("inf")
    bad_epochs: int = 0
    best_epoch: int = 0
    best_step: int = 0
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise ValueError("Early-stopping mode must be 'min' or 'max'.")
        if self.mode == "max" and self.best == float("inf"):
            self.best = float("-inf")

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def update(
        self,
        value: float,
        epoch: int,
        *,
        global_step: int = 0,
        active: bool = True,
    ) -> tuple[bool, bool]:
        if not math.isfinite(value):
            raise ValueError("Early-stopping metric must be finite.")
        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
            self.best_epoch = epoch
            self.best_step = global_step
        elif active and epoch >= self.min_epochs:
            self.bad_epochs += 1
        should_stop = (
            self.enabled
            and active
            and epoch >= self.min_epochs
            and self.bad_epochs >= self.patience
        )
        if should_stop:
            self.stop_reason = (
                f"{self.monitor} did not improve by {self.min_delta} for "
                f"{self.bad_epochs} eligible epochs"
            )
        return improved, should_stop

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: dict) -> None:
        self.best = float(state.get("best", self.best))
        self.bad_epochs = int(state.get("bad_epochs", self.bad_epochs))
        self.best_epoch = int(state.get("best_epoch", self.best_epoch))
        self.best_step = int(state.get("best_step", self.best_step))
        self.stop_reason = state.get("stop_reason", self.stop_reason)


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.failures: list[str] = []

    def log(self, record: dict) -> bool:
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            return True
        except OSError as exc:
            message = f"Unable to write TinyTrace training log {self.path}: {exc}"
            self.failures.append(message)
            print(message, file=sys.stderr)
            return False


def current_learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        str(group.get("group_name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def collect_run_metadata(device: torch.device, seed: int, deterministic: bool) -> dict[str, Any]:
    dependency_versions = {}
    for distribution in ("mobileclip", "torchvision", "numpy"):
        try:
            dependency_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        source_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = None
        source_dirty = None
    return {
        "seed": seed,
        "deterministic": deterministic,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
        ),
        "source_revision": revision,
        "source_dirty": source_dirty,
        "dependency_versions": dependency_versions,
    }


def prune_periodic_checkpoints(checkpoint_dir: str | Path, keep: int) -> list[Path]:
    if keep < 1:
        raise ValueError("keep must be positive.")
    directory = Path(checkpoint_dir)
    periodic = sorted(directory.glob("epoch-*.pt"))
    removed = periodic[:-keep]
    for path in removed:
        path.unlink(missing_ok=True)
    return removed


def validate_checkpoint_version(payload: dict[str, Any]) -> int:
    version = int(payload.get("checkpoint_format_version", 1))
    if version < 1:
        raise ValueError("checkpoint_format_version must be positive.")
    if version > CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint format {version} is newer than supported format "
            f"{CHECKPOINT_FORMAT_VERSION}."
        )
    return version
