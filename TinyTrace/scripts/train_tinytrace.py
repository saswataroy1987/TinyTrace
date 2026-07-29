from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace import (
    JsonTinyTraceDataset,
    SyntheticTinyTraceDataset,
    TinyTraceConfig,
    TinyTraceModel,
    aggregate_caption_budget,
    decode_event_sequence,
    evaluate_event_predictions,
    tinytrace_collate_fn,
)
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer
from tinytrace.training import (
    CHECKPOINT_FORMAT_VERSION,
    OPTIMIZER_FORMAT_VERSION,
    AmpSettings,
    EarlyStopping,
    JsonlLogger,
    SUPPORTED_MONITORS,
    TrainingConfig,
    build_named_optimizer,
    build_warmup_cosine_scheduler,
    capture_rng_state,
    collect_run_metadata,
    current_learning_rates,
    load_optimizer_state_compat,
    optimizer_group_summary,
    prune_periodic_checkpoints,
    resolve_amp_settings,
    restore_rng_state,
    set_scheduler_step,
    validate_checkpoint_version,
)


DETERMINISTIC_CUBLAS_CONFIG = ":4096:8"
VALID_CUBLAS_WORKSPACE_CONFIGS = {":4096:8", ":16:8"}


def configure_cuda_determinism(*, deterministic: bool, device: torch.device) -> str | None:
    """Configure cuBLAS before the first deterministic CUDA matrix multiply."""

    if not deterministic or device.type != "cuda":
        return None
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured is None:
        configured = DETERMINISTIC_CUBLAS_CONFIG
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = configured
    if configured not in VALID_CUBLAS_WORKSPACE_CONFIGS:
        allowed = ", ".join(sorted(VALID_CUBLAS_WORKSPACE_CONFIGS))
        raise ValueError(
            "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG to be one of "
            f"{allowed}; received {configured!r}."
        )
    return configured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyTrace with reproducible artifacts.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--amp", choices=("off", "auto", "fp16", "bf16"), default="off")
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=1)
    parser.add_argument("--monitor", type=str, default="val_loss")
    parser.add_argument("--monitor-mode", choices=("min", "max"), default="min")
    parser.add_argument("--dataset-size", type=int, default=128)
    parser.add_argument("--dataset-json", type=str, default="")
    parser.add_argument("--val-dataset-json", type=str, default="")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/tinytrace_baseline.json"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--frame-cache-dir", type=str, default=str(PROJECT_ROOT / ".cache/frames"))
    parser.add_argument("--visual-feature-cache-dir", type=str, default="")
    parser.add_argument(
        "--require-visual-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--allow-random-frames", action="store_true")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--checkpoint-keep", type=int, default=3)
    parser.add_argument("--prediction-every", type=int, default=1)
    parser.add_argument("--prediction-samples", type=int, default=2)
    parser.add_argument("--metrics-every", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help="Exact total optimizer-step cap; 0 runs the full epoch plan.",
    )
    parser.add_argument("--stage2-start-epoch", type=int, default=0)
    parser.add_argument("--stage2-visual-lr-scale", type=float, default=0.1)
    parser.add_argument("--stage2-unfreeze-strategy", type=str, default="conv_exp")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def build_dataset(
    annotation_path: str,
    config: TinyTraceConfig,
    frame_cache_dir: str,
    allow_random_frames: bool,
    synthetic_size: int | None = None,
    seed: int = 7,
    visual_feature_cache_dir: str = "",
    require_visual_feature_cache: bool = False,
) -> Dataset:
    if annotation_path:
        return JsonTinyTraceDataset(
            annotation_path,
            config=config,
            frame_cache_dir=frame_cache_dir,
            allow_random_frames=allow_random_frames,
            validate_videos_on_init=True,
            strict_media_validation=True,
            visual_feature_cache_dir=visual_feature_cache_dir or None,
            require_visual_feature_cache=require_visual_feature_cache,
        )
    if synthetic_size is None:
        raise ValueError("A validation dataset path is required for real-data validation.")
    return SyntheticTinyTraceDataset(config=config, size=synthetic_size, seed=seed)


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=tinytrace_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def move_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    moved = {
        key: batch[key].to(device, non_blocking=device.type == "cuda")
        for key in ("frames", "frame_times", "frame_mask", "token_ids", "label_types")
    }
    for key in (
        "prompt_lengths",
        "saliency_targets",
        "saliency_target_mask",
        "visual_patch_features",
    ):
        if key in batch:
            moved[key] = batch[key].to(device, non_blocking=device.type == "cuda")
    return moved


def run_epoch(
    model: TinyTraceModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    epoch: int,
    log_every: int,
    split: str,
    max_steps_per_epoch: int = 0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_settings: AmpSettings | None = None,
    event_logger: JsonlLogger | None = None,
    global_step: int = 0,
    global_micro_step: int = 0,
    accumulation_steps: int = 1,
    max_optimizer_steps: int = 0,
) -> tuple[dict, int]:
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive.")
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    component_totals: dict[str, float] = {}
    weighted_component_totals: dict[str, float] = {}
    target_counts: dict[str, int] = {}
    gradient_norm_total = 0.0
    gradient_norm_steps = 0
    clipped_steps = 0
    optimizer_steps = 0
    examples = 0
    frames_processed = 0
    tokens_processed = 0
    caption_budget_reports: list[dict[str, object]] = []
    steps = 0
    total_steps = min(len(loader), max_steps_per_epoch) if max_steps_per_epoch > 0 else len(loader)
    epoch_started = time.perf_counter()
    amp_settings = amp_settings or AmpSettings(False, None, False)
    context = torch.enable_grad() if training else torch.no_grad()
    current_window_size = 1
    with context:
        for step_index, raw_batch in enumerate(loader, start=1):
            if training and max_optimizer_steps > 0 and optimizer_steps >= max_optimizer_steps:
                break
            if max_steps_per_epoch > 0 and step_index > max_steps_per_epoch:
                break
            batch = move_batch(raw_batch, device)
            caption_budget_reports.extend(raw_batch.get("caption_budget", []))
            examples += int(batch["frames"].size(0))
            frames_processed += int(batch["frame_mask"].sum().item())
            tokens_processed += int(batch["token_ids"].ne(model.config.pad_token_id).sum().item())
            window_position = (step_index - 1) % accumulation_steps
            if training and window_position == 0:
                optimizer.zero_grad(set_to_none=True)
                current_window_size = min(accumulation_steps, total_steps - step_index + 1)
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_settings.dtype)
                if amp_settings.enabled
                else nullcontext()
            )
            with autocast_context:
                output = model(
                    batch["frames"],
                    batch["frame_times"],
                    batch["token_ids"],
                    labels=batch["token_ids"],
                    label_types=batch["label_types"],
                    frame_mask=batch["frame_mask"],
                    visual_patch_features=batch.get("visual_patch_features"),
                    prompt_lengths=batch.get("prompt_lengths"),
                    saliency_targets=batch.get("saliency_targets"),
                    saliency_target_mask=batch.get("saliency_target_mask"),
                )
            if output.loss is None:
                continue
            optimizer_step_performed = False
            gradient_norm_value: float | None = None
            if training:
                normalized_loss = output.loss / current_window_size
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(normalized_loss).backward()
                else:
                    normalized_loss.backward()

                at_boundary = window_position + 1 == current_window_size
                if at_boundary:
                    if scaler is not None and scaler.is_enabled():
                        scale_before = scaler.get_scale()
                        scaler.unscale_(optimizer)
                    else:
                        scale_before = None
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        (parameter for parameter in model.parameters() if parameter.requires_grad),
                        gradient_clip if gradient_clip > 0 else float("inf"),
                        error_if_nonfinite=False,
                    )
                    gradient_norm_value = float(gradient_norm.detach().float().cpu())
                    finite_gradient = math.isfinite(gradient_norm_value)
                    if finite_gradient:
                        gradient_norm_total += gradient_norm_value
                        gradient_norm_steps += 1
                        if gradient_clip > 0 and gradient_norm_value > gradient_clip:
                            clipped_steps += 1
                    elif event_logger is not None:
                        event_logger.log(
                            {
                                "record_type": "nonfinite_gradient",
                                "split": split,
                                "epoch": epoch,
                                "micro_step": step_index,
                                "global_micro_step": global_micro_step + step_index,
                                "global_step": global_step,
                                "gradient_norm": gradient_norm_value,
                                "amp_scaled": bool(scaler is not None and scaler.is_enabled()),
                            }
                        )

                    if scaler is not None and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer_step_performed = finite_gradient and scaler.get_scale() >= scale_before
                    elif finite_gradient:
                        optimizer.step()
                        optimizer_step_performed = True
                    else:
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError(
                            f"Non-finite gradient norm at epoch={epoch}, micro_step={step_index}."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if optimizer_step_performed:
                        if scheduler is not None:
                            scheduler.step()
                        global_step += 1
                        optimizer_steps += 1

            loss_value = float(output.loss.detach().float().cpu())
            running_loss += loss_value
            for name, component in output.loss_components.items():
                count = int(output.target_counts[name].cpu())
                component_totals[name] = component_totals.get(name, 0.0) + float(
                    component.detach().float().cpu()
                ) * count
                weighted_component_totals[name] = weighted_component_totals.get(name, 0.0) + float(
                    output.weighted_loss_components[name].detach().float().cpu()
                ) * count
                target_counts[name] = target_counts.get(name, 0) + count
            steps += 1
            step_record = {
                "record_type": "step",
                "split": split,
                "epoch": epoch,
                "micro_step": step_index,
                "global_micro_step": global_micro_step + step_index,
                "global_step": global_step,
                "loss": loss_value,
                "loss_components": {
                    name: float(value.detach().float().cpu())
                    for name, value in output.loss_components.items()
                },
                "weighted_loss_components": {
                    name: float(value.detach().float().cpu())
                    for name, value in output.weighted_loss_components.items()
                },
                "target_counts": {
                    name: int(value.cpu()) for name, value in output.target_counts.items()
                },
                "learning_rates": current_learning_rates(optimizer) if optimizer is not None else {},
            }
            if training and gradient_norm_value is not None:
                step_record["gradient_norm"] = gradient_norm_value
                step_record["gradient_clipped"] = bool(
                    math.isfinite(gradient_norm_value)
                    and gradient_clip > 0
                    and gradient_norm_value > gradient_clip
                )
                step_record["optimizer_step_performed"] = optimizer_step_performed
            if event_logger is not None:
                event_logger.log(step_record)
            if log_every > 0 and (step_index % log_every == 0 or step_index == total_steps):
                average_loss = running_loss / steps
                print(
                    f"{split}_step epoch={epoch} step={step_index}/{total_steps} "
                    f"loss={float(output.loss.detach().cpu()):.6f} avg_loss={average_loss:.6f}"
                )
    if steps == 0:
        raise RuntimeError("No loss-producing batches were available.")
    elapsed = time.perf_counter() - epoch_started
    metrics = {
        "loss": running_loss / steps,
        "loss_components": {
            name: component_totals[name] / target_counts[name] for name in component_totals
        },
        "weighted_loss_components": {
            name: weighted_component_totals[name] / target_counts[name]
            for name in weighted_component_totals
        },
        "target_counts": target_counts,
        "steps": steps,
        "optimizer_steps": optimizer_steps,
        "examples": examples,
        "frames": frames_processed,
        "tokens": tokens_processed,
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed if elapsed > 0 else 0.0,
        "frames_per_second": frames_processed / elapsed if elapsed > 0 else 0.0,
        "tokens_per_second": tokens_processed / elapsed if elapsed > 0 else 0.0,
        "caption_budget": aggregate_caption_budget(caption_budget_reports),
        "mean_gradient_norm": (
            gradient_norm_total / gradient_norm_steps if gradient_norm_steps else None
        ),
        "clipped_steps": clipped_steps,
        "learning_rates": current_learning_rates(optimizer) if optimizer is not None else {},
        "peak_accelerator_memory_allocated": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_accelerator_memory_reserved": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
    }
    if event_logger is not None:
        event_logger.log(
            {
                "record_type": "epoch",
                "split": split,
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
            }
        )
    return metrics, global_step


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(
    path: Path,
    model: TinyTraceModel,
    optimizer: torch.optim.Optimizer,
    config: TinyTraceConfig,
    epoch: int,
    best_loss: float,
    history: list[dict],
    training_state: dict | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    training_config: TrainingConfig | None = None,
    early_stopping: EarlyStopping | None = None,
    global_step: int = 0,
    global_micro_step: int = 0,
    global_examples: int = 0,
    selection_state: dict | None = None,
    run_metadata: dict | None = None,
    accumulation_state: dict | None = None,
) -> None:
    atomic_torch_save(
        path,
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "artifact_type": "resumable_training_checkpoint",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config.to_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
            "history": history,
            "training_state": dict(training_state or {}),
            "optimizer_format_version": OPTIMIZER_FORMAT_VERSION,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
            "training_config": training_config.to_dict() if training_config is not None else None,
            "early_stopping_state": early_stopping.state_dict() if early_stopping is not None else None,
            "global_step": global_step,
            "global_micro_step": global_micro_step,
            "global_examples": global_examples,
            "accumulation_state": dict(accumulation_state or {"pending_micro_steps": 0}),
            "selection_state": dict(selection_state or {}),
            "rng_state": capture_rng_state(),
            "run_metadata": dict(run_metadata or {}),
        },
    )


@torch.no_grad()
def collect_predictions(
    model: TinyTraceModel,
    dataset: Dataset,
    config: TinyTraceConfig,
    device: torch.device,
    sample_limit: int | None = None,
    checkpoint_identity: str | None = None,
) -> list[dict]:
    was_training = model.training
    model.eval()
    text_tokenizer = CharTokenizer(config.text_vocab_size)
    time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
    score_tokenizer = NumericTokenizer(config.score_vocab, width=3)
    predictions = []
    total = len(dataset) if sample_limit is None else min(sample_limit, len(dataset))
    try:
        for index in range(total):
            sample = dataset[index]
            frames = sample["frames"].unsqueeze(0).to(device)
            frame_times = sample["frame_times"].unsqueeze(0).to(device)
            frame_mask = sample.get("frame_mask")
            frame_mask = frame_mask.unsqueeze(0).to(device) if frame_mask is not None else None
            prompt_ids = sample["token_ids"][: sample["prompt_length"]].unsqueeze(0).to(device)
            patch_features = sample.get("visual_patch_features")
            patch_features = (
                patch_features.unsqueeze(0).to(device)
                if patch_features is not None
                else model.visual_encoder.extract_patch_features(frames)
            )
            if config.phase_a_dense_saliency and sample.get("task_mode") == "highlight":
                output = model(
                    frames,
                    frame_times,
                    prompt_ids,
                    frame_mask=frame_mask,
                    visual_patch_features=patch_features,
                    prompt_lengths=torch.tensor(
                        [sample["prompt_length"]], dtype=torch.long, device=device
                    ),
                )
                if output.saliency_scores is None:
                    raise RuntimeError("Dense Phase-A model did not return saliency scores.")
                dense_scores = output.saliency_scores[0].detach().float().cpu().tolist()
                target_scores = sample["saliency_targets"].float().tolist()

                def dense_events(values: list[float], threshold: float) -> list[dict]:
                    events: list[dict] = []
                    start: int | None = None
                    for bin_index in range(len(values) + 1):
                        active = bin_index < len(values) and values[bin_index] >= threshold
                        if active and start is None:
                            start = bin_index
                        if not active and start is not None:
                            end = bin_index
                            events.append(
                                {
                                    "timestamp": [
                                        start * config.phase_a_bin_size_seconds,
                                        end * config.phase_a_bin_size_seconds,
                                    ],
                                    "score": [max(values[start:end])],
                                }
                            )
                            start = None
                    return events

                predicted = dense_events(dense_scores, config.phase_a_positive_threshold)
                ground_truth = dense_events(target_scores, config.phase_a_positive_threshold)
                artifact = {
                    "sample_index": index,
                    "source_id": sample.get("source_id"),
                    "qid": sample.get("source_id"),
                    "query": sample.get("query"),
                    "video_path": sample.get("video_path"),
                    "task_mode": "highlight",
                    "checkpoint_identity": checkpoint_identity,
                    "ground_truth": ground_truth,
                    "predicted": predicted,
                    "pred_saliency_scores": dense_scores,
                    "qvh_mean_score_targets": target_scores,
                    "raw_generated_token_ids": [],
                    "generation": {
                        "protocol": "dense_75x2s",
                        "termination_reason": "not_applicable",
                        "forced_caption_termination": False,
                        "generated_token_count": 0,
                        "completed_event_count": len(predicted),
                        "final_mode": "dense_saliency",
                    },
                    "parser_warnings": [],
                }
                if sample.get("qvh_saliency_scores") is not None:
                    artifact["qvh_saliency_scores"] = sample["qvh_saliency_scores"]
                predictions.append(artifact)
                continue
            generated, generation_metadata = model.generate(
                frames,
                frame_times,
                prompt_ids,
                max_new_tokens=config.max_generated_tokens,
                frame_mask=frame_mask,
                visual_patch_features=patch_features,
                return_metadata=True,
                task_mode=sample.get("task_mode", "caption"),
            )
            raw_ids = generated[0, prompt_ids.size(1) :].tolist()
            parser_warnings: list[str] = []
            predicted = decode_event_sequence(
                raw_ids,
                config,
                text_tokenizer,
                time_tokenizer,
                score_tokenizer,
                warnings=parser_warnings,
                task_mode=sample.get("task_mode", "caption"),
            )
            predictions.append(
                {
                    "sample_index": index,
                    "source_id": sample.get("source_id"),
                    "video_path": sample.get("video_path"),
                    "query": sample.get("query"),
                    "task_mode": sample.get("task_mode", "caption"),
                    "checkpoint_identity": checkpoint_identity,
                    "ground_truth": sample["events"],
                    "ground_truth_caption_budget": sample.get("caption_budget"),
                    "raw_generated_token_ids": raw_ids,
                    "predicted": predicted,
                    "generation": generation_metadata,
                    "parser_warnings": parser_warnings,
                }
            )
    finally:
        model.train(was_training)
    return predictions


@torch.no_grad()
def dump_predictions(
    model: TinyTraceModel,
    dataset: Dataset,
    config: TinyTraceConfig,
    device: torch.device,
    path: Path,
    sample_count: int,
    checkpoint_identity: str | None = None,
) -> None:
    predictions = collect_predictions(
        model,
        dataset,
        config,
        device,
        sample_limit=sample_count,
        checkpoint_identity=checkpoint_identity,
    )
    atomic_json(path, predictions)


def summarize_prediction_artifacts(predictions: list[dict]) -> dict[str, float | int]:
    total = len(predictions)
    if total == 0:
        return {
            "samples": 0,
            "eos_termination_rate": 0.0,
            "maximum_budget_termination_rate": 0.0,
            "forced_caption_termination_rate": 0.0,
            "parser_warning_rate": 0.0,
        }
    dense = [
        item for item in predictions if item.get("generation", {}).get("protocol") == "dense_75x2s"
    ]
    summary = {
        "samples": total,
        "eos_termination_rate": sum(
            item["generation"]["termination_reason"] == "eos" for item in predictions
        )
        / total,
        "maximum_budget_termination_rate": sum(
            item["generation"]["termination_reason"] == "max_tokens" for item in predictions
        )
        / total,
        "forced_caption_termination_rate": sum(
            bool(item["generation"]["forced_caption_termination"]) for item in predictions
        )
        / total,
        "parser_warning_rate": sum(bool(item["parser_warnings"]) for item in predictions) / total,
    }
    if dense:
        curves = [item["pred_saliency_scores"] for item in dense]
        summary.update(
            {
                "dense_phase_a_samples": len(dense),
                "dense_prediction_score_variance": sum(
                    float(torch.tensor(curve).var(unbiased=False)) for curve in curves
                )
                / len(curves),
                "dense_unique_argmax_ratio": len(
                    {max(range(len(curve)), key=curve.__getitem__) for curve in curves}
                )
                / len(curves),
                "dense_distinct_curve_ratio": len(
                    {tuple(round(value, 6) for value in curve) for curve in curves}
                )
                / len(curves),
            }
        )
    return summary


def _run_training() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if not args.dataset_json and not args.allow_random_frames:
        # Synthetic mode is always allowed when no dataset json is provided.
        args.allow_random_frames = True
    requested_output = Path(args.output_dir)
    if not args.resume and requested_output.is_dir() and any(requested_output.iterdir()):
        raise FileExistsError(
            "A fresh TinyTrace run requires a new or empty output directory; "
            f"refusing to mix artifacts in {requested_output}."
        )
    device = torch.device(args.device)
    configure_cuda_determinism(deterministic=args.deterministic, device=device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        validate_checkpoint_version(resume_payload)
        saved_metadata = resume_payload.get("run_metadata", {})
        args.seed = int(saved_metadata.get("seed", args.seed))
        saved_arguments = saved_metadata.get("run_arguments", {})
        for field in (
            "dataset_json",
            "val_dataset_json",
            "dataset_size",
            "batch_size",
            "max_steps_per_epoch",
            "num_workers",
            "allow_random_frames",
        ):
            if field in saved_arguments:
                setattr(args, field, saved_arguments[field])
        saved_deterministic = resume_payload.get("training_state", {}).get("deterministic")
        if saved_deterministic is not None:
            args.deterministic = bool(saved_deterministic)
        config = TinyTraceConfig.from_dict(resume_payload["config"])
    else:
        config = TinyTraceConfig.from_json(args.config)

    if args.require_visual_feature_cache:
        if not config.freeze_visual_encoder:
            raise ValueError(
                "Frozen visual-feature caches require freeze_visual_encoder=true."
            )
        if args.stage2_start_epoch > 0:
            raise ValueError(
                "Frozen visual-feature caches cannot be used with stage-2 visual "
                "encoder unfreezing; set --stage2-start-epoch 0."
            )
        if resume_payload and resume_payload.get("training_state", {}).get(
            "stage2_activated", False
        ):
            raise ValueError(
                "Cannot resume a checkpoint whose visual encoder was unfrozen while "
                "requiring frozen visual-feature caches."
            )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(args.deterministic)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = args.deterministic
        torch.backends.cudnn.benchmark = not args.deterministic

    if args.dataset_json and not Path(args.dataset_json).is_file():
        raise FileNotFoundError(f"Training dataset JSON not found: {args.dataset_json}")
    if args.val_dataset_json and not Path(args.val_dataset_json).is_file():
        raise FileNotFoundError(f"Validation dataset JSON not found: {args.val_dataset_json}")

    train_dataset = build_dataset(
        args.dataset_json,
        config,
        args.frame_cache_dir,
        args.allow_random_frames,
        synthetic_size=args.dataset_size,
        seed=args.seed,
        visual_feature_cache_dir=args.visual_feature_cache_dir,
        require_visual_feature_cache=args.require_visual_feature_cache,
    )
    val_dataset = (
        build_dataset(
            args.val_dataset_json,
            config,
            args.frame_cache_dir,
            False,
            visual_feature_cache_dir=args.visual_feature_cache_dir,
            require_visual_feature_cache=args.require_visual_feature_cache,
        )
        if args.val_dataset_json
        else None
    )
    pin_memory = device.type == "cuda"
    train_loader = build_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    val_loader = (
        build_loader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed,
            pin_memory=pin_memory,
        )
        if val_dataset is not None
        else None
    )

    cli_training_config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        visual_lr_scale=args.stage2_visual_lr_scale,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        amp_mode=args.amp,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        accumulation_steps=args.accumulation_steps,
        monitor=args.monitor,
        monitor_mode=args.monitor_mode,
        checkpoint_keep=args.checkpoint_keep,
    )
    saved_training_config = resume_payload.get("training_config") if resume_payload else None
    training_config = (
        TrainingConfig(**saved_training_config) if saved_training_config else cli_training_config
    )
    if training_config.early_stopping_patience > 0 and val_loader is None:
        raise ValueError("Validation data is required when early stopping is enabled.")
    metric_monitors = SUPPORTED_MONITORS - {"val_loss", "train_loss"}
    if training_config.monitor in metric_monitors and val_dataset is None:
        raise ValueError("A validation dataset is required for structured metric monitoring.")

    micro_steps_per_epoch = len(train_loader)
    if args.max_steps_per_epoch > 0:
        micro_steps_per_epoch = min(micro_steps_per_epoch, args.max_steps_per_epoch)
    optimizer_steps_per_epoch = math.ceil(
        micro_steps_per_epoch / training_config.accumulation_steps
    )
    total_optimizer_steps = args.epochs * optimizer_steps_per_epoch
    if args.max_optimizer_steps > 0:
        total_optimizer_steps = min(total_optimizer_steps, args.max_optimizer_steps)
    if total_optimizer_steps < 1:
        raise ValueError("Training must contain at least one optimizer step.")
    warmup_steps = min(
        int(round(total_optimizer_steps * training_config.warmup_ratio)),
        max(total_optimizer_steps - 1, 0),
    )

    model = TinyTraceModel(config, load_pretrained_visual=resume_payload is None).to(device)
    start_epoch = 1
    best_loss = float("inf")
    history: list[dict] = []
    global_step = 0
    global_micro_step = 0
    global_examples = 0
    best_loss_epoch = 0
    best_loss_step = 0
    stage2_activated = False
    active_stage2_strategy = args.stage2_unfreeze_strategy
    active_stage2_start_epoch = args.stage2_start_epoch
    optimizer_format_version = OPTIMIZER_FORMAT_VERSION
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])
        saved_training_state = resume_payload.get("training_state", {})
        optimizer_format_version = int(resume_payload.get("optimizer_format_version", 1))
        legacy_stage2 = (
            optimizer_format_version < OPTIMIZER_FORMAT_VERSION
            and len(resume_payload["optimizer_state"].get("param_groups", [])) > 1
        )
        stage2_activated = bool(saved_training_state.get("stage2_activated", False)) or legacy_stage2
        active_stage2_strategy = str(
            saved_training_state.get("stage2_unfreeze_strategy", args.stage2_unfreeze_strategy)
        )
        active_stage2_start_epoch = int(
            saved_training_state.get("stage2_start_epoch", args.stage2_start_epoch)
        )
        if stage2_activated:
            model.set_visual_encoder_trainable(True, strategy=active_stage2_strategy)

    optimizer = build_named_optimizer(model, training_config)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=training_config.min_lr_ratio,
    )
    amp_settings = resolve_amp_settings(training_config.amp_mode, device)
    scaler = (
        torch.amp.GradScaler(device.type, enabled=True)
        if amp_settings.use_grad_scaler
        else None
    )
    early_stopping = EarlyStopping(
        patience=training_config.early_stopping_patience,
        min_delta=training_config.early_stopping_min_delta,
        min_epochs=training_config.early_stopping_min_epochs,
        mode=training_config.monitor_mode,
        monitor=training_config.monitor,
    )
    if resume_payload is not None:
        load_optimizer_state_compat(
            optimizer,
            model,
            resume_payload["optimizer_state"],
            training_config,
            optimizer_format_version=optimizer_format_version,
        )
        start_epoch = int(resume_payload["epoch"]) + 1
        best_loss = float(resume_payload.get("best_loss", float("inf")))
        history = list(resume_payload.get("history", []))
        global_step = int(
            resume_payload.get("global_step", (start_epoch - 1) * optimizer_steps_per_epoch)
        )
        global_micro_step = int(
            resume_payload.get("global_micro_step", (start_epoch - 1) * micro_steps_per_epoch)
        )
        global_examples = int(resume_payload.get("global_examples", 0))
        accumulation_state = resume_payload.get("accumulation_state", {})
        if int(accumulation_state.get("pending_micro_steps", 0)) != 0:
            raise ValueError("TinyTrace can resume only from a completed accumulation boundary.")
        saved_selection = resume_payload.get("selection_state", {})
        best_loss_epoch = int(saved_selection.get("best_loss_epoch", 0))
        best_loss_step = int(saved_selection.get("best_loss_step", 0))
        saved_total_steps = resume_payload.get("training_state", {}).get("total_optimizer_steps")
        if saved_total_steps is not None and int(saved_total_steps) != total_optimizer_steps:
            raise ValueError(
                "Cannot change the planned optimizer-step count when resuming a scheduled run: "
                f"checkpoint={saved_total_steps}, requested={total_optimizer_steps}."
            )
        if resume_payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        else:
            set_scheduler_step(scheduler, global_step)
        if scaler is not None and resume_payload.get("scaler_state") is not None:
            scaler.load_state_dict(resume_payload["scaler_state"])
        if resume_payload.get("early_stopping_state") is not None:
            early_stopping.load_state_dict(resume_payload["early_stopping_state"])
        else:
            early_stopping.best = best_loss
        restore_rng_state(resume_payload.get("rng_state"))
        print(f"resumed={args.resume} start_epoch={start_epoch}")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    prediction_dir = output_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    Path(args.frame_cache_dir).mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "config.json", config.to_dict())
    atomic_json(output_dir / "training_config.json", training_config.to_dict())
    run_arguments = vars(args).copy()
    run_arguments.update(
        {
            "lr": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "gradient_clip": training_config.gradient_clip,
            "warmup_ratio": training_config.warmup_ratio,
            "min_lr_ratio": training_config.min_lr_ratio,
            "amp": training_config.amp_mode,
            "accumulation_steps": training_config.accumulation_steps,
            "early_stopping_patience": training_config.early_stopping_patience,
            "early_stopping_min_delta": training_config.early_stopping_min_delta,
            "early_stopping_min_epochs": training_config.early_stopping_min_epochs,
            "monitor": training_config.monitor,
            "monitor_mode": training_config.monitor_mode,
            "checkpoint_keep": training_config.checkpoint_keep,
        }
    )
    atomic_json(output_dir / "run_arguments.json", run_arguments)
    atomic_json(output_dir / "optimizer_groups.json", optimizer_group_summary(optimizer))
    event_logger = JsonlLogger(output_dir / "training_log.jsonl")
    run_metadata = collect_run_metadata(device, args.seed, args.deterministic)
    run_metadata.update(
        {
            "run_id": (
                resume_payload.get("run_metadata", {}).get("run_id")
                if resume_payload is not None
                else uuid.uuid4().hex
            ) or uuid.uuid4().hex,
            "dataset_json": args.dataset_json or None,
            "validation_dataset_json": args.val_dataset_json or None,
            "model_config_sha256": hashlib.sha256(
                json.dumps(config.to_dict(), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "training_config_sha256": hashlib.sha256(
                json.dumps(training_config.to_dict(), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "run_arguments": run_arguments,
        }
    )
    atomic_json(output_dir / "run_metadata.json", run_metadata)
    event_logger.log(
        {
            "record_type": "run_start",
            "start_epoch": start_epoch,
            "total_epochs": args.epochs,
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "accumulation_steps": training_config.accumulation_steps,
            "amp_enabled": amp_settings.enabled,
            "amp_dtype": str(amp_settings.dtype),
            "optimizer_groups": optimizer_group_summary(optimizer),
            "run_metadata": run_metadata,
        }
    )

    if start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint already completed epoch {start_epoch - 1}; target epochs is {args.epochs}."
        )

    completed_epoch = start_epoch - 1
    stop_reason = "completed"
    last_structured_metrics = None
    last_generation_diagnostics = None
    for epoch in range(start_epoch, args.epochs + 1):
        completed_epoch = epoch
        train_loader = build_loader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            seed=args.seed + epoch - 1,
            pin_memory=pin_memory,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if (
            active_stage2_start_epoch > 0
            and epoch >= active_stage2_start_epoch
            and not stage2_activated
        ):
            model.set_visual_encoder_trainable(True, strategy=active_stage2_strategy)
            stage2_activated = True
            current_group_summary = optimizer_group_summary(optimizer)
            atomic_json(output_dir / "optimizer_groups.json", current_group_summary)
            event_logger.log(
                {
                    "record_type": "stage_transition",
                    "epoch": epoch,
                    "stage": 2,
                    "visual_unfreeze_strategy": active_stage2_strategy,
                    "visual_lr_scale": training_config.visual_lr_scale,
                    "optimizer_groups": current_group_summary,
                }
            )
            print(
                f"stage2_enabled epoch={epoch} strategy={active_stage2_strategy} "
                f"visual_lr_scale={training_config.visual_lr_scale}"
            )
        train_metrics, global_step = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            training_config.gradient_clip,
            epoch=epoch,
            log_every=args.log_every,
            split="train",
            max_steps_per_epoch=args.max_steps_per_epoch,
            scheduler=scheduler,
            scaler=scaler,
            amp_settings=amp_settings,
            event_logger=event_logger,
            global_step=global_step,
            global_micro_step=global_micro_step,
            accumulation_steps=training_config.accumulation_steps,
            max_optimizer_steps=(
                max(args.max_optimizer_steps - global_step, 0)
                if args.max_optimizer_steps > 0
                else 0
            ),
        )
        global_micro_step += int(train_metrics["steps"])
        global_examples += int(train_metrics["examples"])
        validation_result = (
            run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                gradient_clip=0.0,
                epoch=epoch,
                log_every=args.log_every,
                split="val",
                max_steps_per_epoch=args.max_steps_per_epoch,
                amp_settings=amp_settings,
                event_logger=event_logger,
                global_step=global_step,
                global_micro_step=global_micro_step,
            )
            if val_loader is not None
            else None
        )
        val_metrics = validation_result[0] if validation_result is not None else None
        selection_split = "validation" if val_metrics is not None else "training_fallback"
        checkpoint_identity = f"{run_metadata['run_id']}:epoch-{epoch:04d}"
        needs_structured_metrics = (
            val_dataset is not None
            and (
                training_config.monitor in metric_monitors
                or (args.metrics_every > 0 and epoch % args.metrics_every == 0)
            )
        )
        metrics_predictions = (
            collect_predictions(
                model,
                val_dataset,
                config,
                device,
                sample_limit=None,
                checkpoint_identity=checkpoint_identity,
            )
            if needs_structured_metrics and val_dataset is not None
            else None
        )
        structured_metrics = (
            evaluate_event_predictions(metrics_predictions)
            if metrics_predictions is not None
            else None
        )
        generation_diagnostics = (
            summarize_prediction_artifacts(metrics_predictions)
            if metrics_predictions is not None
            else None
        )
        if structured_metrics is not None:
            last_structured_metrics = structured_metrics
        if generation_diagnostics is not None:
            last_generation_diagnostics = generation_diagnostics
        if structured_metrics is not None:
            atomic_json(output_dir / "metrics.json", structured_metrics)
            atomic_json(output_dir / f"metrics-epoch-{epoch:04d}.json", structured_metrics)
            print(
                "metrics "
                + " ".join(f"{key}={value:.4f}" for key, value in structured_metrics.items())
            )

        if training_config.monitor == "train_loss":
            monitored_value = float(train_metrics["loss"])
        elif training_config.monitor == "val_loss":
            monitored_value = float(
                val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
            )
        else:
            assert structured_metrics is not None
            monitored_value = float(structured_metrics[training_config.monitor])

        validation_loss = float(val_metrics["loss"]) if val_metrics is not None else float(train_metrics["loss"])
        loss_improved = validation_loss < best_loss
        if loss_improved:
            best_loss = validation_loss
            best_loss_epoch = epoch
            best_loss_step = global_step
        improved, should_stop = early_stopping.update(
            monitored_value,
            epoch,
            global_step=global_step,
            active=global_step >= warmup_steps,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"] if val_metrics is not None else None,
            "train": train_metrics,
            "validation": val_metrics,
            "selection_split": selection_split,
            "improved": improved,
            "loss_improved": loss_improved,
            "monitor": training_config.monitor,
            "monitored_value": monitored_value,
            "structured_metrics": structured_metrics,
            "generation_diagnostics": generation_diagnostics,
            "global_step": global_step,
            "global_micro_step": global_micro_step,
            "global_examples": global_examples,
        }
        history.append(record)
        atomic_json(output_dir / "history.json", history)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            + (
                f"val_loss={val_metrics['loss']:.6f}"
                if val_metrics is not None
                else "val_loss=n/a"
            )
        )

        checkpoint_training_state = {
            "stage2_activated": stage2_activated,
            "stage2_unfreeze_strategy": active_stage2_strategy,
            "stage2_start_epoch": active_stage2_start_epoch,
            "stage2_visual_lr_scale": training_config.visual_lr_scale,
            "total_optimizer_steps": total_optimizer_steps,
            "warmup_steps": warmup_steps,
            "selection_split": selection_split,
            "deterministic": args.deterministic,
        }
        if args.prediction_every > 0 and epoch % args.prediction_every == 0:
            if metrics_predictions is not None:
                atomic_json(
                    prediction_dir / f"epoch-{epoch:04d}.json",
                    metrics_predictions[: args.prediction_samples],
                )
            else:
                prediction_dataset = val_dataset if val_dataset is not None else train_dataset
                prediction_subset = collect_predictions(
                    model,
                    prediction_dataset,
                    config,
                    device,
                    sample_limit=args.prediction_samples,
                    checkpoint_identity=checkpoint_identity,
                )
                atomic_json(prediction_dir / f"epoch-{epoch:04d}.json", prediction_subset)
                generation_diagnostics = summarize_prediction_artifacts(prediction_subset)
                last_generation_diagnostics = generation_diagnostics
            event_logger.log(
                {
                    "record_type": "validation_generation",
                    "epoch": epoch,
                    "global_step": global_step,
                    **(generation_diagnostics or {}),
                }
            )

        selection_state = {
            "monitor": training_config.monitor,
            "monitor_mode": training_config.monitor_mode,
            "best_primary_value": early_stopping.best,
            "best_primary_epoch": early_stopping.best_epoch,
            "best_primary_step": early_stopping.best_step,
            "best_loss": best_loss,
            "best_loss_epoch": best_loss_epoch,
            "best_loss_step": best_loss_step,
        }
        event_logger.log(
            {
                "record_type": "checkpoint_selection",
                "epoch": epoch,
                "global_step": global_step,
                "monitored_value": monitored_value,
                "primary_improved": improved,
                "loss_improved": loss_improved,
                **selection_state,
            }
        )
        checkpoint_arguments = dict(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            best_loss=best_loss,
            history=history,
            training_state=checkpoint_training_state,
            scheduler=scheduler,
            scaler=scaler,
            training_config=training_config,
            early_stopping=early_stopping,
            global_step=global_step,
            global_micro_step=global_micro_step,
            global_examples=global_examples,
            selection_state=selection_state,
            run_metadata=run_metadata,
            accumulation_state={"pending_micro_steps": 0},
        )
        save_checkpoint(checkpoint_dir / "latest.pt", **checkpoint_arguments)
        if loss_improved:
            save_checkpoint(checkpoint_dir / "best-loss.pt", **checkpoint_arguments)
        if improved:
            save_checkpoint(checkpoint_dir / "best-primary-metric.pt", **checkpoint_arguments)
            save_checkpoint(checkpoint_dir / "best.pt", **checkpoint_arguments)
            atomic_torch_save(
                output_dir / "tinytrace.pt",
                {
                    "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                    "artifact_type": "inference_checkpoint",
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "selection_state": selection_state,
                    "run_metadata": run_metadata,
                },
            )
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(checkpoint_dir / f"epoch-{epoch:04d}.pt", **checkpoint_arguments)
            prune_periodic_checkpoints(checkpoint_dir, training_config.checkpoint_keep)
        if should_stop:
            stop_reason = early_stopping.stop_reason or "early_stopping"
            event_logger.log(
                {
                    "record_type": "early_stop",
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "monitor": training_config.monitor,
                    "best_primary_value": early_stopping.best,
                    "best_primary_step": early_stopping.best_step,
                    "bad_epochs": early_stopping.bad_epochs,
                    "reason": stop_reason,
                }
            )
            print(
                f"early_stopping epoch={epoch} best_loss={best_loss:.6f} "
                f"bad_epochs={early_stopping.bad_epochs}"
            )
            break
        if args.max_optimizer_steps > 0 and global_step >= args.max_optimizer_steps:
            stop_reason = "max_optimizer_steps"
            event_logger.log(
                {
                    "record_type": "optimizer_step_cap",
                    "epoch": epoch,
                    "global_step": global_step,
                    "requested_optimizer_steps": args.max_optimizer_steps,
                }
            )
            break

    run_summary = {
        "run_id": run_metadata["run_id"],
        "status": (
            "completed"
            if stop_reason == "completed"
            else "step_limited"
            if stop_reason == "max_optimizer_steps"
            else "early_stopped"
        ),
        "stop_reason": stop_reason,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "global_micro_step": global_micro_step,
        "global_examples": global_examples,
        "best_loss": best_loss,
        "best_loss_epoch": best_loss_epoch,
        "best_primary_value": early_stopping.best,
        "best_primary_epoch": early_stopping.best_epoch,
        "best_primary_step": early_stopping.best_step,
        "monitor": training_config.monitor,
        "logging_failures": event_logger.failures,
        "final_structured_metrics": last_structured_metrics,
        "final_generation_diagnostics": last_generation_diagnostics,
    }
    atomic_json(output_dir / "run_summary.json", run_summary)
    event_logger.log({"record_type": "run_end", **run_summary})


def main() -> None:
    try:
        _run_training()
    except Exception as exc:
        args = parse_args()
        output_dir = Path(args.output_dir)
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "timestamp": time.time(),
        }
        if not isinstance(exc, FileExistsError):
            try:
                atomic_json(output_dir / "failure.json", failure)
                JsonlLogger(output_dir / "training_log.jsonl").log(
                    {"record_type": "run_failure", **failure}
                )
            except OSError:
                pass
        raise


if __name__ == "__main__":
    main()
