"""Train TinyTrace Phase B v2 Stage 1 or Stage 2 from the validated cache manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2 import ActivityNetV2Dataset, PhaseBV2Config, PhaseBV2Model, Stage0Config, activitynet_v2_collate_fn, filter_events
from tinytrace.phase_b_v2.metrics import caption_metrics, localization_metrics, matched_caption_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _progress(log_path: Path | None, *, stage: str, split: str, epoch: int, videos: int, batches: int, losses: dict[str, float]) -> None:
    record = {"stage": stage, "split": split, "epoch": epoch, "videos": videos, "batches": batches, **losses}
    print("progress " + json.dumps(record, sort_keys=True), flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, sort_keys=True) + "\n")


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _checkpoint(path: Path, model: PhaseBV2Model, optimizer: torch.optim.Optimizer, config: PhaseBV2Config, *, stage: str, epoch: int, manifest_sha256: str, best_score: float, component_best: dict[str, float]) -> None:
    _atomic_save(path, {"format_version": 1, "stage": stage, "epoch": epoch, "model_config": config.to_dict(), "manifest_sha256": manifest_sha256, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_score": best_score, "component_best": component_best, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()})


def _load(path: Path, model: PhaseBV2Model, optimizer: torch.optim.Optimizer | None, config: PhaseBV2Config, manifest_sha256: str) -> tuple[int, float, dict[str, float]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Checkpoint does not match the current V2 manifest.")
    saved = payload.get("model_config", {})
    if saved.get("feature_dim") != config.feature_dim or saved.get("d_model") != config.d_model or saved.get("event_queries") != config.event_queries:
        raise ValueError("Checkpoint architecture does not match the requested V2 configuration.")
    model.load_state_dict(payload["model_state"], strict=optimizer is not None)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
    component_best = payload.get("component_best", {})
    if not isinstance(component_best, dict):
        component_best = {}
    return int(payload["epoch"]), float(payload.get("best_score", float("-inf"))), {str(key): float(value) for key, value in component_best.items()}


def _run_localization(model: PhaseBV2Model, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, threshold: float, overlap: float, *, stage: str, split: str, epoch: int, log_every_videos: int, log_path: Path | None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total = event = l1 = giou = 0.0
    predictions: list[list[dict[str, float]]] = []
    target_rows: list[torch.Tensor] = []
    durations: list[float] = []
    videos = reported_videos = 0
    for batch_index, raw in enumerate(loader, start=1):
        batch = _to_device(raw, device)
        outputs, loss = model.forward_localization(batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += float(loss.total.detach()); event += float(loss.event.detach()); l1 += float(loss.l1.detach()); giou += float(loss.giou.detach())
        predictions.extend(filter_events(outputs["segments"][index].detach(), outputs["event_logits"][index].detach(), threshold, overlap) for index in range(outputs["segments"].size(0)))
        for sample_index in range(batch["segments"].size(0)):
            target_rows.append(batch["segments"][sample_index, batch["event_mask"][sample_index]].detach().cpu())
            durations.append(float(batch["duration"][sample_index]))
        videos += batch["segments"].size(0)
        if log_every_videos and videos // log_every_videos > reported_videos // log_every_videos:
            reported_videos = videos
            _progress(log_path, stage=stage, split=split, epoch=epoch, videos=videos, batches=batch_index, losses={"loss": total / batch_index, "event_loss": event / batch_index, "l1_loss": l1 / batch_index, "giou_loss": giou / batch_index})
    count = max(len(loader), 1)
    metrics = localization_metrics(predictions, target_rows, durations)
    return {"loss": total / count, "event_loss": event / count, "l1_loss": l1 / count, "giou_loss": giou / count, **metrics}


def _run_caption(model: PhaseBV2Model, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, generation_limit: int, *, stage: str, split: str, epoch: int, log_every_videos: int, log_path: Path | None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    model.detector.eval()  # Stage 2 freezes the Stage 1 temporal representation.
    loss_total = 0.0
    generated: list[str] = []
    references: list[str] = []
    seen = 0
    videos = reported_videos = 0
    for batch_index, raw in enumerate(loader, start=1):
        batch = _to_device(raw, device)
        loss, _ = model.forward_caption(batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            optimizer.step()
        loss_total += float(loss.detach())
        videos += batch["visual_features"].size(0)
        if log_every_videos and videos // log_every_videos > reported_videos // log_every_videos:
            reported_videos = videos
            _progress(log_path, stage=stage, split=split, epoch=epoch, videos=videos, batches=batch_index, losses={"caption_loss": loss_total / batch_index})
        if not training and seen < generation_limit:
            captions, batch_indices, event_indices = model.generate_ground_truth_events(batch)
            for caption, batch_index, event_index in zip(captions, batch_indices.tolist(), event_indices.tolist()):
                generated.append(caption); references.append(batch["captions"][batch_index][event_index]); seen += 1
                if seen >= generation_limit:
                    break
    return {"loss": loss_total / max(len(loader), 1), **caption_metrics(generated, references), "generated_match_coverage": 1.0 if references else 0.0}


def _run_joint(model: PhaseBV2Model, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, threshold: float, overlap: float, ground_truth_ratio: float) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total = localization = caption = 0.0
    predicted: list[list[dict[str, object]]] = []
    target_rows: list[torch.Tensor] = []
    durations: list[float] = []
    captions: list[list[str]] = []
    for raw in loader:
        batch = _to_device(raw, device)
        local_loss, caption_loss, _ = model.forward_joint(batch, ground_truth_ratio if training else 0.0)
        loss = local_loss.total + model.config.loss_caption_weight * caption_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            optimizer.step()
        total += float(loss.detach()); localization += float(local_loss.total.detach()); caption += float(caption_loss.detach())
        if not training:
            predicted.extend(model.predict_events(batch, threshold=threshold, overlap_threshold=overlap))
            for sample_index in range(batch["segments"].size(0)):
                target_rows.append(batch["segments"][sample_index, batch["event_mask"][sample_index]].detach().cpu())
                durations.append(float(batch["duration"][sample_index]))
                captions.append(batch["captions"][sample_index])
    result = {"loss": total / max(len(loader), 1), "localization_loss": localization / max(len(loader), 1), "caption_loss": caption / max(len(loader), 1)}
    if not training:
        result.update(localization_metrics(predicted, target_rows, durations))
        result.update(matched_caption_metrics(predicted, target_rows, captions))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("localization", "caption", "joint"), required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--stage2-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--overlap-threshold", type=float, default=0.7)
    parser.add_argument("--generation-limit", type=int, default=64)
    parser.add_argument("--log-every-videos", type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0 or args.log_every_videos < 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive; log-every-videos must be non-negative.")
    if args.stage == "caption" and args.stage1_checkpoint is None and args.resume is None:
        raise ValueError("Stage 2 requires --stage1-checkpoint (or a Stage 2 --resume checkpoint).")
    if args.stage == "joint" and (args.stage1_checkpoint is None or args.stage2_checkpoint is None) and args.resume is None:
        raise ValueError("Stage 3 requires both --stage1-checkpoint and --stage2-checkpoint (or a Stage 3 --resume checkpoint).")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; run this training command on the GPU PC.")
    config = PhaseBV2Config.from_json(args.model_config)
    if config.stage != args.stage:
        config = PhaseBV2Config(**{**config.to_dict(), "stage": args.stage})
    stage0 = Stage0Config.from_json(args.stage0_config)
    manifest_sha256 = _sha256(args.manifest)
    train = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="train")
    validation = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train", collate_fn=activitynet_v2_collate_fn, num_workers=0) for dataset, split in ((train, "train"), (validation, "val"))]
    model = PhaseBV2Model.for_language_stage(config) if args.stage in {"caption", "joint"} else PhaseBV2Model(config)
    if args.stage == "caption":
        model.freeze_temporal_encoder()
    model.to(device)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=0.01)
    start_epoch, best_score, component_best = 0, float("-inf"), {"temporal": float("-inf"), "caption": float("-inf")}
    if args.resume:
        start_epoch, best_score, restored_components = _load(args.resume, model, optimizer, config, manifest_sha256)
        component_best.update(restored_components)
    elif args.stage2_checkpoint:
        _load(args.stage2_checkpoint, model, None, config, manifest_sha256)
    elif args.stage1_checkpoint:
        _load(args.stage1_checkpoint, model, None, config, manifest_sha256)
    _atomic_json(args.output_root / "configs" / "resolved_training_config.json", {**config.to_dict(), "stage0_config": str(args.stage0_config.resolve()), "manifest": str(args.manifest.resolve()), "cache_root": str(args.cache_root.resolve()), "manifest_sha256": manifest_sha256, "stage": args.stage, "threshold": args.threshold, "overlap_threshold": args.overlap_threshold})
    history_path = args.output_root / "history.json"
    history: list[object] = []
    if args.resume and history_path.is_file():
        existing_history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(existing_history, list):
            raise ValueError(f"Expected a JSON list in resume history: {history_path}")
        history = existing_history
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if args.stage == "localization":
            training = _run_localization(model, loaders[0], optimizer, device, args.threshold, args.overlap_threshold, stage=args.stage, split="train", epoch=epoch, log_every_videos=args.log_every_videos, log_path=args.output_root / "training_log.jsonl")
            with torch.no_grad(): validation_metrics = _run_localization(model, loaders[1], None, device, args.threshold, args.overlap_threshold, stage=args.stage, split="validation", epoch=epoch, log_every_videos=args.log_every_videos, log_path=args.output_root / "training_log.jsonl")
            score, best_name = validation_metrics["f1"], "best-temporal.pt"
        elif args.stage == "caption":
            training = _run_caption(model, loaders[0], optimizer, device, args.generation_limit, stage=args.stage, split="train", epoch=epoch, log_every_videos=args.log_every_videos, log_path=args.output_root / "training_log.jsonl")
            with torch.no_grad(): validation_metrics = _run_caption(model, loaders[1], None, device, args.generation_limit, stage=args.stage, split="validation", epoch=epoch, log_every_videos=args.log_every_videos, log_path=args.output_root / "training_log.jsonl")
            score, best_name = validation_metrics["bleu1"], "best-caption.pt"
        else:
            training = _run_joint(model, loaders[0], optimizer, device, args.threshold, args.overlap_threshold, config.joint_ground_truth_segment_ratio)
            with torch.no_grad(): validation_metrics = _run_joint(model, loaders[1], None, device, args.threshold, args.overlap_threshold, 0.0)
            score, best_name = validation_metrics["f1"] + validation_metrics["bleu1"], "best-combined.pt"
        history.append({"epoch": epoch, "train": training, "validation": validation_metrics})
        if score > best_score:
            best_score = score
            _checkpoint(args.output_root / "checkpoints" / best_name, model, optimizer, config, stage=args.stage, epoch=epoch, manifest_sha256=manifest_sha256, best_score=best_score, component_best=component_best)
        if args.stage == "joint":
            if validation_metrics["f1"] > component_best["temporal"]:
                component_best["temporal"] = validation_metrics["f1"]
                _checkpoint(args.output_root / "checkpoints" / "best-temporal.pt", model, optimizer, config, stage=args.stage, epoch=epoch, manifest_sha256=manifest_sha256, best_score=component_best["temporal"], component_best=component_best)
            if validation_metrics["bleu1"] > component_best["caption"]:
                component_best["caption"] = validation_metrics["bleu1"]
                _checkpoint(args.output_root / "checkpoints" / "best-caption.pt", model, optimizer, config, stage=args.stage, epoch=epoch, manifest_sha256=manifest_sha256, best_score=component_best["caption"], component_best=component_best)
        _checkpoint(args.output_root / "checkpoints" / "latest.pt", model, optimizer, config, stage=args.stage, epoch=epoch, manifest_sha256=manifest_sha256, best_score=best_score, component_best=component_best)
        _atomic_json(history_path, history)
        print(json.dumps(history[-1], sort_keys=True))


if __name__ == "__main__":
    main()
