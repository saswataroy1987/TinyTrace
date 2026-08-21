"""Train isolated Stage 2 v3 direct-MobileCLIP captioning.

This script never writes the source cache and never loads or modifies either
previous Stage 2 experiment. It uses ground-truth event windows only; Stage 3
is intentionally outside this experiment.
"""

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

from tinytrace.phase_b_v2 import ActivityNetV2Dataset, PhaseBV2Config, Stage0Config, TemporalEventDetector, activitynet_v2_collate_fn
from tinytrace.phase_b_v2.caption import pool_event_features
from tinytrace.phase_b_v2.metrics import caption_metrics
from tinytrace.phase_b_v3 import DirectMobileCLIPCaptionConfig, DirectMobileCLIPCaptionModel, select_event_patch_features


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _flatten_events(batch: dict[str, object], config: DirectMobileCLIPCaptionConfig, stage1_detector: TemporalEventDetector | None = None) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor | None]:
    selected, selected_mask = select_event_patch_features(
        batch["visual_features"], batch["frame_times"], batch["frame_mask"], batch["segments"], config.max_event_frames  # type: ignore[arg-type]
    )
    rows: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    captions: list[str] = []
    context_rows: list[torch.Tensor] = []
    temporal_context = None
    if stage1_detector is not None:
        temporal = stage1_detector.encode(batch["visual_features"], batch["frame_times"], batch["frame_mask"])  # type: ignore[arg-type]
        temporal_context, _ = pool_event_features(temporal, batch["frame_times"], batch["frame_mask"], batch["segments"], config.temporal_context_tokens)  # type: ignore[arg-type]
    event_mask: torch.Tensor = batch["event_mask"]  # type: ignore[assignment]
    source_captions: list[list[str]] = batch["captions"]  # type: ignore[assignment]
    for batch_index, event_index in event_mask.nonzero(as_tuple=False).tolist():
        if not bool(selected_mask[batch_index, event_index].any()):
            continue
        rows.append(selected[batch_index, event_index])
        masks.append(selected_mask[batch_index, event_index])
        captions.append(source_captions[batch_index][event_index])
        if temporal_context is not None:
            context_rows.append(temporal_context[batch_index, event_index])
    if not rows:
        raise RuntimeError("Caption batch contains no usable ground-truth events.")
    return torch.stack(rows), torch.stack(masks), captions, torch.stack(context_rows) if context_rows else None


def _run_epoch(model: DirectMobileCLIPCaptionModel, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device, generation_limit: int, *, epoch: int, split: str, log_path: Path, log_every_videos: int, stage1_detector: TemporalEventDetector | None = None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if stage1_detector is not None:
        stage1_detector.train(training and any(parameter.requires_grad for parameter in stage1_detector.parameters()))
    loss_total, videos, emitted, seen = 0.0, 0, 0, 0
    generated: list[str] = []
    references: list[str] = []
    for batch_number, raw in enumerate(loader, start=1):
        batch = _to_device(raw, device)
        event_features, event_masks, captions, temporal_context = _flatten_events(batch, model.config, stage1_detector)
        loss = model(event_features, event_masks, captions, temporal_context)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
            optimizer.step()
        loss_total += float(loss.detach())
        videos += int(batch["visual_features"].size(0))  # type: ignore[union-attr]
        if not training and seen < generation_limit:
            output = model.generate(event_features, event_masks, temporal_context)
            take = min(generation_limit - seen, len(output))
            generated.extend(output[:take]); references.extend(captions[:take]); seen += take
        if log_every_videos and videos // log_every_videos > emitted // log_every_videos:
            emitted = videos
            record = {"epoch": epoch, "split": split, "videos": videos, "batches": batch_number, "caption_loss": loss_total / batch_number}
            print("progress " + json.dumps(record, sort_keys=True), flush=True)
            with log_path.open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(record, sort_keys=True) + "\n")
    return {"loss": loss_total / max(len(loader), 1), **caption_metrics(generated, references), "generated_match_coverage": 1.0 if references else 0.0}


def _shuffle_features(features: torch.Tensor, frame_mask: torch.Tensor, video_id: str) -> torch.Tensor:
    """Deterministically permute selected event frames while retaining timestamps."""
    valid = int(frame_mask.sum())
    if valid < 2:
        return features.clone()
    seed = int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(valid, generator=generator)
    result = features.clone()
    result[:valid] = features[:valid][permutation.to(features.device)]
    return result


@torch.no_grad()
def _audit_evaluation(model: DirectMobileCLIPCaptionModel, dataset: ActivityNetV2Dataset, audit_root: Path, output: Path, epoch: int, device: torch.device, stage1_detector: TemporalEventDetector | None = None) -> dict[str, float]:
    """Evaluate the fixed ten-video forensic set with real and shuffled evidence."""
    by_id = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}
    rows: list[dict[str, object]] = []
    real, shuffled, references, overlaps = [], [], [], []
    model.eval()
    for directory in sorted(audit_root.glob("video_*")):
        truth = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))["audit_focus_event"]
        video_id = directory.name.split("_", 2)[-1]
        if video_id not in by_id:
            raise ValueError(f"Audit video is absent from validation manifest: {video_id}")
        sample = dataset[by_id[video_id]]
        duration = float(sample["duration"])
        segment = torch.tensor([[float(truth["start"]) / duration, float(truth["end"]) / duration]], device=device).view(1, 1, 2)
        features = sample["visual_features"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        times = sample["frame_times"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        mask = torch.ones(times.shape, dtype=torch.bool, device=device)
        event_features, event_mask = select_event_patch_features(features, times, mask, segment, model.config.max_event_frames)
        event_features, event_mask = event_features[0], event_mask[0]
        temporal_context = None
        if stage1_detector is not None:
            encoded = stage1_detector.encode(features, times, mask)
            pooled, _ = pool_event_features(encoded, times, mask, segment, model.config.temporal_context_tokens)
            temporal_context = pooled[0]
        real_caption = model.generate(event_features, event_mask, temporal_context)[0]
        shuffled_features = _shuffle_features(event_features[0], event_mask[0], video_id).unsqueeze(0)
        shuffled_caption = model.generate(shuffled_features, event_mask, temporal_context)[0]
        reference = str(truth["reference_caption"])
        left, right = set(real_caption.lower().split()), set(shuffled_caption.lower().split())
        overlap = len(left & right) / len(left | right) if left or right else 1.0
        item = {"video_id": video_id, "reference_caption": reference, "ground_truth_start": float(truth["start"]), "ground_truth_end": float(truth["end"]), "real_caption": real_caption, "shuffled_caption": shuffled_caption, "token_overlap": overlap, "selected_event_frames": int(event_mask.sum()), "event_feature_shape": list(event_features.shape)}
        _atomic_json(output / f"epoch-{epoch:04d}" / f"{directory.name}.json", item)
        rows.append(item); real.append(real_caption); shuffled.append(shuffled_caption); references.append(reference); overlaps.append(overlap)
    real_metrics, shuffled_metrics = caption_metrics(real, references), caption_metrics(shuffled, references)
    summary = {"epoch": epoch, "videos": len(rows), "real": real_metrics, "shuffled": shuffled_metrics, "mean_real_vs_shuffled_token_overlap": sum(overlaps) / len(overlaps), "identical_caption_rate": sum(item["real_caption"] == item["shuffled_caption"] for item in rows) / len(rows), "comparisons": rows}
    _atomic_json(output / f"epoch-{epoch:04d}" / "summary.json", summary)
    headers = ["Video", "Reference", "Real", "Shuffled", "Overlap"]
    markdown = "# Stage 2 v3 Audit Evaluation\n\nThe event time windows and frame timestamps are identical in each pair. Only event-frame visual content is deterministically permuted for the shuffled condition.\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join((str(value).replace("|", "\\|")) for value in (row["video_id"], row["reference_caption"], row["real_caption"], row["shuffled_caption"], f'{row["token_overlap"]:.3f}')) + " |" for row in rows) + "\n\n```json\n" + json.dumps({key: value for key, value in summary.items() if key != "comparisons"}, indent=2) + "\n```\n"
    (output / f"epoch-{epoch:04d}" / "SUMMARY.md").write_text(markdown, encoding="utf-8")
    return {"audit_real_bleu1": float(real_metrics["bleu1"]), "audit_real_meteor_unigram": float(real_metrics["meteor_unigram"]), "audit_real_cider_unigram": float(real_metrics["cider_unigram"]), "audit_mean_real_vs_shuffled_token_overlap": float(summary["mean_real_vs_shuffled_token_overlap"]), "audit_identical_caption_rate": float(summary["identical_caption_rate"])}


def _checkpoint(path: Path, model: DirectMobileCLIPCaptionModel, optimizer: torch.optim.Optimizer, config: DirectMobileCLIPCaptionConfig, epoch: int, manifest_sha256: str, best_score: float) -> None:
    _atomic_torch(path, {"format_version": 1, "experiment": "stage2_v3_direct_mobileclip", "epoch": epoch, "model_config": config.to_dict(), "manifest_sha256": manifest_sha256, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_score": best_score, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()})


def _load_stage1_context(path: Path, config: DirectMobileCLIPCaptionConfig) -> TemporalEventDetector:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("model_config")
    if not isinstance(saved, dict):
        raise ValueError("Stage 1 checkpoint lacks its model configuration.")
    detector = TemporalEventDetector(PhaseBV2Config(**saved))
    state = {key.removeprefix("detector."): value for key, value in payload["model_state"].items() if key.startswith("detector.")}
    detector.load_state_dict(state, strict=True)
    for parameter in detector.parameters():
        parameter.requires_grad = False
    if config.train_stage1_temporal_layers:
        for block in list(detector.encoder.layers)[-config.train_stage1_temporal_layers :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--stage1-context-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--flan-learning-rate", type=float, default=2e-5)
    parser.add_argument("--generation-limit", type=int, default=64)
    parser.add_argument("--audit-every", type=int, default=2)
    parser.add_argument("--log-every-videos", type=int, default=50)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.adapter_learning_rate <= 0 or args.flan_learning_rate <= 0 or args.audit_every < 1:
        raise ValueError("epochs, batch-size, learning rates, and audit-every must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; run on the GPU PC.")
    config, stage0 = DirectMobileCLIPCaptionConfig.from_json(args.model_config), Stage0Config.from_json(args.stage0_config)
    if config.use_stage1_temporal_context != (args.stage1_context_checkpoint is not None):
        raise ValueError("--stage1-context-checkpoint is required exactly when use_stage1_temporal_context=true.")
    manifest_sha256 = _digest(args.manifest)
    train = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="train")
    validation = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    loaders = [DataLoader(source, batch_size=args.batch_size, shuffle=split == "train", collate_fn=activitynet_v2_collate_fn, num_workers=0) for source, split in ((train, "train"), (validation, "val"))]
    model = DirectMobileCLIPCaptionModel.from_pretrained(config).to(device)
    stage1_detector = _load_stage1_context(args.stage1_context_checkpoint, config).to(device) if args.stage1_context_checkpoint else None
    adapter_params = list(model.adapter.parameters())
    language_params = [parameter for name, parameter in model.named_parameters() if name.startswith("language_model") and parameter.requires_grad]
    groups = [{"params": adapter_params, "lr": args.adapter_learning_rate}, {"params": language_params, "lr": args.flan_learning_rate}]
    if stage1_detector is not None:
        stage1_params = [parameter for parameter in stage1_detector.parameters() if parameter.requires_grad]
        if stage1_params:
            groups.append({"params": stage1_params, "lr": args.flan_learning_rate / 2})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    start_epoch, best_score = 0, float("-inf")
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if payload.get("experiment") != "stage2_v3_direct_mobileclip" or payload.get("manifest_sha256") != manifest_sha256 or payload.get("model_config") != config.to_dict():
            raise ValueError("Resume checkpoint does not match this direct-MobileCLIP experiment.")
        model.load_state_dict(payload["model_state"]); optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch, best_score = int(payload["epoch"]), float(payload["best_score"])
        torch.set_rng_state(payload["torch_rng"]); random.setstate(payload["python_rng"])
    _atomic_json(args.output_root / "configs" / "resolved_training_config.json", {**config.to_dict(), "experiment": "stage2_v3_direct_mobileclip", "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_sha256, "cache_root": str(args.cache_root.resolve()), "cache_read_only": True, "adapter_learning_rate": args.adapter_learning_rate, "flan_learning_rate": args.flan_learning_rate, "audit_root": str(args.audit_root.resolve()), "audit_every": args.audit_every, "stage1_context_checkpoint": str(args.stage1_context_checkpoint.resolve()) if args.stage1_context_checkpoint else None})
    history_path = args.output_root / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if args.resume and history_path.is_file() else []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        training = _run_epoch(model, loaders[0], optimizer, device, args.generation_limit, epoch=epoch, split="train", log_path=args.output_root / "training_log.jsonl", log_every_videos=args.log_every_videos, stage1_detector=stage1_detector)
        with torch.no_grad():
            validation_metrics = _run_epoch(model, loaders[1], None, device, args.generation_limit, epoch=epoch, split="validation", log_path=args.output_root / "training_log.jsonl", log_every_videos=args.log_every_videos, stage1_detector=stage1_detector)
            if epoch % args.audit_every == 0 or epoch == args.epochs:
                validation_metrics.update(_audit_evaluation(model, validation, args.audit_root, args.output_root / "audit", epoch, device, stage1_detector))
        entry = {"epoch": epoch, "train": training, "validation": validation_metrics}
        history.append(entry)
        score = float(validation_metrics["bleu1"])
        if score > best_score:
            best_score = score
            _checkpoint(args.output_root / "checkpoints" / "best-caption.pt", model, optimizer, config, epoch, manifest_sha256, best_score)
        _checkpoint(args.output_root / "checkpoints" / "latest.pt", model, optimizer, config, epoch, manifest_sha256, best_score)
        _atomic_json(history_path, history)
        print(json.dumps(entry, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
