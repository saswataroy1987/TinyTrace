"""Train B1/B2 Stage 2 bridge ablations without altering the v3 experiment."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_phase_b_v3_direct_mobileclip import _atomic_json, _atomic_torch, _audit_evaluation, _digest, _flatten_events, _run_epoch, _to_device
from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config, activitynet_v2_collate_fn
from tinytrace.phase_b_bridge_ablation import BridgeAblationCaptionModel, build_bridge
from tinytrace.phase_b_v3 import DirectMobileCLIPCaptionConfig, DirectMobileCLIPCaptionModel, select_event_patch_features


@torch.no_grad()
def _zero_control(model: BridgeAblationCaptionModel, dataset: ActivityNetV2Dataset, audit_root: Path, output: Path, epoch: int, device: torch.device) -> None:
    """Save the required zero-visual captions alongside real/shuffled audit output."""
    lookup = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}
    rows = []
    model.eval()
    for directory in sorted(audit_root.glob("video_*")):
        focus = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))["audit_focus_event"]
        video_id = directory.name.split("_", 2)[-1]
        sample = dataset[lookup[video_id]]
        duration = float(sample["duration"])
        segment = torch.tensor([float(focus["start"]) / duration, float(focus["end"]) / duration], device=device).view(1, 1, 2)
        features = sample["visual_features"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        times = sample["frame_times"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        frame_mask = torch.ones(times.shape, dtype=torch.bool, device=device)
        selected, event_mask = select_event_patch_features(features, times, frame_mask, segment, model.config.max_event_frames)
        zero_caption = model.generate(torch.zeros_like(selected[0]), event_mask[0])[0]
        rows.append({"video_id": video_id, "reference_caption": str(focus["reference_caption"]), "zero_caption": zero_caption})
    _atomic_json(output / f"epoch-{epoch:04d}" / "zero_visual_captions.json", {"epoch": epoch, "captions": rows})


def _build(config: DirectMobileCLIPCaptionConfig, variant: str) -> BridgeAblationCaptionModel:
    # Use the exact same FLAN-T5 initialization/freezing implementation as v3.
    base = DirectMobileCLIPCaptionModel.from_pretrained(config)
    model = BridgeAblationCaptionModel(config, base.language_model, base.tokenizer, bridge_name=variant)
    model.adapter = build_bridge(variant, config, int(model.language_model.config.d_model))
    return model


def _checkpoint(path: Path, model: BridgeAblationCaptionModel, optimizer: torch.optim.Optimizer, config: DirectMobileCLIPCaptionConfig, variant: str, epoch: int, manifest_sha256: str, best_score: float, seed: int) -> None:
    _atomic_torch(path, {"format_version": 1, "experiment": "stage2_bridge_ablation", "variant": variant, "epoch": epoch, "model_config": config.to_dict(), "manifest_sha256": manifest_sha256, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_score": best_score, "seed": seed, "torch_rng": torch.get_rng_state(), "python_rng": random.getstate()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("b1", "b2", "c1"), required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--flan-learning-rate", type=float, default=2e-5)
    parser.add_argument("--generation-limit", type=int, default=64)
    parser.add_argument("--audit-every", type=int, default=2)
    parser.add_argument("--log-every-videos", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {args.output_root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; run on the GPU machine.")
    if args.epochs < 1 or args.batch_size < 1 or args.audit_every < 1:
        raise ValueError("epochs, batch-size, and audit-every must be positive.")
    random.seed(args.seed); torch.manual_seed(args.seed)
    config, stage0 = DirectMobileCLIPCaptionConfig.from_json(args.model_config), Stage0Config.from_json(args.stage0_config)
    if config.use_stage1_temporal_context:
        raise ValueError("Bridge ablations require Stage 1 context disabled.")
    manifest_sha256 = _digest(args.manifest)
    train = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="train")
    validation = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    loaders = [DataLoader(source, batch_size=args.batch_size, shuffle=split == "train", collate_fn=activitynet_v2_collate_fn, num_workers=0) for source, split in ((train, "train"), (validation, "val"))]
    model = _build(config, args.variant).to(device)
    instruction_tokens = len(model.tokenizer(config.instruction)["input_ids"])
    visual_prefix_max = config.max_event_frames * config.patch_tokens if args.variant == "c1" else config.max_event_frames
    context_contract = {"visual_tokens_per_event_max": visual_prefix_max, "instruction_tokens": instruction_tokens, "total_encoder_prefix_tokens_max": visual_prefix_max + instruction_tokens, "flan_config_n_positions": getattr(model.language_model.config, "n_positions", None), "visual_tokens_truncated": False}
    if args.variant == "c1":
        # T5 uses relative position bias and the actual frozen model accepts
        # this 522-token prefix; test it explicitly instead of truncating.
        probe_features = torch.zeros((1, config.max_event_frames, config.patch_tokens, config.feature_dim), device=device)
        probe_mask = torch.ones((1, config.max_event_frames), dtype=torch.bool, device=device)
        with torch.no_grad():
            embeddings, attention = model.conditioning(probe_features, probe_mask)
            model.language_model.encoder(inputs_embeds=embeddings, attention_mask=attention, return_dict=True)
        context_contract["explicit_encoder_forward_passed"] = True
    adapter_params = list(model.adapter.parameters())
    language_params = [parameter for name, parameter in model.named_parameters() if name.startswith("language_model") and parameter.requires_grad]
    optimizer = torch.optim.AdamW([{"params": adapter_params, "lr": args.adapter_learning_rate}, {"params": language_params, "lr": args.flan_learning_rate}], weight_decay=0.01)
    trainability = {"adapter_parameters": sum(parameter.numel() for parameter in adapter_params), "flan_trainable_parameters": sum(parameter.numel() for parameter in language_params), "flan_frozen_parameters": sum(parameter.numel() for parameter in model.language_model.parameters() if not parameter.requires_grad), "mobileclip_trainable_parameters": 0, "stage1_trainable_parameters": 0}
    _atomic_json(args.output_root / "configs" / "resolved_training_config.json", {**config.to_dict(), "experiment": "stage2_bridge_ablation", "variant": args.variant, "seed": args.seed, "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_sha256, "cache_root": str(args.cache_root.resolve()), "cache_read_only": True, "adapter_learning_rate": args.adapter_learning_rate, "flan_learning_rate": args.flan_learning_rate, "audit_every": args.audit_every, "trainability": trainability, "context_contract": context_contract})
    history, best_score = [], float("-inf")
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        training = _run_epoch(model, loaders[0], optimizer, device, args.generation_limit, epoch=epoch, split="train", log_path=args.output_root / "training_log.jsonl", log_every_videos=args.log_every_videos)
        train_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        with torch.no_grad():
            validation_metrics = _run_epoch(model, loaders[1], None, device, args.generation_limit, epoch=epoch, split="validation", log_path=args.output_root / "training_log.jsonl", log_every_videos=args.log_every_videos)
            if epoch % args.audit_every == 0 or epoch == args.epochs:
                validation_metrics.update(_audit_evaluation(model, validation, args.audit_root, args.output_root / "audit", epoch, device))
                _zero_control(model, validation, args.audit_root, args.output_root / "audit", epoch, device)
        runtime = {"train_seconds": train_seconds, "validation_and_audit_seconds": time.perf_counter() - validation_started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None, "visual_tokens_per_event": config.max_event_frames * config.patch_tokens if args.variant == "c1" else config.max_event_frames, "flan_visual_prefix_max_tokens": config.max_event_frames * config.patch_tokens if args.variant == "c1" else config.max_event_frames}
        item = {"epoch": epoch, "train": training, "validation": validation_metrics, "runtime": runtime}
        history.append(item)
        score = float(validation_metrics["bleu1"])
        if score > best_score:
            best_score = score
            _checkpoint(args.output_root / "checkpoints" / "best-caption.pt", model, optimizer, config, args.variant, epoch, manifest_sha256, best_score, args.seed)
        _checkpoint(args.output_root / "checkpoints" / "latest.pt", model, optimizer, config, args.variant, epoch, manifest_sha256, best_score, args.seed)
        _atomic_json(args.output_root / "history.json", history)
        print(json.dumps(item, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
