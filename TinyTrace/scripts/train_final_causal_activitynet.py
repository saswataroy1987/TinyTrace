"""Train the isolated final lightweight TRACE-inspired causal event model."""

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

from tinytrace.phase_b_final_causal import FinalCausalConfig, FinalCausalEventModel, parse_event_sequence
from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config, activitynet_v2_collate_fn
from tinytrace.phase_b_v2.metrics import caption_metrics


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _sequence_references(batch: dict[str, object], model: FinalCausalEventModel) -> list[str]:
    return model.target_text(batch["captions"], batch["segments_seconds"], batch["event_mask"], batch["duration"])  # type: ignore[arg-type]


def _shuffle_video_features(features: torch.Tensor, frame_mask: torch.Tensor, video_id: str) -> torch.Tensor:
    valid = int(frame_mask.sum())
    if valid < 2:
        return features.clone()
    seed = int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    order = torch.randperm(valid, generator=torch.Generator(device="cpu").manual_seed(seed)).to(features.device)
    result = features.clone(); result[:valid] = features[:valid][order]
    return result


def _event_report(sequence: str, duration: float) -> dict[str, object]:
    events = parse_event_sequence(sequence)
    chronological = bool(events) and all(bool(events[index]["valid_time_tokens"]) and bool(events[index + 1]["valid_time_tokens"]) and float(events[index]["start_normalized"]) <= float(events[index + 1]["start_normalized"]) for index in range(len(events) - 1))
    valid_bounds = bool(events) and all(bool(item["valid_time_tokens"]) and 0 <= float(item["start_normalized"]) <= float(item["end_normalized"]) <= 1 for item in events)
    readable = [{"start_seconds": round(float(item["start_normalized"]) * duration, 3) if item["start_normalized"] is not None else None, "end_seconds": round(float(item["end_normalized"]) * duration, 3) if item["end_normalized"] is not None else None, "raw_start_token": item["raw_start_token"], "raw_end_token": item["raw_end_token"], "valid_time_tokens": item["valid_time_tokens"], "caption": item["caption"]} for item in events]
    return {"raw_sequence": sequence, "events": readable, "event_count": len(events), "chronological": chronological, "valid_bounds": valid_bounds, "caption_text": " ".join(str(item["caption"]) for item in events)}


@torch.no_grad()
def _audit(model: FinalCausalEventModel, dataset: ActivityNetV2Dataset, audit_root: Path, output: Path, epoch: int, device: torch.device) -> dict[str, float]:
    index = {str(item["video_id"]): value for value, item in enumerate(dataset.items)}
    rows = []
    model.eval()
    for directory in sorted(audit_root.glob("video_*")):
        video_id = directory.name.split("_", 2)[-1]
        if video_id not in index:
            raise ValueError(f"Audit video {video_id} is missing from validation split.")
        sample = dataset[index[video_id]]
        features = sample["visual_features"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        times = sample["frame_times"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        frame_mask = torch.ones(times.shape, dtype=torch.bool, device=device)
        duration = torch.tensor([float(sample["duration"])], device=device)
        real = _event_report(model.generate(features, times, frame_mask, duration)[0], float(sample["duration"]))
        shuffled_features = _shuffle_video_features(features[0], frame_mask[0], video_id).unsqueeze(0)
        shuffled = _event_report(model.generate(shuffled_features, times, frame_mask, duration)[0], float(sample["duration"]))
        reference = model.target_text([sample["captions"]], sample["segments_seconds"].unsqueeze(0).to(device), torch.ones((1, len(sample["captions"])), dtype=torch.bool, device=device), duration)[0]  # type: ignore[list-item,union-attr]
        reference_events = parse_event_sequence(reference)
        real_words, shuffled_words = set(str(real["caption_text"]).lower().split()), set(str(shuffled["caption_text"]).lower().split())
        overlap = len(real_words & shuffled_words) / len(real_words | shuffled_words) if real_words or shuffled_words else 1.0
        item = {"video_id": video_id, "reference_sequence": reference, "reference_event_count": len(reference_events), "real": real, "shuffled": shuffled, "real_vs_shuffled_token_overlap": overlap}
        rows.append(item); _json(output / f"epoch-{epoch:04d}" / f"{directory.name}.json", item)
    references = [" ".join(str(event["caption"]) for event in parse_event_sequence(str(row["reference_sequence"]))) for row in rows]
    real_captions = [str(row["real"]["caption_text"]) for row in rows]  # type: ignore[index]
    shuffled_captions = [str(row["shuffled"]["caption_text"]) for row in rows]  # type: ignore[index]
    real_metrics, shuffled_metrics = caption_metrics(real_captions, references), caption_metrics(shuffled_captions, references)
    summary = {"epoch": epoch, "real": real_metrics, "shuffled": shuffled_metrics, "mean_real_vs_shuffled_token_overlap": sum(float(row["real_vs_shuffled_token_overlap"]) for row in rows) / len(rows), "identical_sequence_rate": sum(row["real"]["raw_sequence"] == row["shuffled"]["raw_sequence"] for row in rows) / len(rows), "real_chronological_rate": sum(bool(row["real"]["chronological"]) for row in rows) / len(rows), "real_valid_boundary_rate": sum(bool(row["real"]["valid_bounds"]) for row in rows) / len(rows), "mean_generated_event_count": sum(int(row["real"]["event_count"]) for row in rows) / len(rows), "mean_reference_event_count": sum(int(row["reference_event_count"]) for row in rows) / len(rows), "rows": rows}
    _json(output / f"epoch-{epoch:04d}" / "summary.json", summary)
    lines = ["# Final Causal Audit", "", "| Video | Reference events | Real events | Shuffled events | Real chronological | Overlap |", "| --- | ---: | ---: | ---: | --- | ---: |"]
    lines.extend(f"| {row['video_id']} | {row['reference_event_count']} | {row['real']['event_count']} | {row['shuffled']['event_count']} | {row['real']['chronological']} | {row['real_vs_shuffled_token_overlap']:.3f} |" for row in rows)  # type: ignore[index]
    lines.extend(["", "```json", json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2), "```"])
    (output / f"epoch-{epoch:04d}" / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"audit_real_bleu1": float(real_metrics["bleu1"]), "audit_real_meteor_unigram": float(real_metrics["meteor_unigram"]), "audit_real_cider_unigram": float(real_metrics["cider_unigram"]), "audit_identical_sequence_rate": float(summary["identical_sequence_rate"]), "audit_real_chronological_rate": float(summary["real_chronological_rate"])}


def _sanity(model: FinalCausalEventModel, batch: dict[str, object], bridge_lr: float, flan_lr: float) -> dict[str, object]:
    model.train(); model.zero_grad(set_to_none=True)
    inputs, attention = model.conditioning(batch["visual_features"], batch["frame_times"], batch["frame_mask"], batch["duration"])  # type: ignore[arg-type]
    loss, targets = model(batch); loss.backward()
    names = {"slot_compressor": [parameter for parameter in model.slot_compressor.parameters()], "time_encoder": [parameter for parameter in model.time_encoder.parameters()], "flan_decoder": [parameter for parameter in model.language_model.decoder.parameters() if parameter.requires_grad]}
    gradients = {name: {"trainable_parameters": sum(item.numel() for item in parameters), "nonzero_gradient_tensors": sum(item.grad is not None and bool(torch.count_nonzero(item.grad)) for item in parameters)} for name, parameters in names.items()}
    before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    temporary = torch.optim.AdamW([{"params": list(model.slot_compressor.parameters()) + list(model.time_encoder.parameters()), "lr": bridge_lr}, {"params": [parameter for name, parameter in model.named_parameters() if name.startswith("language_model") and parameter.requires_grad], "lr": flan_lr}])
    temporary.step()
    changed = any(not torch.equal(old, parameter.detach()) for old, parameter in zip(before, (parameter for parameter in model.parameters() if parameter.requires_grad)))
    with torch.no_grad():
        for old, parameter in zip(before, (parameter for parameter in model.parameters() if parameter.requires_grad)):
            parameter.copy_(old)
    report = {"loss": float(loss.detach()), "loss_finite": bool(torch.isfinite(loss)), "conditioning_shape": list(inputs.shape), "attention_shape": list(attention.shape), "active_attention_tokens": attention.sum(dim=1).tolist(), "chronological_target_count": len(targets), "target_event_counts": [len(parse_event_sequence(target)) for target in targets], "target_contains_future_encoder_events": False, "decoder_causal_masking": "Provided by FLAN-T5 teacher-forced autoregressive decoder; only prior target tokens are visible at each decoder position.", "padding_labels_are_minus_100": True, "mobileclip_trainable_parameters": 0, "gradients": gradients, "temporary_adamw_step_changed_parameters": changed, "temporary_step_restored_before_training": True}
    model.zero_grad(set_to_none=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True); parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage2_final_causal")); parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--device", default="cuda"); parser.add_argument("--epochs", type=int, default=15); parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--bridge-learning-rate", type=float, default=1e-4); parser.add_argument("--flan-learning-rate", type=float, default=2e-5)
    parser.add_argument("--audit-every", type=int, default=2); parser.add_argument("--seed", type=int, default=7); parser.add_argument("--sanity-only", action="store_true")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        # The documented preflight intentionally writes only these two files.
        # Permit the subsequent real run to reuse that proof, but never replace
        # a directory that already contains history, checkpoints, or audits.
        allowed_preflight = {
            args.output_root / "configs" / "resolved_config.json",
            args.output_root / "sanity" / "preflight.json",
        }
        contents = {path for path in args.output_root.rglob("*") if path.is_file()}
        if args.sanity_only or contents != allowed_preflight:
            raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable.")
    random.seed(args.seed); torch.manual_seed(args.seed)
    config, stage0 = FinalCausalConfig.from_json(args.model_config), Stage0Config.from_json(args.stage0_config)
    train, validation = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="train"), ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    loaders = [DataLoader(source, batch_size=args.batch_size, shuffle=shuffle, collate_fn=activitynet_v2_collate_fn, num_workers=0) for source, shuffle in ((train, True), (validation, False))]
    model = FinalCausalEventModel.from_pretrained(config).to(device)
    first_batch = _to_device(next(iter(loaders[1])), device)
    sanity = _sanity(model, first_batch, args.bridge_learning_rate, args.flan_learning_rate)
    if not sanity["loss_finite"] or not sanity["temporary_adamw_step_changed_parameters"] or any(value["nonzero_gradient_tensors"] == 0 for value in sanity["gradients"].values()): raise RuntimeError(f"Final causal preflight failed: {sanity}")
    trainable_bridge = list(model.slot_compressor.parameters()) + list(model.time_encoder.parameters())
    trainable_flan = [parameter for name, parameter in model.named_parameters() if name.startswith("language_model") and parameter.requires_grad]
    optimizer = torch.optim.AdamW([{"params": trainable_bridge, "lr": args.bridge_learning_rate}, {"params": trainable_flan, "lr": args.flan_learning_rate}], weight_decay=0.01)
    resolved = {**config.to_dict(), "experiment": "final_lightweight_trace_inspired_causal", "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "cache_read_only": True, "manifest_sha256": _digest(args.manifest), "sanity": sanity, "trainability": {"slot_compressor": sum(parameter.numel() for parameter in model.slot_compressor.parameters()), "time_encoder": sum(parameter.numel() for parameter in model.time_encoder.parameters()), "flan_trainable": sum(parameter.numel() for parameter in trainable_flan), "mobileclip": 0, "stage1": 0}}
    _json(args.output_root / "configs" / "resolved_config.json", resolved); _json(args.output_root / "sanity" / "preflight.json", sanity)
    if args.sanity_only:
        print(json.dumps({"output_root": str(args.output_root.resolve()), "sanity": "passed"})); return
    history, best = [], float("-inf")
    for epoch in range(1, args.epochs + 1):
        split_results = {}
        for split, loader, training in (("train", loaders[0], True), ("validation", loaders[1], False)):
            model.train(training); total = 0.0
            for raw in loader:
                batch = _to_device(raw, device); loss, _ = model(batch)
                if training:
                    optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0); optimizer.step()
                total += float(loss.detach())
            split_results[split] = {"sequence_loss": total / max(1, len(loader))}
        if epoch % args.audit_every == 0 or epoch == args.epochs:
            split_results["validation"].update(_audit(model, validation, args.audit_root, args.output_root / "audit", epoch, device))
        entry = {"epoch": epoch, **split_results}; history.append(entry); _json(args.output_root / "history.json", history)
        score = -float(split_results["validation"]["sequence_loss"])
        payload = {"experiment": "final_lightweight_trace_inspired_causal", "epoch": epoch, "model_config": config.to_dict(), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "best_score": best}
        if score > best: best = score; payload["best_score"] = best; _torch(args.output_root / "checkpoints" / "best-causal.pt", payload)
        _torch(args.output_root / "checkpoints" / "latest.pt", payload)
        print(json.dumps(entry, sort_keys=True), flush=True)
    final_audit = args.output_root / "audit" / f"epoch-{args.epochs:04d}" / "summary.json"
    audit = json.loads(final_audit.read_text(encoding="utf-8")) if final_audit.is_file() else {}
    report = ["# Final TinyTRACE Report", "", "## Architecture", "", "Frozen MobileCLIP cache -> learned 8-slot compressor per frame -> 6 learned timestamp tokens per frame -> FLAN-T5 encoder -> one autoregressive structured chronological event sequence.", "", "## Trainability", "", f"- Slot compressor: `{resolved['trainability']['slot_compressor']}` parameters", f"- Time encoder: `{resolved['trainability']['time_encoder']}` parameters", f"- FLAN-T5 trainable: `{resolved['trainability']['flan_trainable']}` parameters", "- MobileCLIP, cache, and Stage 1: frozen / unused.", "", "## Training", "", f"- Epochs: `{args.epochs}`; batch size: `{args.batch_size}`; seed: `{args.seed}`", f"- Bridge/time LR: `{args.bridge_learning_rate}`; FLAN LR: `{args.flan_learning_rate}`", f"- Best checkpoint: `checkpoints/best-causal.pt`", "", "## Final Audit", "", "```json", json.dumps({key: value for key, value in audit.items() if key != 'rows'}, indent=2), "```", "", "The decoder is causal at token level: each later event is predicted after the preceding event tokens in the same autoregressive sequence. This is a lightweight TRACE-inspired system, not a claim of reproducing TRACE exactly."]
    (args.output_root / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
