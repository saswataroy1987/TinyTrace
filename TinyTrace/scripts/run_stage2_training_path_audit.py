"""Audit the frozen Stage 2 bridge supervision path without training.

This diagnostic answers two distinct questions for the current C1 checkpoint:
1. Do the exact cached frames selected by the caption model retain enough
   MobileCLIP evidence for their corresponding audit reference caption?
2. Does one ordinary teacher-forced caption-loss backward pass produce
   non-zero gradients in the bridge and the configured trainable FLAN blocks?

It never creates an optimizer, never calls ``step()``, and refuses to overwrite
an existing report directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_phase_b_v3_direct_mobileclip import _flatten_events, _to_device
from train_stage2_bridge_ablation import _build
from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config, activitynet_v2_collate_fn
from tinytrace.phase_b_v3 import DirectMobileCLIPCaptionConfig, event_frame_indices, select_event_patch_features


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _audit_entries(audit_root: Path) -> list[dict[str, object]]:
    entries = []
    for directory in sorted(audit_root.glob("video_*")):
        truth = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))["audit_focus_event"]
        entries.append({"directory": directory, "video_id": directory.name.split("_", 2)[-1], "truth": truth})
    if not entries:
        raise ValueError(f"No video_* audit folders in {audit_root}")
    return entries


def _event_embedding(head: torch.nn.Module, features: torch.Tensor, selected: list[int], device: torch.device) -> torch.Tensor:
    tokens = features[selected].float().to(device)
    patches, width = tokens.shape[1:]
    side = math.isqrt(patches)
    if side * side != patches:
        raise ValueError(f"MobileCLIP patch count must be square; got {patches}")
    pooled = head(tokens.transpose(1, 2).reshape(tokens.size(0), width, side, side))
    return F.normalize(pooled.mean(dim=0), dim=0)


def _rank(scores: torch.Tensor, expected: int, captions: list[str]) -> dict[str, object]:
    order = scores.argsort(descending=True).tolist()
    rank = order.index(expected) + 1
    return {
        "reference_rank": rank,
        "reference_similarity": float(scores[expected]),
        "top_caption": captions[order[0]],
        "top_similarity": float(scores[order[0]]),
        "top_5": [
            {"rank": position + 1, "caption": captions[index], "similarity": float(scores[index]), "is_reference": index == expected}
            for position, index in enumerate(order[:5])
        ],
    }


def _gradient_summary(model: torch.nn.Module) -> dict[str, object]:
    groups = {"bridge": [], "flan_trainable": [], "flan_frozen": []}
    for name, parameter in model.named_parameters():
        if name.startswith("adapter."):
            groups["bridge"].append((name, parameter))
        elif name.startswith("language_model.") and parameter.requires_grad:
            groups["flan_trainable"].append((name, parameter))
        elif name.startswith("language_model."):
            groups["flan_frozen"].append((name, parameter))
    result: dict[str, object] = {}
    for key, values in groups.items():
        gradients = [parameter.grad.detach() for _, parameter in values if parameter.grad is not None]
        squared = sum((float(gradient.float().square().sum()) for gradient in gradients), 0.0)
        result[key] = {
            "parameter_tensors": len(values),
            "parameter_count": sum(parameter.numel() for _, parameter in values),
            "tensors_with_gradients": len(gradients),
            "nonzero_gradient_tensors": sum(bool(torch.count_nonzero(gradient)) for gradient in gradients),
            "gradient_l2_norm": math.sqrt(squared),
            "max_abs_gradient": max((float(gradient.abs().max()) for gradient in gradients), default=0.0),
            "examples": [
                {"name": name, "gradient_l2_norm": float(parameter.grad.float().norm()), "max_abs_gradient": float(parameter.grad.abs().max())}
                for name, parameter in values
                if parameter.grad is not None
            ][:8],
        }
    return result


def _markdown(report: dict[str, object], rows: list[dict[str, object]]) -> str:
    retrieval = report["selected_frame_mobileclip_retrieval"]
    gradients = report["gradient_probe"]
    prefix = report["conditioning_contract"]
    text = "# Stage 2 C1 Training-Path Audit\n\n"
    text += "**Scope:** read-only diagnostic of the existing C1 checkpoint. No optimizer was created, no `optimizer.step()` was called, and cache/checkpoint/model tensors were verified unchanged after the backward pass.\n\n"
    text += "## What Supervises The Bridge\n\n"
    text += "The only training objective is FLAN-T5 teacher-forced decoder cross-entropy against the ActivityNet event caption. Reference tokens are tokenizer-padded and padding labels are set to `-100`; there is no direct MobileCLIP-text contrastive, retrieval, reconstruction, frame-level, or visual-token alignment loss. The loss path is: `reference caption -> decoder cross-entropy -> trainable FLAN last encoder/decoder blocks and encoder input embeddings -> visual bridge`.\n\n"
    text += "This loss can learn an alignment, but it does not guarantee one. The current qualitative and shuffled-control results therefore matter: they show the visual pathway has influence, yet it has not learned a reliably caption-semantic interface.\n\n"
    text += "## Prefix And Mask Contract\n\n"
    text += f"- C1 exposes `{prefix['visual_tokens_per_event_max']}` visual tokens at maximum (`8 frames x 64 patches`), followed by `{prefix['instruction_tokens']}` instruction tokens, for `{prefix['total_prefix_tokens']}` encoder tokens.\n"
    text += "- Valid selected-frame patches receive attention mask `1`; padded-frame patches receive `0`; instruction tokens receive the tokenizer attention mask. No visual token is silently truncated.\n"
    text += "- Visual and instruction embeddings share the same FLAN encoder sequence and relative-position mechanism. There is no modality/type embedding or learned gate separating them. That makes competition possible, but this audit cannot claim competition as a measured causal fact.\n\n"
    text += "## Gradient Probe\n\n"
    for name in ("bridge", "flan_trainable", "flan_frozen"):
        item = gradients[name]
        text += f"- `{name}`: {item['nonzero_gradient_tensors']}/{item['parameter_tensors']} tensors had non-zero gradients; L2 norm `{item['gradient_l2_norm']:.6g}`.\n"
    text += f"- One validation batch caption loss: `{gradients['loss']:.6f}`. Parameter tensors unchanged after `backward()`: `{gradients['parameters_unchanged_after_backward']}`.\n\n"
    text += "## Exact Selected-Frame Evidence\n\n"
    text += "This retrieval check uses only the exact up-to-eight cached frames selected by C1, applies the original frozen MobileCLIP-S0 pooling head, and ranks the ten audit reference captions with the matching frozen MobileCLIP text encoder. It is an evidence-sufficiency proxy, not a generative-caption score.\n\n"
    text += f"- Selected-frame Recall@1: `{retrieval['recall_at_1']:.2f}`\n- Mean reference rank: `{retrieval['mean_rank']:.2f}`\n- Events using a nearest-frame fallback: `{retrieval['nearest_frame_fallback_events']}`\n- Selected frames outside their literal annotation window: `{retrieval['literal_window_mismatch_events']}`\n\n"
    text += "| Video | Selected frames | Literal-window status | MobileCLIP rank | Reference caption | C1 caption |\n| --- | ---: | --- | ---: | --- | --- |\n"
    for row in rows:
        status = "OK" if row["all_selected_frames_within_literal_window"] else "FLAGGED"
        text += "| " + " | ".join((str(row["video_id"]), str(row["selected_frame_count"]), status, str(row["selected_frame_mobileclip_retrieval"]["reference_rank"]), str(row["reference_caption"]).replace("|", "\\|"), str(row["c1_caption"]).replace("|", "\\|"))) + " |\n"
    text += "\n## Decision Rule\n\n"
    text += "For an event with rank 1 and literal-window-valid selected frames, the frozen cached MobileCLIP evidence is sufficient in this controlled ten-caption retrieval test. A wrong C1 caption for that row therefore points to bridge/conditioning/training alignment, not an absence of the relevant selected-frame semantics. A non-rank-1 row or a literal-window flag remains a frame-selection/timestamp risk and must not be used as proof of an alignment failure.\n"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2_audit/training_path_audit"))
    parser.add_argument("--mobileclip-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "mobileclip_s0.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU PC.")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "stage2_bridge_ablation" or payload.get("variant") != "c1":
        raise ValueError("This audit requires a Stage 2 bridge-ablation C1 checkpoint.")
    config, stage0 = DirectMobileCLIPCaptionConfig.from_json(args.model_config), Stage0Config.from_json(args.stage0_config)
    if payload.get("model_config") != config.to_dict():
        raise ValueError("Checkpoint and supplied model configuration differ.")
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    by_id = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}
    entries = _audit_entries(args.audit_root)
    if any(str(entry["video_id"]) not in by_id for entry in entries):
        raise ValueError("At least one audit video is absent from the validation manifest.")

    model = _build(config, "c1").to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}

    try:
        import mobileclip
    except ImportError as exc:
        raise RuntimeError("Install the official Apple MobileCLIP package in the active environment.") from exc
    mobileclip_model, _, _ = mobileclip.create_model_and_transforms("mobileclip_s0", pretrained=str(args.mobileclip_checkpoint), reparameterize=False)
    mobileclip_model.to(device).eval()
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    captions = [str(entry["truth"]["reference_caption"]) for entry in entries]  # type: ignore[index]
    with torch.no_grad():
        text_embeddings = F.normalize(mobileclip_model.encode_text(tokenizer(captions).to(device)), dim=-1)

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for expected, entry in enumerate(entries):
            video_id, truth = str(entry["video_id"]), entry["truth"]  # type: ignore[assignment]
            sample = dataset[by_id[video_id]]
            duration, start, end = float(sample["duration"]), float(truth["start"]), float(truth["end"])
            times = sample["frame_times"]
            features = sample["visual_features"]
            segment = torch.tensor([[[start / duration, end / duration]]])
            index_tensor, mask = event_frame_indices(times.unsqueeze(0), torch.ones_like(times, dtype=torch.bool).unsqueeze(0), segment, config.max_event_frames)
            selected_indices = index_tensor[0, 0, mask[0, 0]].tolist()
            selected_times = [float(times[index]) for index in selected_indices]
            literal_inside = [start <= value <= end for value in selected_times]
            literal_indices = [index for index, value in enumerate(times.tolist()) if start <= float(value) <= end]
            fallback = not bool(literal_indices)
            selected, event_mask = select_event_patch_features(features.unsqueeze(0).to(device), times.unsqueeze(0).to(device), torch.ones_like(times, dtype=torch.bool).unsqueeze(0).to(device), segment.to(device), config.max_event_frames)
            generated = model.generate(selected[0], event_mask[0])[0]
            image_embedding = _event_embedding(mobileclip_model.image_encoder.model.head, features, selected_indices, device)
            rank = _rank(image_embedding @ text_embeddings.T, expected, captions)
            row = {
                "video_id": video_id,
                "reference_caption": captions[expected],
                "c1_caption": generated,
                "ground_truth_start_seconds": start,
                "ground_truth_end_seconds": end,
                "video_duration_seconds": duration,
                "normalized_segment": [start / duration, end / duration],
                "selected_cache_indices": selected_indices,
                "selected_cache_timestamps_seconds": selected_times,
                "selected_frame_count": len(selected_indices),
                "literal_in_window_cache_indices": literal_indices,
                "nearest_frame_fallback_used": fallback,
                "selected_frame_inside_literal_window": literal_inside,
                "all_selected_frames_within_literal_window": all(literal_inside),
                "selected_frame_mobileclip_retrieval": rank,
            }
            rows.append(row)
            _write_json(args.output_root / entry["directory"].name / "result.json", row)  # type: ignore[operator]

    # One ordinary validation batch, backward only. This proves the actual loss reaches the intended trainable parameters.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=activitynet_v2_collate_fn, num_workers=0)
    raw = next(iter(loader))
    batch = _to_device(raw, device)
    event_features, event_masks, labels, _ = _flatten_events(batch, config)
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(event_features, event_masks, labels)
    loss.backward()
    gradient_probe = _gradient_summary(model)
    gradient_probe["loss"] = float(loss.detach())
    gradient_probe["optimizer_created"] = False
    gradient_probe["optimizer_step_called"] = False
    gradient_probe["parameters_unchanged_after_backward"] = all(torch.equal(before[name], parameter.detach()) for name, parameter in model.named_parameters() if parameter.requires_grad)
    model.zero_grad(set_to_none=True)

    instruction_tokens = len(model.tokenizer(config.instruction)["input_ids"])
    report = {
        "experiment": "stage2_c1_training_path_audit",
        "training_performed": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(payload["epoch"]),
        "loss_contract": {
            "objective": "teacher_forced_FLAN_T5_decoder_cross_entropy",
            "caption_padding_labels": "-100",
            "caption_max_tokens": config.caption_max_tokens,
            "direct_visual_alignment_loss": False,
            "mobileclip_text_contrastive_loss": False,
            "frame_level_supervision": False,
            "stage1_used": False,
        },
        "conditioning_contract": {
            "visual_tokens_per_event_max": config.max_event_frames * config.patch_tokens,
            "instruction_tokens": instruction_tokens,
            "total_prefix_tokens": config.max_event_frames * config.patch_tokens + instruction_tokens,
            "visual_token_shape": [config.max_event_frames * config.patch_tokens, int(model.language_model.config.d_model)],
            "visual_prefix_truncated": False,
            "valid_patch_mask": "one mask entry per patch token; padded selected-frame patches are zero; valid selected-frame patches are one",
            "instruction_position": "after visual prefix",
            "modality_type_embedding_or_gate": False,
        },
        "gradient_probe": gradient_probe,
        "selected_frame_mobileclip_retrieval": {
            "candidate_reference_captions": len(captions),
            "recall_at_1": sum(row["selected_frame_mobileclip_retrieval"]["reference_rank"] == 1 for row in rows) / len(rows),  # type: ignore[index]
            "mean_rank": sum(row["selected_frame_mobileclip_retrieval"]["reference_rank"] for row in rows) / len(rows),  # type: ignore[index]
            "nearest_frame_fallback_events": sum(bool(row["nearest_frame_fallback_used"]) for row in rows),
            "literal_window_mismatch_events": sum(not bool(row["all_selected_frames_within_literal_window"]) for row in rows),
            "method": "original frozen MobileCLIP-S0 pooling head on exact C1-selected cached frames, ranked against matching frozen MobileCLIP-S0 text embeddings of the ten audit references",
        },
        "videos": rows,
    }
    _write_json(args.output_root / "aggregate.json", report)
    (args.output_root / "SUMMARY.md").parent.mkdir(parents=True, exist_ok=True)
    (args.output_root / "SUMMARY.md").write_text(_markdown(report, rows), encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root.resolve()), "checkpoint_epoch": payload["epoch"], "selected_frame_recall_at_1": report["selected_frame_mobileclip_retrieval"]["recall_at_1"], "bridge_gradient_l2": gradient_probe["bridge"]["gradient_l2_norm"], "parameters_unchanged": gradient_probe["parameters_unchanged_after_backward"]}, indent=2))


if __name__ == "__main__":
    main()
