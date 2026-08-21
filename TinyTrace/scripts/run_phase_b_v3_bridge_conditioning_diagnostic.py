"""Read-only visual-bridge diagnostic for the Epoch-20 Stage 2 v3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config
from tinytrace.phase_b_v2.metrics import caption_metrics
from tinytrace.phase_b_v3 import DirectMobileCLIPCaptionConfig, DirectMobileCLIPCaptionModel, event_frame_indices, select_event_patch_features


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shuffle(features: torch.Tensor, frame_mask: torch.Tensor, video_id: str) -> tuple[torch.Tensor, list[int], int]:
    count = int(frame_mask.sum())
    seed = int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    permutation = torch.randperm(count, generator=torch.Generator(device="cpu").manual_seed(seed)).tolist() if count > 1 else list(range(count))
    output = features.clone()
    if count > 1:
        output[:count] = features[:count][torch.tensor(permutation, device=features.device)]
    return output, permutation, seed


def _overlap(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.lower().split()), set(right.lower().split())
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens or right_tokens else 1.0


def _stats(tokens: torch.Tensor) -> dict[str, object]:
    norms = torch.linalg.vector_norm(tokens, dim=-1)
    return {"shape": list(tokens.shape), "mean": float(tokens.mean()), "std": float(tokens.std(unbiased=False)), "mean_token_norm": float(norms.mean()), "min_token_norm": float(norms.min()), "max_token_norm": float(norms.max())}


def _pairwise_cosine(rows: dict[str, torch.Tensor]) -> dict[str, float]:
    ids = list(rows)
    values: dict[str, float] = {}
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            values[f"{left}__{right}"] = float(F.cosine_similarity(rows[left].flatten(), rows[right].flatten(), dim=0))
    return values


def _retrieval_sanity(dataset: ActivityNetV2Dataset, audit_items: list[dict[str, object]], checkpoint: Path, device: torch.device) -> dict[str, object]:
    """Use the original frozen MobileCLIP image/text space; no v3 tensors enter it."""
    import mobileclip

    model, _, _ = mobileclip.create_model_and_transforms("mobileclip_s0", pretrained=str(checkpoint), reparameterize=False)
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    model.to(device).eval()
    labels = [str(item["reference_caption"]) for item in audit_items]
    with torch.no_grad():
        text = F.normalize(model.encode_text(tokenizer(labels).to(device)), dim=-1)
        ranks = []
        for expected, item in enumerate(audit_items):
            sample = dataset[item["index"]]
            start, end = float(item["start"]), float(item["end"])
            indices = [index for index, value in enumerate(sample["frame_times"].tolist()) if start <= value <= end]
            if not indices:
                midpoint = (start + end) / 2
                indices = [min(range(len(sample["frame_times"])), key=lambda index: abs(float(sample["frame_times"][index]) - midpoint))]
            patches = sample["visual_features"][indices].float().to(device)
            side = math.isqrt(patches.size(1))
            image = model.image_encoder.model.head(patches.transpose(1, 2).reshape(patches.size(0), patches.size(2), side, side))
            scores = F.normalize(image.mean(dim=0), dim=0) @ text.T
            ranks.append(scores.argsort(descending=True).tolist().index(expected) + 1)
    return {"candidate_captions": len(labels), "reference_ranks": ranks, "mean_rank": sum(ranks) / len(ranks), "recall_at_1": sum(rank == 1 for rank in ranks) / len(ranks)}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mobileclip-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "mobileclip_s0.pt")
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2_audit/bridge_conditioning_diagnostic"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {args.output_root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU machine.")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = DirectMobileCLIPCaptionConfig(**payload["model_config"])
    if config.use_stage1_temporal_context:
        raise ValueError("The supplied checkpoint is not the direct-only v3 configuration.")
    captioner = DirectMobileCLIPCaptionModel.from_pretrained(config)
    captioner.load_state_dict(payload["model_state"], strict=True)
    captioner.to(device).eval()
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=Stage0Config.from_json(args.stage0_config), split="val")
    by_id = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}
    rows: list[dict[str, object]] = []
    token_rows: dict[str, torch.Tensor] = {}
    audit_items: list[dict[str, object]] = []
    for directory in sorted(args.audit_root.glob("video_*")):
        focus = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))["audit_focus_event"]
        video_id = directory.name.split("_", 2)[-1]
        sample = dataset[by_id[video_id]]
        duration = float(sample["duration"])
        segment = torch.tensor([float(focus["start"]) / duration, float(focus["end"]) / duration], device=device).view(1, 1, 2)
        features = sample["visual_features"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        times = sample["frame_times"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        frame_mask = torch.ones(times.shape, dtype=torch.bool, device=device)
        cache_indices, event_mask = event_frame_indices(times, frame_mask, segment, config.max_event_frames)
        selected, selected_mask = select_event_patch_features(features, times, frame_mask, segment, config.max_event_frames)
        if not torch.equal(event_mask, selected_mask):
            raise RuntimeError("Event selection mismatch.")
        event_features, event_mask = selected[0], event_mask[0]
        shuffled, permutation, seed = _shuffle(event_features[0], event_mask[0], video_id)
        zero = torch.zeros_like(event_features)
        real_caption = captioner.generate(event_features, event_mask)[0]
        zero_caption = captioner.generate(zero, event_mask)[0]
        shuffled_caption = captioner.generate(shuffled.unsqueeze(0), event_mask)[0]
        real_tokens, real_attention = captioner.conditioning(event_features, event_mask)
        zero_tokens, _ = captioner.conditioning(zero, event_mask)
        visual_tokens = real_tokens[:, : config.visual_tokens]
        token_rows[video_id] = visual_tokens[0].detach().cpu()
        count = int(event_mask[0].sum())
        item = {"video_id": video_id, "reference_caption": str(focus["reference_caption"]), "real_caption": real_caption, "zero_caption": zero_caption, "shuffled_caption": shuffled_caption, "real_vs_zero_token_overlap": _overlap(real_caption, zero_caption), "real_vs_shuffled_token_overlap": _overlap(real_caption, shuffled_caption), "zero_vs_shuffled_token_overlap": _overlap(zero_caption, shuffled_caption), "selected_cache_indices": cache_indices[0, 0, :count].tolist(), "selected_frame_timestamps": [float(times[0, index]) for index in cache_indices[0, 0, :count].tolist()], "event_feature_shape": list(event_features.shape), "event_frame_permutation": permutation, "permutation_seed": seed, "visual_token_stats": _stats(visual_tokens), "zero_visual_token_stats": _stats(zero_tokens[:, : config.visual_tokens]), "flan_conditioning_shape": list(real_tokens.shape), "flan_attention_shape": list(real_attention.shape)}
        _json(args.output_root / directory.name / "result.json", item)
        rows.append(item)
        audit_items.append({"index": by_id[video_id], "reference_caption": item["reference_caption"], "start": float(focus["start"]), "end": float(focus["end"])})
    references = [str(row["reference_caption"]) for row in rows]
    captions = {"real": [str(row["real_caption"]) for row in rows], "zero": [str(row["zero_caption"]) for row in rows], "shuffled": [str(row["shuffled_caption"]) for row in rows]}
    pairwise = {"real_zero": [float(row["real_vs_zero_token_overlap"]) for row in rows], "real_shuffled": [float(row["real_vs_shuffled_token_overlap"]) for row in rows], "zero_shuffled": [float(row["zero_vs_shuffled_token_overlap"]) for row in rows]}
    aggregate_pairs = {name: {"mean_token_overlap": sum(values) / len(values), "identical_caption_rate": sum(values[index] == 1.0 and captions[name.split('_')[0]][index] == captions[name.split('_')[1]][index] for index in range(len(values))) / len(values), "caption_change_rate": sum(captions[name.split('_')[0]][index] != captions[name.split('_')[1]][index] for index in range(len(values))) / len(values)} for name, values in pairwise.items()}
    cosine = _pairwise_cosine(token_rows)
    retrieval = _retrieval_sanity(dataset, audit_items, args.mobileclip_checkpoint, device)
    real_zero_change = aggregate_pairs["real_zero"]["caption_change_rate"]
    real_metrics, zero_metrics = caption_metrics(captions["real"], references), caption_metrics(captions["zero"], references)
    if real_zero_change < 0.2:
        verdict = "A. VISUAL CONDITIONING IS BEING IGNORED"
    elif real_metrics["meteor_unigram"] > zero_metrics["meteor_unigram"] and real_metrics["cider_unigram"] > zero_metrics["cider_unigram"] and retrieval["recall_at_1"] == 1.0:
        verdict = "B. VISUAL INFORMATION REACHES FLAN-T5 BUT IS SEMANTICALLY MISALIGNED"
    else:
        verdict = "D. INSUFFICIENT EVIDENCE — ANOTHER CONTROL IS REQUIRED"
    aggregate = {"experiment": "stage2_v3_bridge_conditioning_diagnostic", "training_performed": False, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": payload.get("epoch"), "stage1_used": False, "caption_metrics": {name: caption_metrics(values, references) for name, values in captions.items()}, "pairwise_caption_controls": aggregate_pairs, "visual_token_shape": [config.visual_tokens, int(captioner.language_model.config.d_model)], "visual_token_pairwise_cosine": {"mean": sum(cosine.values()) / len(cosine), "min": min(cosine.values()), "max": max(cosine.values()), "pairs": cosine}, "retrieval_sanity": retrieval, "transformation_pipeline": [{"operation": "cached MobileCLIP event features", "shape": "[selected_frames<=8, 64, 1024]", "pooling_or_loss": "No pooling; frozen cache."}, {"operation": "LayerNorm + Linear projection", "shape": "[selected_frames, 64, 512]", "pooling_or_loss": "Dimensionality 1024->512; no token pooling."}, {"operation": "spatial/temporal learned positions", "shape": "[selected_frames, 64, 512]", "pooling_or_loss": "Addition only."}, {"operation": "flattened cross-attention resampler", "shape": "[selected_frames*64, 512] -> [16, 512]", "pooling_or_loss": "Learned aggregation from up to 512 patch tokens into 16 soft tokens."}, {"operation": "FLAN input", "shape": "[16 visual + instruction tokens, 512]", "pooling_or_loss": "Visual tokens leave MobileCLIP embedding space; MobileCLIP text retrieval is not mathematically meaningful here."}], "verdict": verdict}
    _json(args.output_root / "aggregate_metrics.json", aggregate)
    headers = ["Video", "Reference", "Real", "Zero", "Shuffled"]
    table_rows = [[str(row["video_id"]), str(row["reference_caption"]), str(row["real_caption"]), str(row["zero_caption"]), str(row["shuffled_caption"])] for row in rows]
    summary = "# Visual Bridge / Conditioning Diagnostic\n\n**" + verdict + "**\n\nInference-only; MobileCLIP, FLAN-T5, Stage 1, and checkpoint weights were read-only.\n\n## Captions\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in table_rows) + "\n\n## Aggregate\n```json\n" + json.dumps(aggregate, indent=2) + "\n```\n\nPer-video reports contain exact selected cache frames, permutations, and visual-token statistics.\n"
    (args.output_root / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root.resolve()), "verdict": verdict, "retrieval_recall_at_1": retrieval["recall_at_1"]}, sort_keys=True))


if __name__ == "__main__":
    main()
