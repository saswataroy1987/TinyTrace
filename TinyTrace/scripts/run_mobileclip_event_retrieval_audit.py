"""Rank audit captions with frozen MobileCLIP cached event features.

This is a no-training diagnostic. It applies the original MobileCLIP-S0 image
pooling head to already cached spatial features and compares the result with
frozen MobileCLIP-S0 text embeddings.
"""

from __future__ import annotations

import argparse
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


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frame_indices(times: torch.Tensor, start: float, end: float) -> list[int]:
    selected = [index for index, value in enumerate(times.tolist()) if start <= value <= end]
    if selected:
        return selected
    midpoint = (start + end) / 2
    return [min(range(len(times)), key=lambda index: abs(float(times[index]) - midpoint))]


def _event_embedding(head: torch.nn.Module, features: torch.Tensor, indices: list[int], device: torch.device) -> torch.Tensor:
    tokens = features[indices].float().to(device)
    patch_count, width = tokens.shape[1:]
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise ValueError(f"Cached patch count must be square; got {patch_count}")
    pooled = head(tokens.transpose(1, 2).reshape(tokens.size(0), width, side, side))
    return F.normalize(pooled.mean(dim=0), dim=0)


def _rank(scores: torch.Tensor, expected: int, labels: list[str]) -> tuple[int, float, float, list[dict[str, object]]]:
    order = scores.argsort(descending=True).tolist()
    rank = order.index(expected) + 1
    best_other = max(float(scores[index]) for index in range(len(labels)) if index != expected)
    top = [{"rank": position + 1, "caption": labels[index], "similarity": float(scores[index]), "is_reference": index == expected} for position, index in enumerate(order[:5])]
    return rank, float(scores[expected]), float(scores[expected]) - best_other, top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--mobileclip-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "mobileclip_s0.pt")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_root or args.audit_root / "mobileclip_retrieval"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU PC.")
    try:
        import mobileclip
    except ImportError as exc:
        raise RuntimeError("Install the official Apple MobileCLIP package in the active environment.") from exc
    if not args.mobileclip_checkpoint.is_file():
        raise FileNotFoundError(f"MobileCLIP checkpoint missing: {args.mobileclip_checkpoint}")

    stage0 = Stage0Config.from_json(args.stage0_config)
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    index_by_id = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}
    audits = []
    for directory in sorted(args.audit_root.glob("video_*")):
        focus = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))["audit_focus_event"]
        audits.append({"directory": directory, "video_id": directory.name.split("_", 2)[-1], "focus": focus})
    if not audits:
        raise ValueError(f"No audit video directories found in {args.audit_root}")
    for item in audits:
        if item["video_id"] not in index_by_id:
            raise ValueError(f"Audit ID is absent from validation manifest: {item['video_id']}")

    model, _, _ = mobileclip.create_model_and_transforms("mobileclip_s0", pretrained=str(args.mobileclip_checkpoint), reparameterize=False)
    tokenizer = mobileclip.get_tokenizer("mobileclip_s0")
    model.to(device).eval()
    labels = [str(item["focus"]["reference_caption"]) for item in audits]
    prompted = [f"a video of {caption}" for caption in labels]
    with torch.no_grad():
        text_raw = F.normalize(model.encode_text(tokenizer(labels).to(device)), dim=-1)
        text_prompted = F.normalize(model.encode_text(tokenizer(prompted).to(device)), dim=-1)

    output.mkdir(parents=True)
    summary_rows: list[dict[str, str]] = []
    raw_ranks, prompted_ranks = [], []
    with torch.no_grad():
        for expected, item in enumerate(audits):
            sample = dataset[index_by_id[item["video_id"]]]
            focus = item["focus"]
            indices = _frame_indices(sample["frame_times"], float(focus["start"]), float(focus["end"]))
            image = _event_embedding(model.image_encoder.model.head, sample["visual_features"], indices, device)
            raw_scores, prompted_scores = image @ text_raw.T, image @ text_prompted.T
            raw_rank, raw_score, raw_margin, raw_top = _rank(raw_scores, expected, labels)
            prompt_rank, prompt_score, prompt_margin, prompt_top = _rank(prompted_scores, expected, labels)
            result = {
                "video_id": item["video_id"],
                "focus_segment_source": "ground_truth",
                "ground_truth_start": float(focus["start"]),
                "ground_truth_end": float(focus["end"]),
                "reference_caption": labels[expected],
                "cached_frame_indices": indices,
                "cached_frame_timestamps": [float(sample["frame_times"][index]) for index in indices],
                "candidate_count": len(labels),
                "raw_caption_retrieval": {"reference_rank": raw_rank, "reference_similarity": raw_score, "reference_margin_over_best_distractor": raw_margin, "top_5": raw_top},
                "prompted_caption_retrieval": {"prompt_template": "a video of {caption}", "reference_rank": prompt_rank, "reference_similarity": prompt_score, "reference_margin_over_best_distractor": prompt_margin, "top_5": prompt_top},
            }
            _json(output / item["directory"].name / "mobileclip_caption_retrieval.json", result)
            raw_ranks.append(raw_rank)
            prompted_ranks.append(prompt_rank)
            summary_rows.append({"Video": item["directory"].name, "Reference": labels[expected], "Raw Rank": str(raw_rank), "Prompted Rank": str(prompt_rank), "Raw Top Caption": str(raw_top[0]["caption"]), "Prompted Top Caption": str(prompt_top[0]["caption"])})
    headers = list(summary_rows[0])
    table = "# MobileCLIP Event-to-Caption Retrieval\n\nEach image embedding is produced by applying the original frozen MobileCLIP-S0 global-pooling head to cached event patch features. It is ranked against the ten audited ground-truth captions embedded by the matching frozen MobileCLIP-S0 text encoder. Rank 1 is best. This is a small, difficult within-audit retrieval set, not a caption-generation score.\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(row[key].replace("|", "\\|") for key in headers) + " |" for row in summary_rows) + f"\n\n- Raw-caption mean rank: `{sum(raw_ranks) / len(raw_ranks):.2f}`\n- Prompted-caption mean rank: `{sum(prompted_ranks) / len(prompted_ranks):.2f}`\n- Raw Recall@1: `{sum(rank == 1 for rank in raw_ranks) / len(raw_ranks):.2f}`\n- Prompted Recall@1: `{sum(rank == 1 for rank in prompted_ranks) / len(prompted_ranks):.2f}`\n"
    (output / "SUMMARY.md").write_text(table, encoding="utf-8")
    (output / "README.md").write_text("# MobileCLIP Retrieval Diagnostic\n\nThis is inference-only and uses no TinyTrace or FLAN-T5 training. It tests whether the original frozen MobileCLIP image/text space can rank each audit event's own ActivityNet caption above the other nine audit captions.\n\nA strong result suggests the cache retains caption-relevant semantics and the Phase 2 bridge is the main failure. A weak result suggests that either MobileCLIP-S0 is not aligned to these detailed ActivityNet descriptions or the cached pre-pooling features cannot be reliably used for caption semantics without a better representation.\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output.resolve()), "videos": len(audits), "raw_mean_rank": sum(raw_ranks) / len(raw_ranks), "prompted_mean_rank": sum(prompted_ranks) / len(prompted_ranks)}, indent=2))


if __name__ == "__main__":
    main()
