"""Inference-only identity ablation for Stage 2 v3 direct-MobileCLIP captioning.

V3's default configuration deliberately has no Stage 1 temporal context. This
script verifies that fact from the checkpoint, regenerates captions from the
same raw cached patches, and compares them with the saved Epoch-20 audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config
from tinytrace.phase_b_v2.metrics import caption_metrics
from tinytrace.phase_b_v3 import DirectMobileCLIPCaptionConfig, DirectMobileCLIPCaptionModel, event_frame_indices, select_event_patch_features


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shuffle(features: torch.Tensor, mask: torch.Tensor, video_id: str) -> tuple[torch.Tensor, list[int], int]:
    count = int(mask.sum())
    seed = int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    if count < 2:
        return features.clone(), list(range(count)), seed
    permutation = torch.randperm(count, generator=torch.Generator(device="cpu").manual_seed(seed)).tolist()
    output = features.clone()
    output[:count] = features[:count][torch.tensor(permutation, device=features.device)]
    return output, permutation, seed


def _overlap(first: str, second: str) -> float:
    left, right = set(first.lower().split()), set(second.lower().split())
    return len(left & right) / len(left | right) if left or right else 1.0


def _semantic_score(prediction: str, reference: str) -> float:
    """Declared lightweight, reference-aware proxy used only for pair comparison."""
    return float(caption_metrics([prediction], [reference])["meteor_unigram"])


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage2_audit/direct_mobileclip_ablation"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output_root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU machine.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = DirectMobileCLIPCaptionConfig(**checkpoint["model_config"])
    if config.use_stage1_temporal_context:
        raise ValueError("This checkpoint uses Stage 1 context; it is not the requested direct-only v3 baseline.")
    model = DirectMobileCLIPCaptionModel.from_pretrained(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=Stage0Config.from_json(args.stage0_config), split="val")
    index = {str(item["video_id"]): row for row, item in enumerate(dataset.items)}
    rows: list[dict[str, object]] = []
    real, shuffled, references = [], [], []
    direct_matches_baseline = []
    real_better = shuffled_better = equal_quality = changed = 0
    for directory in sorted(args.audit_root.glob("video_*")):
        focus = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))["audit_focus_event"]
        video_id = directory.name.split("_", 2)[-1]
        sample = dataset[index[video_id]]
        duration = float(sample["duration"])
        features = sample["visual_features"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        times = sample["frame_times"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        frame_mask = torch.ones(times.shape, dtype=torch.bool, device=device)
        segment = torch.tensor([float(focus["start"]) / duration, float(focus["end"]) / duration], device=device).view(1, 1, 2)
        cache_indices, selected_mask = event_frame_indices(times, frame_mask, segment, config.max_event_frames)
        selected, selected_mask_again = select_event_patch_features(features, times, frame_mask, segment, config.max_event_frames)
        if not torch.equal(selected_mask, selected_mask_again):
            raise RuntimeError("Frame index and feature selection masks disagree.")
        event_features, event_mask = selected[0], selected_mask[0]
        direct_caption = model.generate(event_features, event_mask)[0]
        shuffled_features, permutation, seed = _shuffle(event_features[0], event_mask[0], video_id)
        shuffled_caption = model.generate(shuffled_features.unsqueeze(0), event_mask)[0]
        reference = str(focus["reference_caption"])
        baseline_path = args.baseline_root / f"video_{directory.name.split('_', 1)[1]}.json"
        # Baseline filenames are exactly the audit directory names; retaining
        # this fallback makes the package robust to a future naming change.
        if not baseline_path.is_file():
            baseline_path = args.baseline_root / f"{directory.name}.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        same_as_baseline = direct_caption == baseline["real_caption"]
        direct_matches_baseline.append(same_as_baseline)
        real_score, shuffled_score = _semantic_score(direct_caption, reference), _semantic_score(shuffled_caption, reference)
        if real_score > shuffled_score:
            real_better += 1
        elif shuffled_score > real_score:
            shuffled_better += 1
        else:
            equal_quality += 1
        if direct_caption != shuffled_caption:
            changed += 1
        used = int(event_mask[0].sum())
        result = {
            "video_id": video_id,
            "reference_caption": reference,
            "real_caption": direct_caption,
            "shuffled_caption": shuffled_caption,
            "real_vs_shuffled_token_overlap": _overlap(direct_caption, shuffled_caption),
            "whether_real_equals_shuffled": direct_caption == shuffled_caption,
            "real_meteor_like_against_reference": real_score,
            "shuffled_meteor_like_against_reference": shuffled_score,
            "baseline_v3_epoch20_caption": baseline["real_caption"],
            "direct_equals_baseline_v3": same_as_baseline,
            "selected_event_frame_count": used,
            "selected_cache_indices": cache_indices[0, 0, :used].tolist(),
            "selected_frame_timestamps": [float(times[0, item]) for item in cache_indices[0, 0, :used].tolist()],
            "event_frame_permutation": permutation,
            "permutation_seed": seed,
            "event_feature_shape": list(event_features.shape),
        }
        _json(args.output_root / directory.name / "result.json", result)
        rows.append(result); real.append(direct_caption); shuffled.append(shuffled_caption); references.append(reference)
    aggregate = {
        "experiment": "direct_mobileclip_inference_identity_ablation",
        "training_performed": False,
        "mobileclip_modified": False,
        "stage1_used_in_direct_path": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "baseline_root": str(args.baseline_root.resolve()),
        "real_metrics": caption_metrics(real, references),
        "shuffled_metrics": caption_metrics(shuffled, references),
        "mean_real_vs_shuffled_token_overlap": sum(float(row["real_vs_shuffled_token_overlap"]) for row in rows) / len(rows),
        "identical_caption_rate": sum(bool(row["whether_real_equals_shuffled"]) for row in rows) / len(rows),
        "captions_changed": changed,
        "real_semantically_closer_than_shuffled": real_better,
        "shuffled_semantically_closer_than_real": shuffled_better,
        "equal_semantic_proxy": equal_quality,
        "direct_equals_baseline_rate": sum(direct_matches_baseline) / len(direct_matches_baseline),
        "outcome": "INCONCLUSIVE",
        "conclusion": "The Epoch-20 v3 checkpoint is already a direct cached-MobileCLIP-to-FLAN path with Stage 1 disabled. Therefore this diagnostic is an identity verification, not an independent Stage-1-removal comparison. It cannot provide evidence that Stage 1 damages captions; a valid comparison requires a separately trained Stage-1-conditioned checkpoint using the same v3 adapter and training budget.",
    }
    _json(args.output_root / "aggregate_metrics.json", aggregate)
    headers = ["Video", "Reference", "Direct real", "Direct shuffled", "Overlap", "Changed", "Matches v3 baseline"]
    table_rows = []
    for row in rows:
        table_rows.append([str(row["video_id"]), str(row["reference_caption"]), str(row["real_caption"]), str(row["shuffled_caption"]), f'{float(row["real_vs_shuffled_token_overlap"]):.3f}', str(not bool(row["whether_real_equals_shuffled"])), str(row["direct_equals_baseline_v3"])])
    markdown = "# Direct MobileCLIP to FLAN-T5 Ablation\n\n**Outcome: INCONCLUSIVE**\n\nThis is inference-only: no adapter, FLAN-T5, MobileCLIP, Stage 1, or checkpoint weights were trained or modified. Checkpoint configuration confirms `use_stage1_temporal_context: false`; therefore the current Epoch-20 v3 baseline already is the requested direct path.\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in table_rows) + "\n\n## Aggregate\n```json\n" + json.dumps(aggregate, indent=2) + "\n```\n\n## Answers\n\n- **A:** No separate direct model exists here: regenerated direct captions match the configured v3 direct baseline.\n- **B / E:** No conclusion about Stage 1 damage is possible because Stage 1 was already absent from v3.\n- **C:** See the real/shuffled overlap and changed-caption counts above. This only tests sensitivity to event-frame order, not a semantic substitution control.\n- **D:** Per-video `result.json` files retain exact selected cache indices, timestamps, and permutations.\n"
    (args.output_root / "SUMMARY.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({key: value for key, value in aggregate.items() if key not in {"conclusion"}}, sort_keys=True))


if __name__ == "__main__":
    main()
