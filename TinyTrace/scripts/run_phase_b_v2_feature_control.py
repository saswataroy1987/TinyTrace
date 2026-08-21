"""Compare real and deterministically shuffled cached features without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2 import ActivityNetV2Dataset, PhaseBV2Config, PhaseBV2Model, Stage0Config
from tinytrace.phase_b_v2.caption import pool_event_features


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed(video_id: str) -> int:
    return int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def _caption(model: PhaseBV2Model, sample: dict[str, object], segment_seconds: list[float], device: torch.device, permutation: torch.Tensor | None) -> str:
    if model.captioner is None:
        raise RuntimeError("Captioner is unavailable.")
    duration = float(sample["duration"])
    features = sample["visual_features"].unsqueeze(0).to(device)  # type: ignore[union-attr]
    if permutation is not None:
        features = features[:, permutation.to(device)]
    times = sample["frame_times"].unsqueeze(0).to(device)  # type: ignore[union-attr]
    frame_mask = torch.ones(times.shape, dtype=torch.bool, device=device)
    segment = torch.tensor(segment_seconds, dtype=torch.float32, device=device).view(1, 1, 2) / duration
    temporal = model.detector.encode(features, times, frame_mask)
    pooled, valid = pool_event_features(temporal, times, frame_mask, segment, model.config.conditioning_tokens)
    if not bool(valid.item()):
        raise RuntimeError("Focus event is invalid after normalization.")
    return model.captioner.generate(pooled[0])[0]


def _overlap(a: str, b: str) -> float:
    left, right = set(a.lower().split()), set(b.lower().split())
    return len(left & right) / len(left | right) if left or right else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_root or args.audit_root / "feature_control"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty control directory: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU PC.")

    base = PhaseBV2Config.from_json(args.model_config)
    config = PhaseBV2Config(**{**base.to_dict(), "stage": "caption"})
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PhaseBV2Model.for_language_stage(config)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        raise ValueError(f"Checkpoint/model mismatch; missing={missing}, unexpected={unexpected}")
    model.to(device).eval()
    stage0 = Stage0Config.from_json(args.stage0_config)
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    by_id = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}

    output.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for directory in sorted(args.audit_root.glob("video_*")):
        truth = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))
        video_id = str(truth["video_id"])
        if video_id not in by_id:
            raise ValueError(f"Audit video is not in the validation manifest: {video_id}")
        focus = truth["audit_focus_event"]
        sample = dataset[by_id[video_id]]
        generator = torch.Generator(device="cpu").manual_seed(_seed(video_id))
        permutation = torch.randperm(sample["visual_features"].shape[0], generator=generator)  # type: ignore[union-attr]
        with torch.no_grad():
            real = _caption(model, sample, [float(focus["start"]), float(focus["end"])], device, None)
            shuffled = _caption(model, sample, [float(focus["start"]), float(focus["end"])], device, permutation)
        result = {
            "video_id": video_id,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": payload.get("epoch"),
            "focus_segment_source": "ground_truth",
            "ground_truth_start": float(focus["start"]),
            "ground_truth_end": float(focus["end"]),
            "reference_caption": str(focus["reference_caption"]),
            "real_feature_caption": real,
            "shuffled_feature_caption": shuffled,
            "identical_caption": real.strip() == shuffled.strip(),
            "token_set_jaccard": _overlap(real, shuffled),
            "shuffle_seed": _seed(video_id),
            "frame_count": int(permutation.numel()),
            "permutation": permutation.tolist(),
        }
        control_dir = output / directory.name
        _json(control_dir / "real_vs_shuffled.json", result)
        with (control_dir / "frame_permutation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["target_cache_index", "source_cache_index"])
            writer.writeheader()
            writer.writerows({"target_cache_index": target, "source_cache_index": source} for target, source in enumerate(permutation.tolist()))
        rows.append({"Video": directory.name, "Reference": str(focus["reference_caption"]), "Real features": real, "Shuffled features": shuffled, "Identical": str(result["identical_caption"]), "Token overlap": f"{result['token_set_jaccard']:.3f}"})
    headers = list(rows[0]) if rows else []
    table = "# Real vs Shuffled Feature Control\n\nThe same ground-truth event segment and checkpoint were used for each pair. Only the temporal order/content of the cached frame feature sequence was permuted; timestamps and event bounds stayed fixed. If captions remain the same or nearly the same, the caption path is insensitive to visual evidence.\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(row[key].replace("|", "\\|") for key in headers) + " |" for row in rows) + "\n"
    (output / "SUMMARY.md").write_text(table, encoding="utf-8")
    (output / "README.md").write_text("# Feature Control\n\nThis is inference-only. It uses the existing Stage 2 checkpoint and never writes cache tensors or model weights. For every audit event it compares a real cached frame sequence to a deterministic permutation of that sequence, using the same ground-truth event interval.\n\n- `SUMMARY.md`: side-by-side captions.\n- `video_*/real_vs_shuffled.json`: reproducible evidence, including the seed and full permutation.\n- `video_*/frame_permutation.csv`: cache-index mapping for the shuffled input.\n\nInterpretation: identical or highly overlapping captions indicate that the current caption conditioning is weakly sensitive to visual input. Different but equally incorrect captions show sensitivity without reliable visual grounding.\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output.resolve()), "videos": len(rows), "identical_captions": sum(row["Real features"] == row["Shuffled features"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
