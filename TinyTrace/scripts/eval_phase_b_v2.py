"""Evaluate and export final TinyTrace Phase B v2 dense-caption predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2 import ActivityNetV2Dataset, PhaseBV2Config, PhaseBV2Model, Stage0Config, activitynet_v2_collate_fn
from tinytrace.phase_b_v2.metrics import localization_metrics, matched_caption_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--overlap-threshold", type=float, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; run final V2 evaluation on the GPU PC.")
    base_config = PhaseBV2Config.from_json(args.model_config)
    config = PhaseBV2Config(**{**base_config.to_dict(), "stage": "joint"})
    stage0 = Stage0Config.from_json(args.stage0_config)
    manifest_sha256 = _sha256(args.manifest)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Checkpoint does not match the supplied immutable V2 manifest.")
    model = PhaseBV2Model.for_language_stage(config)
    model.load_state_dict(payload["model_state"], strict=False)
    model.to(device).eval()
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=activitynet_v2_collate_fn, num_workers=0)
    exports: list[dict[str, object]] = []
    predicted: list[list[dict[str, object]]] = []
    targets: list[torch.Tensor] = []
    durations: list[float] = []
    captions: list[list[str]] = []
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
            events_per_video = model.predict_events(batch, threshold=args.threshold, overlap_threshold=args.overlap_threshold)
            predicted.extend(events_per_video)
            for sample_index in range(batch["segments"].size(0)):
                targets.append(batch["segments"][sample_index, batch["event_mask"][sample_index]].cpu())
                durations.append(float(batch["duration"][sample_index]))
                captions.append(batch["captions"][sample_index])
            for video_id, duration, events in zip(batch["video_id"], batch["duration"].tolist(), events_per_video):
                ordered = []
                for event in events:
                    start, end = float(event["start"]), float(event["end"])
                    if not 0 <= start < end <= 1:
                        raise RuntimeError(f"Invalid normalized predicted segment for {video_id}: {event}")
                    ordered.append({"start": start * duration, "end": end * duration, "normalized_start": start, "normalized_end": end, "confidence": float(event["score"]), "caption": str(event["caption"])})
                exports.append({"video_id": video_id, "events": ordered, "model_version": "tinytrace-phase-b-v2"})
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": _sha256(args.checkpoint), "manifest_sha256": manifest_sha256, "threshold": args.threshold, "overlap_threshold": args.overlap_threshold, "localization": localization_metrics(predicted, targets, durations), "captions": matched_caption_metrics(predicted, targets, captions), "prediction_count": len(exports)}
    _atomic_json(args.output, exports)
    _atomic_json(args.output.with_name(args.output.stem + "_report.json"), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
