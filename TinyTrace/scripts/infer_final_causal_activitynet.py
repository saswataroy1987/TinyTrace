"""Autoregressively generate one chronological event sequence for a cached video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_final_causal import FinalCausalConfig, FinalCausalEventModel, parse_event_sequence
from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True); parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable.")
    config, stage0 = FinalCausalConfig.from_json(args.model_config), Stage0Config.from_json(args.stage0_config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "final_lightweight_trace_inspired_causal" or payload.get("model_config") != config.to_dict(): raise ValueError("Checkpoint does not match the supplied final causal configuration.")
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0)
    lookup = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}
    if args.video_id not in lookup: raise ValueError(f"Video is absent from manifest: {args.video_id}")
    sample = dataset[lookup[args.video_id]]
    model = FinalCausalEventModel.from_pretrained(config).to(device); model.load_state_dict(payload["model_state"], strict=True); model.eval()
    with torch.no_grad():
        sequence = model.generate(sample["visual_features"].unsqueeze(0).to(device), sample["frame_times"].unsqueeze(0).to(device), torch.ones((1, sample["visual_features"].size(0)), dtype=torch.bool, device=device), torch.tensor([float(sample["duration"])], device=device))[0]
    events = [{"start_seconds": round(float(event["start_normalized"]) * float(sample["duration"]), 3), "end_seconds": round(float(event["end_normalized"]) * float(sample["duration"]), 3), "caption": event["caption"]} for event in parse_event_sequence(sequence)]
    result = {"video_id": args.video_id, "duration_seconds": float(sample["duration"]), "raw_autoregressive_sequence": sequence, "events": events, "chronological": all(events[index]["start_seconds"] <= events[index + 1]["start_seconds"] for index in range(len(events) - 1))}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "events": len(events), "chronological": result["chronological"]}))


if __name__ == "__main__": main()
