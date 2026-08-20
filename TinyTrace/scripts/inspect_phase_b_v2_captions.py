"""Export side-by-side Stage 2 generated captions and ActivityNet references."""

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
    parser.add_argument("--limit", type=int, default=32, help="Maximum event-caption pairs to export.")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 1:
        raise ValueError("batch-size and limit must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU PC.")

    manifest_sha256 = _sha256(args.manifest)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1 or payload.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Checkpoint does not match the supplied immutable V2 manifest.")
    base_config = PhaseBV2Config.from_json(args.model_config)
    config = PhaseBV2Config(**{**base_config.to_dict(), "stage": "caption"})
    model = PhaseBV2Model.for_language_stage(config)
    model.load_state_dict(payload["model_state"], strict=False)
    model.to(device).eval()

    stage0 = Stage0Config.from_json(args.stage0_config)
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=activitynet_v2_collate_fn, num_workers=0)
    comparisons: list[dict[str, object]] = []
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in raw.items()}
            generated, batch_indices, event_indices = model.generate_ground_truth_events(batch)
            for caption, batch_index, event_index in zip(generated, batch_indices.tolist(), event_indices.tolist()):
                duration = float(batch["duration"][batch_index])
                normalized = batch["segments"][batch_index, event_index].tolist()
                comparisons.append(
                    {
                        "video_id": str(batch["video_id"][batch_index]),
                        "start": float(normalized[0]) * duration,
                        "end": float(normalized[1]) * duration,
                        "reference_caption": str(batch["captions"][batch_index][event_index]),
                        "generated_caption": str(caption),
                    }
                )
                if len(comparisons) >= args.limit:
                    break
            if len(comparisons) >= args.limit:
                break
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": payload.get("epoch"),
        "split": "val",
        "segment_source": "ground_truth",
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }
    _atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "comparisons"}, sort_keys=True))


if __name__ == "__main__":
    main()
