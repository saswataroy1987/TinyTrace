"""Forward-pass and event-selection sanity check for Stage 2 v3, without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config
from tinytrace.phase_b_v3 import DirectMobileCLIPCaptionConfig, DirectMobileCLIPCaptionModel, event_frame_indices, select_event_patch_features


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU PC.")
    config = DirectMobileCLIPCaptionConfig.from_json(args.model_config)
    stage0 = Stage0Config.from_json(args.stage0_config)
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    by_id = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}
    model = DirectMobileCLIPCaptionModel.from_pretrained(config).to(device).eval()
    reports = []
    for directory in sorted(args.audit_root.glob("video_*")):
        focus = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))["audit_focus_event"]
        video_id = directory.name.split("_", 2)[-1]
        if video_id not in by_id:
            raise ValueError(f"Audit video absent from validation manifest: {video_id}")
        sample = dataset[by_id[video_id]]
        duration = float(sample["duration"])
        features = sample["visual_features"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        times = sample["frame_times"].unsqueeze(0).to(device)  # type: ignore[union-attr]
        mask = torch.ones(times.shape, dtype=torch.bool, device=device)
        segment = torch.tensor([float(focus["start"]) / duration, float(focus["end"]) / duration], device=device).view(1, 1, 2)
        indices, selected_mask = event_frame_indices(times, mask, segment, config.max_event_frames)
        selected, selected_mask_again = select_event_patch_features(features, times, mask, segment, config.max_event_frames)
        if not torch.equal(selected_mask, selected_mask_again):
            raise RuntimeError("Frame selector mask disagrees with selected features.")
        selected = selected[0]
        selected_mask = selected_mask[0]
        conditioning, conditioning_mask = model.conditioning(selected, selected_mask)
        if conditioning.shape[1] != config.visual_tokens + int(conditioning_mask.shape[1] - config.visual_tokens):
            raise RuntimeError("Invalid FLAN conditioning sequence length.")
        forward_loss = model(selected, selected_mask, [str(focus["reference_caption"])])
        generated = model.generate(selected, selected_mask)[0]
        count = int(selected_mask[0].sum())
        reports.append({
            "video_id": video_id,
            "ground_truth_seconds": [float(focus["start"]), float(focus["end"])],
            "cached_input_shape": list(features.shape),
            "selected_event_feature_shape": list(selected.shape),
            "selected_event_frame_count": count,
            "selected_cache_indices": indices[0, 0, :count].tolist(),
            "selected_frame_timestamps": [float(times[0, index]) for index in indices[0, 0, :count].tolist()],
            "visual_token_shape": list(model.adapter(selected, selected_mask).shape),
            "flan_conditioning_shape": list(conditioning.shape),
            "flan_conditioning_mask_shape": list(conditioning_mask.shape),
            "untrained_teacher_forced_loss": float(forward_loss),
            "untrained_generation": generated,
        })
    result = {"experiment": "stage2_v3_direct_mobileclip", "training_started": False, "cache_written": False, "mobileclip_trainable": False, "fixed_instruction": config.instruction, "videos": len(reports), "reports": reports}
    _json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, sort_keys=True))


if __name__ == "__main__":
    main()
