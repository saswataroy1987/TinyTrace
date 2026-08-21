"""Inference-only frame-density ablation for the fixed Stage 2 v3 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.config import TinyTraceConfig
from tinytrace.data import sample_uniform_frame_times
from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config
from tinytrace.phase_b_v2.metrics import caption_metrics
from tinytrace.phase_b_v3 import DirectMobileCLIPCaptionConfig, DirectMobileCLIPCaptionModel, event_frame_indices, select_event_patch_features
from tinytrace.vision import MobileCLIPSpatialEncoder


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_uniform_frames(video: Path, duration: float, frame_count: int, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the same fps/scale/crop decode contract as TinyTrace's cache loader."""
    times = sample_uniform_frame_times(duration, frame_count)
    safe_duration = max(duration - 0.25, 1e-6)
    fps = (frame_count - 1) / safe_duration if frame_count > 1 else 1.0 / safe_duration
    command = ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"fps={fps:.12f}:round=near,scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}", "-frames:v", str(frame_count), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    expected = frame_count * size * size * 3
    if result.returncode == 0 and len(result.stdout) == expected:
        frames = torch.frombuffer(bytearray(result.stdout), dtype=torch.uint8).view(frame_count, size, size, 3).permute(0, 3, 1, 2).clone()
        return frames, times
    # This is the same defensive path used by JsonTinyTraceDataset for codecs
    # that yield fewer frames through an fps filter near their endpoint.
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        source_safe_end = max(float(probe.stdout.decode("utf-8").strip()) - 0.5, 0.0)
    except ValueError:
        source_safe_end = max(duration - 0.5, 0.0)
    individual: list[torch.Tensor] = []
    for timestamp in times.tolist():
        # A copied source can be a few milliseconds shorter than its manifest.
        # This cap only affects the final non-event frame and prevents a failed
        # seek from turning a sampling-density test into a decoder test.
        decoded_timestamp = min(timestamp, source_safe_end)
        seek = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), "-ss", f"{decoded_timestamp:.9f}", "-frames:v", "1", "-vf", f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if seek.returncode != 0 or len(seek.stdout) != size * size * 3:
            detail = seek.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Failed to seek {video} at {timestamp:.3f}s: {detail or f'{len(seek.stdout)} bytes'}")
        individual.append(torch.frombuffer(bytearray(seek.stdout), dtype=torch.uint8).view(size, size, 3).permute(2, 0, 1).clone())
    frames = torch.stack(individual)
    return frames, times


def _shuffle(features: torch.Tensor, frame_mask: torch.Tensor, video_id: str) -> tuple[torch.Tensor, list[int], int]:
    count = int(frame_mask.sum())
    seed = int.from_bytes(hashlib.sha256(video_id.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)
    permutation = torch.randperm(count, generator=torch.Generator(device="cpu").manual_seed(seed)).tolist() if count > 1 else list(range(count))
    result = features.clone()
    if count > 1:
        result[:count] = features[:count][torch.tensor(permutation, device=features.device)]
    return result, permutation, seed


def _overlap(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.lower().split()), set(right.lower().split())
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens or right_tokens else 1.0


def _median_spacing(times: list[float]) -> float | None:
    return statistics.median([right - left for left, right in zip(times, times[1:])]) if len(times) > 1 else None


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--output-root", type=Path, default=Path("stage2_audit/frame_density"))
    parser.add_argument("--densities", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="Reuse completed per-video reports after an interrupted inference-only run.")
    args = parser.parse_args()
    if sorted(set(args.densities)) != args.densities or 32 not in args.densities or any(value < 1 for value in args.densities):
        raise ValueError("densities must be increasing positive integers and include 32.")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty directory: {args.output_root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU machine.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    caption_config = DirectMobileCLIPCaptionConfig(**checkpoint["model_config"])
    if caption_config.use_stage1_temporal_context:
        raise ValueError("This experiment requires the v3 direct-only checkpoint with Stage 1 context disabled.")
    captioner = DirectMobileCLIPCaptionModel.from_pretrained(caption_config)
    captioner.load_state_dict(checkpoint["model_state"], strict=True)
    captioner.to(device).eval()
    visual_config = TinyTraceConfig(max_frames=max(args.densities), visual_encoder_chunk_size=16)
    encoder = MobileCLIPSpatialEncoder(visual_config).to(device).eval()
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=Stage0Config.from_json(args.stage0_config), split="val")
    by_id = {str(item["video_id"]): index for index, item in enumerate(dataset.items)}
    all_rows: dict[str, dict[str, object]] = {}
    aggregate_rows: dict[int, list[dict[str, object]]] = {density: [] for density in args.densities}
    for directory in sorted(args.audit_root.glob("video_*")):
        truth = json.loads((directory / "ground_truth.json").read_text(encoding="utf-8"))["audit_focus_event"]
        video_id = directory.name.split("_", 2)[-1]
        existing = args.output_root / directory.name / "result.json"
        if args.resume and existing.is_file():
            result = json.loads(existing.read_text(encoding="utf-8"))
            if set(result.get("conditions", {})) != {str(value) for value in args.densities}:
                raise ValueError(f"Existing report has incompatible densities: {existing}")
            changed_existing = False
            for condition in result["conditions"].values():
                if "out_of_window_selected_frame_indices" not in condition:
                    inside = set(condition["available_frame_indices_inside_event"])
                    outside = [item for item in condition["selected_frame_indices"] if item not in inside]
                    condition["out_of_window_selected_frame_indices"] = outside
                    condition["all_selected_frames_inside_event"] = not outside
                    changed_existing = True
            if changed_existing:
                _json(existing, result)
            for density in args.densities:
                condition = result["conditions"][str(density)]
                aggregate_rows[density].append({"reference": str(result["reference_caption"]), "real": str(condition["real_caption"]), "shuffled": str(condition["shuffled_caption"]), "overlap": float(condition["real_vs_shuffled_token_overlap"])})
            all_rows[video_id] = result
            continue
        sample = dataset[by_id[video_id]]
        duration = float(sample["duration"])
        start, end = float(truth["start"]), float(truth["end"])
        segment = torch.tensor([start / duration, end / duration], device=device).view(1, 1, 2)
        video_files = list(directory.glob("video.*"))
        if len(video_files) != 1:
            raise ValueError(f"Expected exactly one copied source video in {directory}")
        conditions: dict[str, object] = {}
        for density in args.densities:
            if density == 32:
                features = sample["visual_features"].unsqueeze(0).to(device)  # type: ignore[union-attr]
                times = sample["frame_times"].unsqueeze(0).to(device)  # type: ignore[union-attr]
                source = "existing_read_only_cache"
            else:
                frames, dense_times = _read_uniform_frames(video_files[0], duration, density, visual_config.image_size)
                feature_chunks = [encoder(chunk.to(device)) for chunk in frames.split(visual_config.visual_encoder_chunk_size)]
                features = torch.cat(feature_chunks).unsqueeze(0)
                times = dense_times.unsqueeze(0).to(device)
                source = "inference_only_source_decode_and_frozen_mobileclip"
            frame_mask = torch.ones(times.shape, dtype=torch.bool, device=device)
            indices, selected_mask = event_frame_indices(times, frame_mask, segment, caption_config.max_event_frames)
            selected, selected_mask_again = select_event_patch_features(features, times, frame_mask, segment, caption_config.max_event_frames)
            if not torch.equal(selected_mask, selected_mask_again):
                raise RuntimeError("Selected feature mask disagrees with selected frame indices.")
            event_features, event_mask = selected[0], selected_mask[0]
            real_caption = captioner.generate(event_features, event_mask)[0]
            shuffled_features, permutation, seed = _shuffle(event_features[0], event_mask[0], video_id)
            shuffled_caption = captioner.generate(shuffled_features.unsqueeze(0), event_mask)[0]
            selected_count = int(event_mask[0].sum())
            all_times = [float(value) for value in times[0].tolist()]
            in_event = [index for index, value in enumerate(all_times) if start <= value <= end]
            selected_indices = indices[0, 0, :selected_count].tolist()
            out_of_window_selected = [index for index in selected_indices if index not in in_event]
            conditions[str(density)] = {
                "source": source,
                "source_frame_count": len(all_times),
                "source_frame_timestamps": all_times,
                "event_duration_seconds": end - start,
                "median_source_spacing_seconds": _median_spacing(all_times),
                "available_frames_inside_event": len(in_event),
                "available_frame_indices_inside_event": in_event,
                "selected_event_frames": selected_count,
                "selected_frame_indices": selected_indices,
                "selected_frame_timestamps": [all_times[index] for index in selected_indices],
                "all_selected_frames_inside_event": not out_of_window_selected,
                "out_of_window_selected_frame_indices": out_of_window_selected,
                "event_feature_shape": list(event_features.shape),
                "real_caption": real_caption,
                "shuffled_caption": shuffled_caption,
                "real_vs_shuffled_token_overlap": _overlap(real_caption, shuffled_caption),
                "whether_real_equals_shuffled": real_caption == shuffled_caption,
                "event_frame_permutation": permutation,
                "permutation_seed": seed,
            }
            aggregate_rows[density].append({"reference": str(truth["reference_caption"]), "real": real_caption, "shuffled": shuffled_caption, "overlap": _overlap(real_caption, shuffled_caption)})
        result = {"video_id": video_id, "reference_caption": str(truth["reference_caption"]), "ground_truth_start": start, "ground_truth_end": end, "conditions": conditions}
        _json(args.output_root / directory.name / "result.json", result)
        all_rows[video_id] = result
    aggregates: dict[str, object] = {}
    for density, rows in aggregate_rows.items():
        real, shuffled, references = [str(row["real"]) for row in rows], [str(row["shuffled"]) for row in rows], [str(row["reference"]) for row in rows]
        overlaps = [float(row["overlap"]) for row in rows]
        aggregates[str(density)] = {"real_metrics": caption_metrics(real, references), "shuffled_metrics": caption_metrics(shuffled, references), "mean_real_vs_shuffled_token_overlap": sum(overlaps) / len(overlaps), "identical_caption_rate": sum(row["real"] == row["shuffled"] for row in rows) / len(rows), "caption_change_rate_under_shuffling": sum(row["real"] != row["shuffled"] for row in rows) / len(rows)}
    baseline = aggregates["32"]["real_metrics"]  # type: ignore[index]
    denser_improved = [density for density in args.densities if density != 32 and aggregates[str(density)]["real_metrics"]["meteor_unigram"] > baseline["meteor_unigram"] and aggregates[str(density)]["real_metrics"]["cider_unigram"] > baseline["cider_unigram"]]  # type: ignore[index]
    verdict = "FRAME DENSITY IS A PRIMARY BOTTLENECK" if denser_improved else "FRAME DENSITY IS NOT A PRIMARY BOTTLENECK"
    aggregate = {"experiment": "frame_density_inference_ablation", "training_performed": False, "mobileclip_modified": False, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_epoch": checkpoint.get("epoch"), "stage1_used": False, "adapter_max_event_frames": caption_config.max_event_frames, "important_constraint": "The fixed v3 adapter accepts at most 8 selected event frames. 64/128-frame conditions therefore test whether denser whole-video sampling gives the frozen adapter better event-frame choices, not whether the adapter can consume 64/128 visual frames simultaneously.", "densities": aggregates, "verdict": verdict, "verdict_reason": "A density is counted as an improvement only when both METEOR-like and CIDEr-like real-caption scores exceed the exact cached-32 baseline; caption changes alone are not evidence."}
    _json(args.output_root / "aggregate_metrics.json", aggregate)
    headers = ["Video", "Reference", "32 Real", "64 Real", "128 Real", "32 Shuffle", "64 Shuffle", "128 Shuffle"]
    body = []
    for video_id, row in all_rows.items():
        conditions = row["conditions"]
        body.append([video_id, str(row["reference_caption"])] + [str(conditions[str(density)]["real_caption"]) for density in args.densities] + [str(conditions[str(density)]["shuffled_caption"]) for density in args.densities])
    density_headers = ["Video", "Event seconds", "32 in event", "64 in event", "128 in event", "32 median spacing", "64 median spacing", "128 median spacing", "Selection flags"]
    density_rows = []
    for video_id, row in all_rows.items():
        conditions = row["conditions"]
        flags = "; ".join(f"{density}:{conditions[str(density)]['out_of_window_selected_frame_indices'] or 'ok'}" for density in args.densities)
        density_rows.append([video_id, f"{float(conditions['32']['event_duration_seconds']):.2f}"] + [str(conditions[str(density)]["available_frames_inside_event"]) for density in args.densities] + ["n/a" if conditions[str(density)]["median_source_spacing_seconds"] is None else f"{float(conditions[str(density)]['median_source_spacing_seconds']):.3f}s" for density in args.densities] + [flags])
    table = "# Frame-Density Ablation\n\n**" + verdict + "**\n\nThis is inference-only. The existing cache/checkpoint and MobileCLIP weights were read-only.\n\n## Captions\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in body) + "\n\n## Event Sampling Coverage\n\n| " + " | ".join(density_headers) + " |\n| " + " | ".join(["---"] * len(density_headers)) + " |\n" + "\n".join("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |" for row in density_rows) + "\n\n`ok` means every selected frame was within the literal annotation window. `32:[30]` flags a pre-existing v3 normalized-time selection that chose cache index 30 outside the literal event window even though in-window cache frames existed; it is retained rather than silently corrected so the baseline stays exact.\n\n## Aggregate Results\n```json\n" + json.dumps(aggregate, indent=2) + "\n```\n\nThe higher-density source conditions select at most eight event frames because that is the fixed trained adapter capacity. Their uniform-grid features are out-of-distribution relative to the cache sampling grid, so degradation means the frozen bridge does not benefit from denser sampling without retraining; it does not prove that a newly trained higher-density system could never improve. Per-video `result.json` files include every source frame timestamp, event-in-window count, selected frame indices/timestamps, and deterministic permutation.\n"
    (args.output_root / "SUMMARY.md").write_text(table, encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root.resolve()), "verdict": verdict, "densities": args.densities}, sort_keys=True))


if __name__ == "__main__":
    main()
