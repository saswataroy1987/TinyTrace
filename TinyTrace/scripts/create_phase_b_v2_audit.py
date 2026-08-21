"""Create a self-contained, deterministic 10-video forensic audit package.

This script performs inference only. It never trains a model or writes to the
read-only MobileCLIP cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2 import ActivityNetV2Dataset, PhaseBV2Config, PhaseBV2Model, Stage0Config, activitynet_v2_collate_fn, filter_events
from tinytrace.phase_b_v2.temporal import temporal_iou


# All IDs come from Stage 2's deterministic validation comparison export. The
# selected ten are chosen below from this pool using measured Phase 1 quality.
CANDIDATE_IDS = (
    "v_--1DO2V4K74", "v_-01K1HxqPB8", "v_-02DygXbn6w", "v_-0r0HEwAYiQ",
    "v_-2VzSMAdzl4", "v_-5c9WHk408g", "v_-76d-7Ju7L0", "v_-79MZQX4CEA",
    "v_-7eQ2bHNPUw", "v_-CEi03j4-Bw", "v_-DGsqL65o4k", "v_-DpnaHTk8PA",
    "v_-DzTAnE1t3w", "v_-E2dqOULQgY", "v_-E9YQ_Uhu50", "v_-F7QWQA8Eh8",
    "v_-FWGLSfI13Q", "v_-GRvxWH4axc",
)


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _format(seconds: float) -> str:
    minutes, remainder = divmod(max(seconds, 0.0), 60.0)
    return f"{int(minutes):02d}:{remainder:04.1f}"


def _ffprobe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return None


def _event_iou(predicted: dict[str, float], segment: torch.Tensor) -> float:
    return float(temporal_iou(torch.tensor([[predicted["start"], predicted["end"]]]), segment.unsqueeze(0))[0])


def _to_device(value: dict[str, object], device: torch.device) -> dict[str, object]:
    return {key: item.to(device) if isinstance(item, torch.Tensor) else item for key, item in value.items()}


def _load_phase1(config: PhaseBV2Config, checkpoint: Path, device: torch.device) -> PhaseBV2Model:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = PhaseBV2Model(PhaseBV2Config(**{**config.to_dict(), "stage": "localization"}))
    model.load_state_dict(payload["model_state"], strict=True)
    return model.to(device).eval()


def _phase1_predictions(model: PhaseBV2Model, sample: dict[str, object], device: torch.device, threshold: float, overlap: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    raw = activitynet_v2_collate_fn([sample])
    batch = _to_device(raw, device)
    with torch.no_grad():
        outputs, _ = model.forward_localization(batch)
    segments = outputs["segments"][0].detach().cpu()
    probabilities = outputs["event_logits"][0].sigmoid().detach().cpu()
    filtered = filter_events(segments, torch.logit(probabilities.clamp(1e-6, 1 - 1e-6)), threshold, overlap)
    raw_queries = [
        {"query_index": index, "start_normalized": float(segment[0]), "end_normalized": float(segment[1]), "confidence": float(probabilities[index])}
        for index, segment in enumerate(segments.tolist())
    ]
    return filtered, raw_queries


def _choose_ids(records: dict[str, dict[str, Any]]) -> list[str]:
    """Five strongest and five weakest Phase 1 examples, deterministically."""
    ranked = sorted(records, key=lambda item: (records[item]["phase1_mean_best_iou"], item))
    low, high = ranked[:5], list(reversed(ranked[-5:]))
    selected: list[str] = []
    for item in low + high:
        if item not in selected:
            selected.append(item)
    for item in ranked:
        if len(selected) == 10:
            break
        if item not in selected:
            selected.append(item)
    return selected


def _contact_sheet(video: Path, destination: Path, timestamps: list[float]) -> str | None:
    if not timestamps:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return "Pillow unavailable; contact sheet was not created."
    thumbs: list[tuple[Any, str]] = []
    temporary = destination.parent / ".frames"
    temporary.mkdir(exist_ok=True)
    for index, timestamp in enumerate(timestamps):
        frame = temporary / f"{index:02d}.jpg"
        try:
            subprocess.run(["ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=240:-2", str(frame)], check=True, capture_output=True)
            image = Image.open(frame).convert("RGB")
            thumbs.append((image, _format(timestamp)))
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            continue
    shutil.rmtree(temporary, ignore_errors=True)
    if not thumbs:
        return "ffmpeg could not extract thumbnails; inspect the copied video directly."
    cell_width, cell_height, columns = 250, 180, 4
    rows = math.ceil(len(thumbs) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(thumbs):
        image.thumbnail((240, 145))
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        canvas.paste(image, (x + 5, y + 5))
        draw.text((x + 5, y + 154), label, fill="black")
    canvas.save(destination)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-map", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--phase2-comparisons", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--overlap-threshold", type=float, default=0.7)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a non-empty audit directory: {args.output_root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable; use --device cpu or run on the GPU PC.")

    config, stage0 = PhaseBV2Config.from_json(args.model_config), Stage0Config.from_json(args.stage0_config)
    model = _load_phase1(config, args.stage1_checkpoint, device)
    dataset = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=stage0, split="val")
    samples = {str(item["video_id"]): item for item in dataset.items if str(item["video_id"]) in CANDIDATE_IDS}
    if set(samples) != set(CANDIDATE_IDS):
        raise ValueError(f"Candidate IDs missing from validation manifest: {sorted(set(CANDIDATE_IDS) - set(samples))}")
    mapping = {item["video_id"]: item for item in json.loads(args.cache_map.read_text(encoding="utf-8"))["entries"]}
    comparisons: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in json.loads(args.phase2_comparisons.read_text(encoding="utf-8"))["comparisons"]:
        comparisons[str(item["video_id"])].append(item)

    records: dict[str, dict[str, Any]] = {}
    for video_id in CANDIDATE_IDS:
        sample = dataset[dataset.items.index(samples[video_id])]
        predicted, raw_queries = _phase1_predictions(model, sample, device, args.threshold, args.overlap_threshold)
        target_segments = sample["segments"]
        best_ious = [max((_event_iou(item, segment) for item in predicted), default=0.0) for segment in target_segments]
        records[video_id] = {"sample": sample, "predicted": predicted, "raw_queries": raw_queries, "phase1_mean_best_iou": sum(best_ious) / len(best_ious), "phase1_best_iou": max(best_ious, default=0.0)}
    selected = _choose_ids(records)

    args.output_root.mkdir(parents=True)
    summary_rows: list[dict[str, str]] = []
    alignment_rows: list[str] = []
    selection = []
    for ordinal, video_id in enumerate(selected, start=1):
        record, sample = records[video_id], records[video_id]["sample"]
        item, map_item = samples[video_id], mapping.get(video_id)
        if map_item is None:
            raise ValueError(f"No cache-map entry for {video_id}")
        source = Path(str(map_item["source_video_path_used_for_v1_hash"]))
        if not source.is_file():
            raise FileNotFoundError(f"Source video missing for {video_id}: {source}")
        directory = args.output_root / f"video_{ordinal:02d}_{video_id}"
        directory.mkdir()
        copied = directory / f"video{source.suffix.lower()}"
        shutil.copy2(source, copied)
        duration = float(sample["duration"])
        video_duration = _ffprobe_duration(copied)
        features, times = sample["visual_features"], sample["frame_times"]
        cache_path = args.cache_root / str(item["visual_feature_path"])
        events = [{"event_index": index, "start": float(row[0]), "end": float(row[1]), "caption": str(sample["captions"][index])} for index, row in enumerate(sample["segments_seconds"].tolist())]
        target = comparisons[video_id][0] if comparisons[video_id] else {"start": events[0]["start"], "end": events[0]["end"], "reference_caption": events[0]["caption"], "generated_caption": "Unavailable: no Phase 2 comparison export for this event."}
        target_segment = torch.tensor([float(target["start"]) / duration, float(target["end"]) / duration])
        matched = max(record["predicted"], key=lambda row: _event_iou(row, target_segment), default=None)
        matched_iou = _event_iou(matched, target_segment) if matched else 0.0
        intervals = [float(times[index] - times[index - 1]) for index in range(1, len(times))]
        interval = median(intervals) if intervals else 0.0
        target_frame_indices = [index for index, value in enumerate(times.tolist()) if float(target["start"]) <= value <= float(target["end"])]
        if not target_frame_indices:
            target_frame_indices = [min(range(len(times)), key=lambda index: abs(float(times[index]) - (float(target["start"]) + float(target["end"])) / 2))]
        if len(target_frame_indices) > 16:
            target_frame_indices = [target_frame_indices[round(index * (len(target_frame_indices) - 1) / 15)] for index in range(16)]
        flags = []
        if abs((video_duration or duration) - duration) > 0.5:
            flags.append("SUSPICIOUS: ffprobe duration differs from manifest duration by more than 0.5 seconds.")
        if str(map_item["video_id"]) != video_id or source.stem != video_id:
            flags.append("SUSPICIOUS: cache mapping video ID and source filename do not agree.")
        if any(float(value) < 0 or float(value) > duration + 0.5 for value in times):
            flags.append("SUSPICIOUS: cached frame time lies outside the manifest duration.")
        if any(event["start"] < 0 or event["end"] > duration or event["end"] <= event["start"] for event in events):
            flags.append("SUSPICIOUS: invalid ground-truth timestamp in manifest.")
        if not flags:
            flags.append("No automatic cache/video/timestamp mismatch found. Manual visual inspection is still required.")
        with (directory / "frame_timeline.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["frame_index", "timestamp_seconds", "cache_index", "inside_ground_truth_event", "inside_phase1_predicted_event"])
            writer.writeheader()
            for index, value in enumerate(times.tolist()):
                writer.writerow({"frame_index": index, "timestamp_seconds": f"{value:.6f}", "cache_index": index, "inside_ground_truth_event": str(float(target["start"]) <= value <= float(target["end"])).lower(), "inside_phase1_predicted_event": str(bool(matched and matched["start"] * duration <= value <= matched["end"] * duration)).lower()})
        _json(directory / "ground_truth.json", {"video_id": video_id, "manifest_duration_seconds": duration, "events": events, "audit_focus_event": target})
        _json(directory / "phase1_prediction.json", {"threshold": args.threshold, "overlap_threshold": args.overlap_threshold, "filtered_predictions_normalized": record["predicted"], "filtered_predictions_seconds": [{**row, "start_seconds": row["start"] * duration, "end_seconds": row["end"] * duration} for row in record["predicted"]], "raw_query_predictions": record["raw_queries"], "audit_focus_match": {"prediction": matched, "iou_with_ground_truth": matched_iou}})
        (directory / "phase2_caption.txt").write_text(f"Checkpoint: {json.loads(args.phase2_comparisons.read_text(encoding='utf-8'))['checkpoint']}\nSegment source: ground_truth\n\nGround truth: {target['reference_caption']}\nGenerated: {target['generated_caption']}\n", encoding="utf-8")
        contact_error = _contact_sheet(copied, directory / "contact_sheet.jpg", [float(times[index]) for index in target_frame_indices])
        cache_section = f"- Cache file: `{cache_path}`\n- Cache tensor shape: `{list(features.shape)}` (`[cached_frames, patch_tokens, feature_dim]`)\n- Cached frame count: `{len(times)}`\n- Median sampling interval: `{interval:.3f}` seconds\n- Focus-event cached frame indices: `{target_frame_indices}`\n- Focus-event cached timestamps: `{[round(float(times[index]), 3) for index in target_frame_indices]}`\n- Event/cached-frame overlap: `{len([value for value in times.tolist() if float(target['start']) <= value <= float(target['end'])])}` cached frames lie inside the focus event.\n"
        if contact_error:
            cache_section += f"- Contact sheet: {contact_error}\n"
        match_text = "No filtered Phase 1 event overlaps the focus ground-truth event." if matched is None else f"Predicted `{_format(matched['start'] * duration)} - {_format(matched['end'] * duration)}` ({matched['start'] * duration:.2f}s - {matched['end'] * duration:.2f}s), confidence `{matched['score']:.3f}`, IoU `{matched_iou:.3f}`."
        audit = f"# Video {ordinal:02d}: {video_id}\n\n## Video\n- Copied source: `{copied.name}`\n- Source path used by V1 cache hash: `{source}`\n- Manifest duration: `{duration:.2f}` seconds\n- ffprobe duration: `{video_duration:.2f}` seconds\n\n## Ground Truth\n- **Audit focus event:** `{target['reference_caption']}`\n- **Start:** `{_format(float(target['start']))}` ({float(target['start']):.2f}s)\n- **End:** `{_format(float(target['end']))}` ({float(target['end']):.2f}s)\n- **Event duration:** `{float(target['end']) - float(target['start']):.2f}` seconds\n- All events are in `ground_truth.json`.\n\n## Phase 1 Prediction\n{match_text}\n- All filtered events and all 32 raw queries are in `phase1_prediction.json`.\n\n## Phase 2 Caption\n- Generated from the **ground-truth** focus window: \"{target['generated_caption']}\"\n- This is not an end-to-end detector caption; it isolates the current visual-to-language path.\n\n## Quick Comparison\n- Ground truth: {target['reference_caption']}\n- Phase 1: {match_text}\n- Phase 2: \"{target['generated_caption']}\"\n\n## Cache / Frame Mapping\n{cache_section}\n\n## Alignment Status\n" + "\n".join(f"- {flag}" for flag in flags) + "\n\n## Manual Review\n1. Open the copied video and jump to the ground-truth start, midpoint, and end.\n2. Compare what you see with the focus caption.\n3. Jump to the Phase 1 predicted interval, then compare its boundary and confidence.\n4. Open `contact_sheet.jpg` and `frame_timeline.csv`; verify cached timestamps correspond to visible event content.\n5. Decide whether Phase 2 is wrong because visual evidence is wrong/misaligned or because the language conditioning ignores correct evidence.\n"
        (directory / "audit.md").write_text(audit, encoding="utf-8")
        category = str(target["reference_caption"])
        selection.append({"ordinal": ordinal, "video_id": video_id, "phase1_mean_best_iou": record["phase1_mean_best_iou"], "phase1_best_iou": record["phase1_best_iou"], "phase2_generated_caption": target["generated_caption"], "ground_truth_focus_caption": category, "selection_reason": "Five lowest mean-best-IoU candidates" if video_id in _choose_ids(records)[:5] else "Five highest mean-best-IoU candidates"})
        summary_rows.append({"Video": directory.name, "Actual Event": category, "GT Start": f"{float(target['start']):.2f}", "GT End": f"{float(target['end']):.2f}", "Phase1 Start": "" if not matched else f"{matched['start'] * duration:.2f}", "Phase1 End": "" if not matched else f"{matched['end'] * duration:.2f}", "Phase1 Confidence": "" if not matched else f"{matched['score']:.3f}", "Phase2 Caption": str(target["generated_caption"]), "Alignment Status": "FLAGGED" if any(flag.startswith("SUSPICIOUS") for flag in flags) else "No automatic mismatch"})
        alignment_rows.append(f"## Video {ordinal:02d}: {video_id}\n" + "\n".join(f"- {flag}" for flag in flags) + f"\n- Source video: `{source}`\n- Cache: `{cache_path}`\n- Manifest/cache time base: seconds from video start; cache range `{float(times.min()):.3f}` to `{float(times.max()):.3f}` seconds, manifest duration `{duration:.3f}` seconds.\n")
    _json(args.output_root / "selection.json", {"selection_method": "Deterministic: five lowest and five highest Phase 1 mean-best-IoU videos from the fixed Stage 2 comparison candidate list.", "candidate_ids": list(CANDIDATE_IDS), "selected": selection})
    headers = ["Video", "Actual Event", "GT Start", "GT End", "Phase1 Start", "Phase1 End", "Phase1 Confidence", "Phase2 Caption", "Alignment Status"]
    summary = "# Stage 2 Manual Audit Summary\n\nThis package is self-contained: each directory contains a copied video, reports, timestamp timeline, cache evidence, and Phase 1/2 evidence. Phase 2 captions were generated from ground-truth windows.\n\n| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join("| " + " | ".join(row[header].replace("|", "\\|") for header in headers) + " |" for row in summary_rows) + "\n\nSelection details: `selection.json`.\n"
    (args.output_root / "SUMMARY.md").write_text(summary, encoding="utf-8")
    (args.output_root / "ALIGNMENT_CHECK.md").write_text("# Alignment Check\n\nAutomatic checks verify source-video existence, ID mapping, duration agreement, cache timestamp range, and ground-truth timestamp validity. They cannot prove that a cached feature visually depicts the expected frame, so inspect each copied video and contact sheet manually.\n\n" + "\n".join(alignment_rows), encoding="utf-8")
    (args.output_root / "README.md").write_text("# TinyTrace 10-Video Forensic Audit\n\nThis package was created without training or modifying model weights. It is designed for manual inspection before any Stage 2/3 model decision.\n\n## Per-video files\n- `video.*`: actual copied source video\n- `audit.md`: plain-language guide and result summary\n- `ground_truth.json`: all validated ActivityNet events\n- `phase1_prediction.json`: filtered events plus all raw query predictions\n- `phase2_caption.txt`: ground-truth caption and generated Stage 2 caption when available\n- `frame_timeline.csv`: every cached frame timestamp and event-membership flags\n- `contact_sheet.jpg`: cached-time thumbnails within the focus event\n\n## Manual procedure\n1. Read `audit.md`, then open the copied video. Jump to the ground-truth start, midpoint, and end.\n2. Confirm the ground-truth caption matches what is visible.\n3. Jump to the Phase 1 interval and compare it with the ground-truth interval.\n4. Compare cached timestamps in `frame_timeline.csv` and thumbnails with the visible video at the same times.\n5. Compare the generated Phase 2 caption with the actual event.\n\n## How to interpret findings\n- **Dataset issue:** ground-truth caption or time window does not describe the copied video.\n- **Cache/alignment issue:** video/cache ID, duration, or timestamp mapping is flagged, or thumbnails at cache times show a different video moment.\n- **Phase 1 issue:** ground truth is valid and cache frames align, but Phase 1 boundaries have low IoU or miss the event.\n- **Phase 2 conditioning issue:** ground truth and cache frames align, but the Stage 2 caption is wrong despite using the ground-truth event window.\n\nDo not start Stage 3 based on this package alone. Record findings in a copy of `SUMMARY.md` before changing the model.\n", encoding="utf-8")
    print(json.dumps({"output_root": str(args.output_root.resolve()), "selected_video_ids": selected, "selection_file": str((args.output_root / "selection.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
