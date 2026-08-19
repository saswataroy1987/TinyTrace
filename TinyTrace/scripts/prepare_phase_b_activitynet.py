from __future__ import annotations

import argparse
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable


VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ActivityNet Captions annotations into TinyTrace Phase-B "
            "video-only event JSON."
        )
    )
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--output-train-json", type=Path, required=True)
    parser.add_argument("--output-val-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=6)
    parser.add_argument("--default-score", type=float, default=1.0)
    parser.add_argument(
        "--media-validation-timeout-seconds",
        type=float,
        default=180.0,
        help="Maximum time allowed to fully decode one video during validation.",
    )
    parser.add_argument(
        "--media-validation-workers",
        type=int,
        default=8,
        help="Parallel FFprobe workers used for the fast media pre-check.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _video_candidates(video_id: str) -> list[str]:
    return [f"{video_id}{suffix}" for suffix in VIDEO_SUFFIXES]


def _build_video_index(videos_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in videos_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        index.setdefault(path.stem, path)
    return index


def _normalize_text(value: object) -> str:
    text = str(value).strip()
    return " ".join(text.split())


def _validate_video(video_path: Path, timeout_seconds: float) -> str | None:
    """Fast, non-destructive media pre-check; decoding happens during caching."""
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        return "file is missing or empty"
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type,duration:format=duration",
                "-of",
                "json",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"ffprobe failed: {exc}"
    if probe.returncode != 0:
        return f"ffprobe failed: {probe.stderr.strip() or f'exit status {probe.returncode}'}"
    try:
        payload = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        return f"ffprobe returned invalid JSON: {exc}"
    if not payload.get("streams"):
        return "no readable video stream"

    return None


def _validate_video_index(
    video_index: dict[str, Path], timeout_seconds: float, workers: int
) -> dict[Path, str | None]:
    paths = sorted(set(video_index.values()))
    results: dict[Path, str | None] = {}
    print(f"Fast-validating {len(paths)} videos with {workers} FFprobe workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_validate_video, path, timeout_seconds): path for path in paths}
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                results[path] = future.result()
            except Exception as exc:  # pragma: no cover - defensive worker boundary
                results[path] = f"media pre-check failed: {exc}"
            if completed % 250 == 0 or completed == len(paths):
                rejected = sum(reason is not None for reason in results.values())
                print(f"Fast validation [{completed}/{len(paths)}] rejected={rejected}", flush=True)
    return results


def _build_events(
    *,
    sentences: list[object],
    timestamps: list[object],
    max_events: int,
    default_score: float,
) -> list[dict]:
    if len(sentences) != len(timestamps):
        raise ValueError("ActivityNet sample has mismatched sentences/timestamps lengths.")
    pairs = []
    for sentence, timestamp in zip(sentences, timestamps):
        if (
            not isinstance(timestamp, list)
            or len(timestamp) != 2
        ):
            continue
        caption = _normalize_text(sentence)
        if not caption:
            continue
        try:
            start = float(timestamp[0])
            end = float(timestamp[1])
        except (TypeError, ValueError):
            continue
        if not (start >= 0.0 and end > start):
            continue
        pairs.append(
            {
                "timestamp": [round(start, 3), round(end, 3)],
                "score": [float(default_score)],
                "caption": caption,
            }
        )
    pairs.sort(key=lambda item: (item["timestamp"][0], item["timestamp"][1], item["caption"]))
    return pairs[:max_events]


def _convert_split(
    *,
    payload: dict,
    split_name: str,
    video_index: dict[str, Path],
    videos_root: Path,
    max_events: int,
    default_score: float,
    media_validator: Callable[[Path], str | None] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    kept = 0
    skipped_missing_video = 0
    skipped_invalid_video = 0
    skipped_invalid_annotation = 0
    skipped_empty_events = 0
    converted: list[dict] = []

    for video_id, entry in sorted(payload.items()):
        if not isinstance(entry, dict):
            continue
        video_path = video_index.get(video_id)
        if video_path is None:
            skipped_missing_video += 1
            continue
        if media_validator is not None:
            rejection_reason = media_validator(video_path)
            if rejection_reason is not None:
                skipped_invalid_video += 1
                continue
        sentences = entry.get("sentences", [])
        timestamps = entry.get("timestamps", [])
        if not isinstance(sentences, list) or not isinstance(timestamps, list):
            skipped_invalid_annotation += 1
            continue
        try:
            events = _build_events(
                sentences=sentences,
                timestamps=timestamps,
                max_events=max_events,
                default_score=default_score,
            )
        except (TypeError, ValueError):
            skipped_invalid_annotation += 1
            continue
        if not events:
            skipped_empty_events += 1
            continue
        try:
            duration = float(entry.get("duration", 0.0))
        except (TypeError, ValueError):
            skipped_invalid_annotation += 1
            continue
        if not math.isfinite(duration) or duration <= 0.0:
            skipped_invalid_annotation += 1
            continue
        if max(event["timestamp"][1] for event in events) > duration + 0.5:
            skipped_invalid_annotation += 1
            continue
        relative_video_path = video_path.resolve().relative_to(videos_root.resolve())
        converted.append(
            {
                "source_id": f"activitynet_{video_id}",
                "video_id": video_id,
                "video_path": str(Path("videos") / relative_video_path),
                "duration_seconds": round(duration, 3) if duration > 0 else None,
                "instruction": "Watch the video and emit structured events as timestamp, score, and caption.",
                "task_mode": "caption",
                "events": events,
                "dataset_name": "ActivityNet Captions",
                "split": split_name,
            }
        )
        kept += 1

    return converted, {
        "kept": kept,
        "skipped_missing_video": skipped_missing_video,
        "skipped_invalid_video": skipped_invalid_video,
        "skipped_invalid_annotation": skipped_invalid_annotation,
        "skipped_empty_events": skipped_empty_events,
    }


def main() -> None:
    args = parse_args()
    if args.media_validation_timeout_seconds <= 0:
        raise ValueError("--media-validation-timeout-seconds must be positive.")
    if args.media_validation_workers < 1:
        raise ValueError("--media-validation-workers must be positive.")
    video_index = _build_video_index(args.videos_root)
    validation_results = _validate_video_index(
        video_index, args.media_validation_timeout_seconds, args.media_validation_workers
    )
    train_payload = _load_json(args.train_json)
    val_payload = _load_json(args.val_json)

    train_items, train_summary = _convert_split(
        payload=train_payload,
        split_name="train",
        video_index=video_index,
        videos_root=args.videos_root,
        max_events=args.max_events,
        default_score=args.default_score,
        media_validator=lambda path: validation_results.get(path, "video was not pre-validated"),
    )
    val_items, val_summary = _convert_split(
        payload=val_payload,
        split_name="val",
        video_index=video_index,
        videos_root=args.videos_root,
        max_events=args.max_events,
        default_score=args.default_score,
        media_validator=lambda path: validation_results.get(path, "video was not pre-validated"),
    )

    _write_json(args.output_train_json, train_items)
    _write_json(args.output_val_json, val_items)
    _write_json(
        args.manifest_json,
        {
            "dataset": "ActivityNet Captions",
            "train_json": str(args.train_json),
            "val_json": str(args.val_json),
            "videos_root": str(args.videos_root),
            "max_events": args.max_events,
            "default_score": args.default_score,
            "media_validation": {
                "method": "parallel ffprobe video-stream check; full frame decoding occurs during feature caching",
                "timeout_seconds": args.media_validation_timeout_seconds,
                "workers": args.media_validation_workers,
            },
            "train_summary": train_summary,
            "val_summary": val_summary,
        },
    )
    print(
        "Prepared ActivityNet Phase B dataset: "
        f"train={train_summary['kept']} val={val_summary['kept']} "
        f"missing_videos={train_summary['skipped_missing_video'] + val_summary['skipped_missing_video']} "
        f"invalid_videos={train_summary['skipped_invalid_video'] + val_summary['skipped_invalid_video']} "
        f"invalid_annotations={train_summary['skipped_invalid_annotation'] + val_summary['skipped_invalid_annotation']}"
    )


if __name__ == "__main__":
    main()
