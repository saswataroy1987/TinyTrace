from __future__ import annotations

import argparse
import json
from pathlib import Path


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
) -> tuple[list[dict], dict[str, int]]:
    kept = 0
    skipped_missing_video = 0
    skipped_empty_events = 0
    converted: list[dict] = []

    for video_id, entry in sorted(payload.items()):
        if not isinstance(entry, dict):
            continue
        video_path = video_index.get(video_id)
        if video_path is None:
            skipped_missing_video += 1
            continue
        events = _build_events(
            sentences=list(entry.get("sentences", [])),
            timestamps=list(entry.get("timestamps", [])),
            max_events=max_events,
            default_score=default_score,
        )
        if not events:
            skipped_empty_events += 1
            continue
        try:
            duration = float(entry.get("duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
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
        "skipped_empty_events": skipped_empty_events,
    }


def main() -> None:
    args = parse_args()
    video_index = _build_video_index(args.videos_root)
    train_payload = _load_json(args.train_json)
    val_payload = _load_json(args.val_json)

    train_items, train_summary = _convert_split(
        payload=train_payload,
        split_name="train",
        video_index=video_index,
        videos_root=args.videos_root,
        max_events=args.max_events,
        default_score=args.default_score,
    )
    val_items, val_summary = _convert_split(
        payload=val_payload,
        split_name="val",
        video_index=video_index,
        videos_root=args.videos_root,
        max_events=args.max_events,
        default_score=args.default_score,
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
            "train_summary": train_summary,
            "val_summary": val_summary,
        },
    )
    print(
        "Prepared ActivityNet Phase B dataset: "
        f"train={train_summary['kept']} val={val_summary['kept']} "
        f"missing_videos={train_summary['skipped_missing_video'] + val_summary['skipped_missing_video']}"
    )


if __name__ == "__main__":
    main()
