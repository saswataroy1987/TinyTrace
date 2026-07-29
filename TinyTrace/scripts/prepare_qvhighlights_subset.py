from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

BAD_VIDEO_NAMES = {
    "T2O2eC8SdDk_360.0_510.0.mp4",
    "dW4wpGg64pE_60.0_210.0.mp4",
    "p9xXyLDqcMQ_60.0_210.0.mp4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=str, default="dataset/mt_fmt-8k.json")
    parser.add_argument("--video-dir", type=str, default="qvhighlights/videos/train")
    parser.add_argument("--output-json", type=str, default="TinyTrace/data/qvh_tinytrace_subset.json")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-events", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=3.5)
    return parser.parse_args()


def extract_query(prompt: str) -> str:
    match = re.search(r"sentence query: '(.*?)'\. Please return", prompt, flags=re.DOTALL)
    if match:
        return match.group(1)
    return prompt.replace("<video>\n", "").strip()


def is_video_decodable(video_path: Path) -> bool:
    if video_path.name in BAD_VIDEO_NAMES:
        return False

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if probe.returncode != 0:
        return False

    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        return False
    if duration <= 0.0:
        return False

    sample_time = min(max(duration * 0.25, 0.5), max(duration - 1.0, 0.5))
    decode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            f"{sample_time:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return decode.returncode == 0 and not decode.stderr


def build_events(times: list[list[float]], scores: list[list[float]], max_events: int, score_threshold: float, caption: str) -> list[dict]:
    points = [(float(t[0]), float(s[0])) for t, s in zip(times, scores) if t and s]
    segments = []
    current = None

    for time_value, score_value in points:
        if score_value < score_threshold:
            if current is not None:
                segments.append(current)
                current = None
            continue

        if current is None:
            current = {"start": time_value, "end": time_value, "scores": [score_value]}
        elif abs(time_value - current["end"]) <= 0.51:
            current["end"] = time_value
            current["scores"].append(score_value)
        else:
            segments.append(current)
            current = {"start": time_value, "end": time_value, "scores": [score_value]}

    if current is not None:
        segments.append(current)

    if not segments:
        top_points = sorted(points, key=lambda pair: pair[1], reverse=True)[:max_events]
        segments = [{"start": t, "end": t, "scores": [s]} for t, s in top_points]

    segments = sorted(segments, key=lambda seg: (sum(seg["scores"]) / len(seg["scores"])), reverse=True)[:max_events]
    segments = sorted(segments, key=lambda seg: seg["start"])

    return [
        {
            "timestamp": [round(seg["start"], 1), round(seg["end"], 1)],
            "score": [round(sum(seg["scores"]) / len(seg["scores"]), 1)],
            "caption": caption,
        }
        for seg in segments
    ]


def main() -> None:
    args = parse_args()
    source = Path(args.source_json)
    video_dir = Path(args.video_dir)
    output = Path(args.output_json)

    payload = json.loads(source.read_text())
    converted = []

    for item in payload:
        video_name = Path(item["video"]).name
        local_video_path = video_dir / video_name
        if not local_video_path.exists():
            continue
        if not is_video_decodable(local_video_path):
            print(f"skipping unreadable video: {local_video_path}")
            continue

        query = extract_query(item["conversations"][0]["value"])
        events = build_events(item["times"], item["scores"], args.max_events, args.score_threshold, query)
        converted.append(
            {
                "source_id": item["id"],
                "video_path": str(local_video_path),
                "instruction": "localize highlight events and describe them",
                "events": events,
            }
        )
        if len(converted) >= args.max_samples:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(converted, indent=2))
    print(f"saved {len(converted)} samples to {output}")


if __name__ == "__main__":
    main()
