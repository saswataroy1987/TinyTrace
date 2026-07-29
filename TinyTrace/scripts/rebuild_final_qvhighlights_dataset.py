from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the final TinyTrace-ready QVHighlights dataset folder from downloaded clips."
    )
    parser.add_argument("--source-json", type=str, default="dataset/qvhighlights/mt_fmt-2000-valid.json")
    parser.add_argument("--video-dir", type=str, default="dataset/qvhighlights/videos/train")
    parser.add_argument("--output-dir", type=str, default="final_qvhighlights_tinytrace")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=26)
    parser.add_argument("--link-mode", type=str, default="hardlink", choices=("hardlink", "copy"))
    return parser.parse_args()


def extract_query(prompt: str) -> str:
    match = re.search(r"sentence query: '(.*?)'\. Please return", prompt, flags=re.DOTALL)
    if match:
        return match.group(1)
    return prompt.replace("<video>\n", "").strip()


def build_events(
    times: list[list[float]],
    scores: list[list[float]],
    max_events: int,
) -> list[dict]:
    points = [(float(t[0]), float(s[0])) for t, s in zip(times, scores) if t and s]
    runs: list[dict] = []
    for time_value, score_value in points:
        if not runs:
            runs.append({"start": time_value, "end": time_value, "score": score_value, "points": 1})
        elif abs(time_value - runs[-1]["end"]) <= 0.51 and score_value == runs[-1]["score"]:
            runs[-1]["end"] = time_value
            runs[-1]["points"] += 1
        else:
            runs.append({"start": time_value, "end": time_value, "score": score_value, "points": 1})

    # Preserve the complete timeline under an edge-generation budget by
    # repeatedly merging the least-different adjacent saliency runs.
    while len(runs) > max_events:
        merge_index = min(
            range(len(runs) - 1),
            key=lambda index: abs(runs[index]["score"] - runs[index + 1]["score"]),
        )
        left = runs[merge_index]
        right = runs[merge_index + 1]
        points = left["points"] + right["points"]
        merged = {
            "start": left["start"],
            "end": right["end"],
            "score": (left["score"] * left["points"] + right["score"] * right["points"]) / points,
            "points": points,
        }
        runs[merge_index : merge_index + 2] = [merged]

    return [
        {
            "timestamp": [round(run["start"], 1), round(run["end"], 1)],
            "score": [round(run["score"], 1)],
        }
        for run in runs
    ]


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def write_readme(
    path: Path,
    total_count: int,
    train_count: int,
    val_count: int,
    source_json_name: str,
    seed: int,
) -> None:
    content = f"""# Final QVHighlights TinyTrace Dataset

This folder contains the cleaned TinyTrace-ready QVHighlights subset prepared from successfully downloaded valid clips.

## Structure

```text
final_qvhighlights_tinytrace/
├── videos/
│   ├── train/
│   └── val/
├── annotations/
│   ├── qvh_raw_valid.json
│   ├── tinytrace_train.json
│   ├── tinytrace_val.json
│   ├── train_ids.txt
│   ├── val_ids.txt
│   └── download_manifest.json
└── README.md
```

## Contents

- `videos/train/`: {train_count} valid QVHighlights clips
- `videos/val/`: {val_count} valid QVHighlights clips
- `annotations/qvh_raw_valid.json`: original-format filtered annotations for the {total_count} valid clips
- `annotations/tinytrace_train.json`: TinyTrace-ready training JSON
- `annotations/tinytrace_val.json`: TinyTrace-ready validation JSON
- `annotations/train_ids.txt`: train clip filenames
- `annotations/val_ids.txt`: val clip filenames
- `annotations/download_manifest.json`: creation metadata and split settings

## Split Rule

- filenames collected from the valid downloaded clips in `{source_json_name}`
- sorted, then shuffled with seed `{seed}`
- 90/10 split
- final counts:
  - train: `{train_count}`
  - val: `{val_count}`
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    source_json_path = root / args.source_json
    video_dir = root / args.video_dir
    output_dir = root / args.output_dir

    if not source_json_path.is_file():
        raise FileNotFoundError(f"Source JSON not found: {source_json_path}")
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    items = json.loads(source_json_path.read_text(encoding="utf-8"))
    existing_by_name = {path.name: path.resolve() for path in video_dir.glob("*.mp4")}

    valid_items = []
    seen_names = set()
    for item in items:
        video_name = Path(item["video"]).name
        if video_name in seen_names:
            continue
        local_path = existing_by_name.get(video_name)
        if local_path is None:
            continue
        cloned = dict(item)
        cloned["local_video_path"] = str(local_path)
        valid_items.append(cloned)
        seen_names.add(video_name)

    if not valid_items:
        raise ValueError("No valid downloaded items were found to rebuild the final dataset.")

    ordered_names = sorted(Path(item["local_video_path"]).name for item in valid_items)
    rng = random.Random(args.seed)
    rng.shuffle(ordered_names)
    train_count = int(len(ordered_names) * args.train_ratio)
    train_names = set(ordered_names[:train_count])
    val_names = set(ordered_names[train_count:])

    output_tmp = output_dir.with_name(output_dir.name + "_tmp")
    if output_tmp.exists():
        shutil.rmtree(output_tmp)
    (output_tmp / "videos" / "train").mkdir(parents=True, exist_ok=True)
    (output_tmp / "videos" / "val").mkdir(parents=True, exist_ok=True)
    (output_tmp / "annotations").mkdir(parents=True, exist_ok=True)

    raw_valid = []
    tinytrace_train = []
    tinytrace_val = []

    for item in valid_items:
        source_path = Path(item["local_video_path"])
        video_name = source_path.name
        split = "train" if video_name in train_names else "val"
        target_path = output_tmp / "videos" / split / video_name
        link_or_copy(source_path, target_path, args.link_mode)

        query = extract_query(item["conversations"][0]["value"])
        events = build_events(item["times"], item["scores"], args.max_events)
        source_runs = build_events(item["times"], item["scores"], max_events=max(len(item["times"]), 1))
        tinytrace_item = {
            "source_id": item["id"],
            "video_path": str(target_path.relative_to(output_tmp)),
            "instruction": (
                "Find the video highlights for this query and return their timestamps "
                f"and saliency scores: {query}"
            ),
            "query": query,
            "task_mode": "highlight",
            "source_saliency_points": len(item["times"]),
            "source_score_runs": len(source_runs),
            "compressed_score_runs": len(events),
            "events": events,
        }
        raw_item = dict(item)
        raw_item.pop("local_video_path", None)
        raw_valid.append(raw_item)
        if split == "train":
            tinytrace_train.append(tinytrace_item)
        else:
            tinytrace_val.append(tinytrace_item)

    annotations_dir = output_tmp / "annotations"
    (annotations_dir / "qvh_raw_valid.json").write_text(json.dumps(raw_valid, indent=2), encoding="utf-8")
    (annotations_dir / "tinytrace_train.json").write_text(json.dumps(tinytrace_train, indent=2), encoding="utf-8")
    (annotations_dir / "tinytrace_val.json").write_text(json.dumps(tinytrace_val, indent=2), encoding="utf-8")
    (annotations_dir / "train_ids.txt").write_text("\n".join(sorted(train_names)) + "\n", encoding="utf-8")
    (annotations_dir / "val_ids.txt").write_text("\n".join(sorted(val_names)) + "\n", encoding="utf-8")
    manifest = {
        "source_json": args.source_json,
        "video_dir": args.video_dir,
        "total_valid_videos": len(valid_items),
        "train_count": len(tinytrace_train),
        "val_count": len(tinytrace_val),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "max_events": args.max_events,
        "link_mode": args.link_mode,
    }
    (annotations_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_readme(
        output_tmp / "README.md",
        total_count=len(valid_items),
        train_count=len(tinytrace_train),
        val_count=len(tinytrace_val),
        source_json_name=args.source_json,
        seed=args.seed,
    )

    backup_dir = output_dir.with_name(output_dir.name + "_backup")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if output_dir.exists():
        output_dir.replace(backup_dir)
    output_tmp.replace(output_dir)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)

    print(f"Rebuilt {output_dir} with {len(valid_items)} valid clips.")
    print(f"train={len(tinytrace_train)} val={len(tinytrace_val)}")


if __name__ == "__main__":
    main()
