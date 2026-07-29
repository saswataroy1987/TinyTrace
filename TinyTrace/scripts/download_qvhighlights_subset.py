from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a valid subset of QVHighlights clips for TinyTrace."
    )
    parser.add_argument(
        "--source-json",
        type=str,
        default="dataset/mt_fmt-8k.json",
        help="Path to the QVHighlights-style annotation JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/qvhighlights/videos/train",
        help="Where the downloaded mp4 clips will be stored.",
    )
    parser.add_argument(
        "--subset-json",
        type=str,
        default="dataset/qvhighlights/mt_fmt-50-valid.json",
        help="Where to save the matched annotation subset.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=50,
        help="Target total number of valid clips to have in the output folder.",
    )
    parser.add_argument(
        "--max-scan",
        type=int,
        default=2000,
        help="How many source rows to scan before stopping.",
    )
    parser.add_argument(
        "--check-timeout",
        type=int,
        default=20,
        help="Timeout in seconds for availability checks.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each clip download.",
    )
    parser.add_argument(
        "--yt-dlp-bin",
        type=str,
        default="TinyTrace/.venv/bin/python -m yt_dlp",
        help="Command used to run yt-dlp.",
    )
    return parser.parse_args()


def parse_clip_name(path_str: str) -> tuple[str, str, str, str]:
    stem = Path(path_str).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid clip path format: {path_str}")
    youtube_id = "_".join(parts[:-2])
    start = parts[-2]
    end = parts[-1]
    return youtube_id, start, end, stem


def yt_dlp_command(command: str) -> list[str]:
    return command.split()


def check_video_available(root: Path, yt_dlp_bin: str, url: str, timeout: int) -> bool:
    try:
        result = subprocess.run(
            yt_dlp_command(yt_dlp_bin) + ["--skip-download", "--quiet", url],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


def download_clip(
    root: Path,
    yt_dlp_bin: str,
    url: str,
    start: str,
    end: str,
    output_file: Path,
    timeout: int,
) -> bool:
    try:
        result = subprocess.run(
            yt_dlp_command(yt_dlp_bin)
            + [
                url,
                "-f",
                "bv*[height<=360]+ba/b[height<=360]/b",
                "--download-sections",
                f"*{start}-{end}",
                "--force-keyframes-at-cuts",
                "--merge-output-format",
                "mp4",
                "-o",
                str(output_file),
            ],
            cwd=root,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and output_file.exists()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    source_json = root / args.source_json
    output_dir = root / args.output_dir
    subset_json = root / args.subset_json
    yt_dlp_bin = args.yt_dlp_bin

    if not source_json.exists():
        raise FileNotFoundError(f"Source JSON not found: {source_json}")
    yt_parts = yt_dlp_command(yt_dlp_bin)
    runner = Path(yt_parts[0])
    if not (root / runner).exists() and not runner.exists():
        raise FileNotFoundError(f"yt-dlp runner not found: {runner}")

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_json.parent.mkdir(parents=True, exist_ok=True)

    items = json.loads(source_json.read_text())
    existing_mp4_names = {path.name for path in output_dir.glob("*.mp4")}
    selected = []

    print(f"Existing completed mp4 files: {len(existing_mp4_names)}")
    if len(existing_mp4_names) >= args.target_count:
        print(f"Target already reached: {len(existing_mp4_names)} >= {args.target_count}")
        all_valid_items = []
        seen = set()
        for item in items:
            name = Path(item["video"]).name
            if name in existing_mp4_names and name not in seen:
                all_valid_items.append(item)
                seen.add(name)
        subset_json.write_text(json.dumps(all_valid_items, indent=2))
        print(f"Video folder: {output_dir}")
        print(f"Subset annotations: {subset_json}")
        return

    for item in items[: args.max_scan]:
        current_total = len(existing_mp4_names) + len(selected)
        if current_total >= args.target_count:
            break

        clip_path = item["video"]
        youtube_id, start, end, clip_name = parse_clip_name(clip_path)
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        output_file = output_dir / f"{clip_name}.mp4"

        if output_file.name in existing_mp4_names:
            print(f"[keep] {clip_name}")
            selected.append(item)
            continue

        if not check_video_available(root, yt_dlp_bin, url, args.check_timeout):
            print(f"[skip unavailable] {clip_name}")
            continue

        print(f"[download] {clip_name}")
        ok = download_clip(
            root=root,
            yt_dlp_bin=yt_dlp_bin,
            url=url,
            start=start,
            end=end,
            output_file=output_file,
            timeout=args.download_timeout,
        )

        if ok:
            selected.append(item)
            print(f"[saved total {len(existing_mp4_names) + len(selected)}/{args.target_count}] {output_file}")
        else:
            print(f"[failed] {clip_name}")

    final_mp4_names = {path.name for path in output_dir.glob("*.mp4")}
    all_valid_items = []
    seen = set()
    for item in items:
        name = Path(item["video"]).name
        if name in final_mp4_names and name not in seen:
            all_valid_items.append(item)
            seen.add(name)

    subset_json.write_text(json.dumps(all_valid_items, indent=2))
    print()
    print(f"Completed valid mp4 files now present: {len(final_mp4_names)}")
    print(f"Video folder: {output_dir}")
    print(f"Subset annotations: {subset_json}")


if __name__ == "__main__":
    main()
