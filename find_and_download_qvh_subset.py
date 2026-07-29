import json
import subprocess
from pathlib import Path

ROOT = Path("/home/vikaspal/Desktop/Traceall")
DATASET = ROOT / "dataset/mt_fmt-8k.json"
OUT_DIR = ROOT / "qvhighlights/videos/train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COUNT = 6
MAX_SCAN = 1200
CHECK_TIMEOUT = 20
DOWNLOAD_TIMEOUT = 300


def parse_clip(path_str: str):
    name = Path(path_str).stem
    parts = name.split("_")
    youtube_id = "_".join(parts[:-2])
    start = parts[-2]
    end = parts[-1]
    return youtube_id, start, end, name


def main():
    data = json.loads(DATASET.read_text())
    selected = []

    for item in data[:MAX_SCAN]:
        clip_path = item["video"]
        youtube_id, start, end, clip_name = parse_clip(clip_path)
        url = f"https://www.youtube.com/watch?v={youtube_id}"

        try:
            check = subprocess.run(
                ["TinyTrace/.venv/bin/yt-dlp", "--skip-download", "--quiet", url],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=CHECK_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"Skipping timeout: {clip_name}")
            continue
        if check.returncode != 0:
            print(f"Skipping unavailable: {clip_name}")
            continue

        out_file = OUT_DIR / f"{clip_name}.mp4"
        if out_file.exists():
            print(f"Already exists: {clip_name}")
            selected.append(item)
            if len(selected) >= TARGET_COUNT:
                break
            continue

        print(f"Downloading {clip_name}")
        try:
            dl = subprocess.run(
                [
                "TinyTrace/.venv/bin/yt-dlp",
                url,
                "-f",
                "bv*[height<=360]+ba/b[height<=360]/b",
                "--download-sections",
                f"*{start}-{end}",
                "--force-keyframes-at-cuts",
                    "--merge-output-format",
                    "mp4",
                    "-o",
                    str(out_file),
                ],
                cwd=ROOT,
                timeout=DOWNLOAD_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"Download timeout: {clip_name}")
            continue
        if dl.returncode == 0 and out_file.exists():
            selected.append(item)
            print(f"Saved {len(selected)}/{TARGET_COUNT}: {out_file}")
        else:
            print(f"Failed: {clip_name}")

        if len(selected) >= TARGET_COUNT:
            break

    subset_path = ROOT / "dataset/mt_fmt-6-working.json"
    subset_path.write_text(json.dumps(selected))
    print(f"\nDone. Saved subset annotations to {subset_path}")
    print(f"Downloaded clips: {len(selected)}")


if __name__ == "__main__":
    main()
