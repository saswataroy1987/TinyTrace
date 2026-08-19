from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_v2.cache import legacy_v1_cache_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a portable video_id-to-cache mapping on the original V1 cache PC. "
            "This is read-only and does not generate or modify feature files."
        )
    )
    parser.add_argument("--verified-json", type=Path, action="append", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tinytrace_activitynet_phase_b_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_video_path(annotation_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    for candidate in (
        annotation_path.parent / path,
        annotation_path.parent.parent / path,
        annotation_path.parent.parent.parent / path,
    ):
        if candidate.is_file():
            return candidate.resolve()
    return (annotation_path.parent / path).resolve()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite cache mapping: {args.output}")
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    cache_root = args.cache_root.resolve()
    entries: dict[str, dict[str, object]] = {}
    for annotation_path in args.verified_json:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Verified V1 annotation must be a list: {annotation_path}")
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"Verified V1 annotation contains a non-object: {annotation_path}")
            video_id = item.get("video_id")
            video_path = item.get("video_path")
            if not isinstance(video_id, str) or not isinstance(video_path, str):
                raise ValueError(f"Verified item lacks video_id/video_path: {item!r}")
            if video_id in entries:
                raise ValueError(f"Duplicate video_id across verified annotations: {video_id}")
            resolved_video = _resolve_video_path(annotation_path.resolve(), video_path)
            if not resolved_video.is_file():
                raise FileNotFoundError(
                    f"Cannot reproduce V1 cache hash because source video is unavailable: {resolved_video}"
                )
            num_frames = int(item.get("num_frames", config["max_frames"]))
            filename = legacy_v1_cache_filename(
                video_path=resolved_video,
                num_frames=num_frames,
                image_size=int(config["image_size"]),
                mobileclip_model_name=str(config["mobileclip_model_name"]),
                mobileclip_checkpoint_sha256=str(config["mobileclip_checkpoint_sha256"]),
                mobileclip_apply_normalization=bool(config.get("mobileclip_apply_normalization", False)),
            )
            feature_path = cache_root / filename
            entries[video_id] = {
                "video_id": video_id,
                "visual_feature_path": filename,
                "cache_exists": feature_path.is_file(),
                "source_video_path_used_for_v1_hash": str(resolved_video),
            }
    output = {
        "schema_version": "tinytrace.phase-b-v2.cache-map.v1",
        "cache_read_only": True,
        "entries": [entries[key] for key in sorted(entries)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    found = sum(bool(item["cache_exists"]) for item in entries.values())
    print(f"Exported {len(entries)} mappings; existing cache entries={found}; missing={len(entries)-found}")


if __name__ == "__main__":
    main()
