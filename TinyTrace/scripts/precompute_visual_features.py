from __future__ import annotations

import argparse
import copy
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace import JsonTinyTraceDataset, TinyTraceConfig, TinyTraceModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute frozen MobileCLIP spatial features for one or more "
            "TinyTrace JSON datasets."
        )
    )
    parser.add_argument("--config", required=True, help="TinyTrace model JSON config.")
    parser.add_argument(
        "--dataset-json",
        action="append",
        required=True,
        help="Dataset JSON to process. Repeat this option for train/validation datasets.",
    )
    parser.add_argument(
        "--frame-cache-dir",
        default="",
        help="Optional decoded-frame cache directory.",
    )
    parser.add_argument(
        "--visual-feature-cache-dir",
        required=True,
        help="Destination for frozen MobileCLIP feature cache entries.",
    )
    parser.add_argument("--device", default="cuda", help="Extraction device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--amp",
        choices=("auto", "off"),
        default="auto",
        help="Use safe CUDA autocast when available, or disable autocast.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute entries even when an existing cache entry is valid.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N visited samples.",
    )
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid extraction device {value!r}.") from exc
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for feature precomputation but is unavailable.")
        device_index = device.index
        if device_index is not None and not 0 <= device_index < torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device_index} is unavailable; "
                f"detected {torch.cuda.device_count()} CUDA device(s)."
            )
    return device


def _autocast_context(device: torch.device, amp: str):
    if amp == "off" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _build_dataset(
    dataset_json: str,
    config: TinyTraceConfig,
    frame_cache_dir: str,
    visual_feature_cache_dir: str,
) -> JsonTinyTraceDataset:
    dataset = JsonTinyTraceDataset(
        annotation_path=dataset_json,
        config=config,
        frame_cache_dir=frame_cache_dir or None,
        allow_random_frames=False,
        validate_videos_on_init=True,
        strict_media_validation=True,
        visual_feature_cache_dir=visual_feature_cache_dir,
        require_visual_feature_cache=False,
    )
    missing_video_paths = [
        str(item.get("source_id", index))
        for index, item in enumerate(dataset.items)
        if not item.get("video_path")
    ]
    if missing_video_paths:
        preview = ", ".join(missing_video_paths[:8])
        raise ValueError(
            "Frozen visual-feature precomputation requires video_path on every sample; "
            f"missing for {len(missing_video_paths)} sample(s): {preview}"
        )
    return dataset


@torch.inference_mode()
def precompute(args: argparse.Namespace) -> dict[str, int | float]:
    if args.progress_every < 1:
        raise ValueError("progress_every must be a positive integer.")
    device = _resolve_device(args.device)
    config = TinyTraceConfig.from_json(args.config)
    if not config.freeze_visual_encoder:
        raise ValueError(
            "Visual feature caching is valid only when freeze_visual_encoder=true. "
            "A trainable MobileCLIP tower would make cached features stale."
        )

    datasets = [
        _build_dataset(
            dataset_json=dataset_json,
            config=config,
            frame_cache_dir=args.frame_cache_dir,
            visual_feature_cache_dir=args.visual_feature_cache_dir,
        )
        for dataset_json in args.dataset_json
    ]
    total = sum(len(dataset) for dataset in datasets)
    if total < 1:
        raise ValueError("The requested datasets contain no samples.")

    # TinyTraceModel performs the pinned checkpoint existence and SHA-256
    # verification while constructing the official MobileCLIP-S0 backbone.
    model = TinyTraceModel(config).to(device)
    model.set_visual_encoder_trainable(False)
    model.eval()

    visited = 0
    written = 0
    skipped = 0
    started = time.perf_counter()
    for dataset_index, dataset in enumerate(datasets):
        # A shallow copy retains the strictly validated/resolved item list but
        # bypasses feature-cache reads when --overwrite requests fresh RGB
        # extraction. Frame-cache behavior remains unchanged.
        decode_dataset = copy.copy(dataset)
        decode_dataset.visual_feature_cache_dir = None
        decode_dataset.require_visual_feature_cache = False

        for item_index in range(len(dataset)):
            visited += 1
            cache_path = dataset.visual_feature_cache_path(item_index)
            sample = None
            if cache_path.is_file() and not args.overwrite:
                # Going through __getitem__ invokes the canonical cache reader,
                # including all format, shape, finiteness, and alignment checks.
                sample = dataset[item_index]
                if "visual_patch_features" in sample:
                    skipped += 1
                    if visited % args.progress_every == 0 or visited == total:
                        elapsed = time.perf_counter() - started
                        print(
                            f"[{visited}/{total}] valid cache; skipped "
                            f"(written={written}, skipped={skipped}, {elapsed:.1f}s)"
                        )
                    continue

            if sample is None or "visual_patch_features" in sample:
                sample = decode_dataset[item_index]
            frames = sample["frames"]
            frame_times = sample["frame_times"]
            if frames.ndim != 4 or frames.size(0) != frame_times.numel():
                raise ValueError(
                    f"Decoded frame/time alignment failed for dataset {dataset_index + 1}, "
                    f"sample {item_index}."
                )

            with _autocast_context(device, args.amp):
                patch_features = model.visual_encoder.extract_patch_features(
                    frames.unsqueeze(0).to(device)
                ).squeeze(0)
            if not torch.isfinite(patch_features).all():
                source_id = sample.get("source_id", item_index)
                raise ValueError(f"MobileCLIP produced non-finite features for {source_id}.")
            dataset.write_visual_feature_cache(item_index, patch_features, frame_times)
            written += 1

            if visited % args.progress_every == 0 or visited == total:
                elapsed = time.perf_counter() - started
                print(
                    f"[{visited}/{total}] cached {cache_path.name} "
                    f"(written={written}, skipped={skipped}, {elapsed:.1f}s)"
                )

    elapsed = time.perf_counter() - started
    return {
        "datasets": len(datasets),
        "samples": total,
        "written": written,
        "skipped_valid": skipped,
        "elapsed_seconds": elapsed,
    }


def main() -> None:
    args = parse_args()
    summary = precompute(args)
    print(
        "Visual feature precomputation complete: "
        f"datasets={summary['datasets']} samples={summary['samples']} "
        f"written={summary['written']} skipped_valid={summary['skipped_valid']} "
        f"elapsed={summary['elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
