from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace import JsonTinyTraceDataset, TinyTraceConfig, TinyTraceModel, visual_feature_diversity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a trained Priority 4 representation profile.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-json", required=True)
    parser.add_argument("--frame-cache-dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence.")
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be an object.")
    state = payload.get("model_state", payload)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain model_state.")
    return state


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.samples < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("samples/repeats must be positive and warmup must be non-negative.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    config = TinyTraceConfig.from_json(args.config)
    dataset = JsonTinyTraceDataset(
        args.dataset_json,
        config,
        frame_cache_dir=args.frame_cache_dir or None,
        allow_random_frames=False,
    )
    model = TinyTraceModel(config).to(device)
    model.load_state_dict(_load_state(Path(args.checkpoint)))
    model.eval()
    sample_count = min(args.samples, len(dataset))
    if sample_count == 0:
        raise ValueError("Benchmark dataset is empty.")

    records: list[dict[str, float | int]] = []
    diversity_records: list[dict[str, float | int | None]] = []
    for iteration in range(args.warmup + args.repeats):
        for sample_index in range(sample_count):
            _sync(device)
            started = time.perf_counter()
            sample = dataset[sample_index]
            _sync(device)
            decode_done = time.perf_counter()
            frames = sample["frames"].unsqueeze(0).to(device)
            frame_times = sample["frame_times"].unsqueeze(0).to(device)
            frame_mask = torch.ones(1, frames.size(1), dtype=torch.bool, device=device)
            prompt = sample["token_ids"][: sample["prompt_length"]].unsqueeze(0).to(device)
            patch_features = model.visual_encoder.extract_patch_features(frames)
            _sync(device)
            mobileclip_done = time.perf_counter()
            compressed = model.visual_encoder.compress_patch_features(patch_features)
            _sync(device)
            compression_done = time.perf_counter()
            generated, metadata = model.generate(
                frames,
                frame_times,
                prompt,
                config.max_generated_tokens,
                frame_mask=frame_mask,
                visual_patch_features=patch_features,
                return_metadata=True,
            )
            _sync(device)
            finished = time.perf_counter()
            if iteration >= args.warmup:
                records.append(
                    {
                        "decoded_frames": int(frames.size(1)),
                        "flattened_mobileclip_frame_batch": int(frames.size(0) * frames.size(1)),
                        "visual_time_prefix_tokens": int(
                            frames.size(1)
                            * (config.compressed_visual_tokens + config.time_tokens_per_frame)
                        ),
                        "generated_tokens": int(generated.size(1) - prompt.size(1)),
                        "decode_preprocess_ms": (decode_done - started) * 1000,
                        "mobileclip_ms": (mobileclip_done - decode_done) * 1000,
                        "compression_ms": (compression_done - mobileclip_done) * 1000,
                        "generation_ms": (finished - compression_done) * 1000,
                        "end_to_end_ms": (finished - started) * 1000,
                    }
                )
                diversity_records.append(
                    visual_feature_diversity(patch_features, compressed, frame_mask)
                )

    latency_values = [float(record["end_to_end_ms"]) for record in records]
    generated_tokens = sum(int(record["generated_tokens"]) for record in records)
    generation_seconds = sum(float(record["generation_ms"]) for record in records) / 1000
    similarity_keys = (
        "mean_frame_patch_summary_cosine_similarity",
        "mean_compressed_slot_cosine_similarity",
    )
    diversity = {}
    for key in similarity_keys:
        values = [float(item[key]) for item in diversity_records if item[key] is not None]
        diversity[key] = statistics.fmean(values) if values else None
    return {
        "profile": {
            "max_frames": config.max_frames,
            "max_caption_tokens": config.max_caption_tokens,
            "max_generated_tokens": config.max_generated_tokens,
            "visual_time_prefix_tokens": config.max_frames
            * (config.compressed_visual_tokens + config.time_tokens_per_frame),
        },
        "measurement": {
            "device": str(device),
            "samples": sample_count,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.repeats,
            "median_end_to_end_ms": statistics.median(latency_values),
            "p90_end_to_end_ms": _percentile(latency_values, 0.9),
            "mean_decode_preprocess_ms": statistics.fmean(float(item["decode_preprocess_ms"]) for item in records),
            "mean_mobileclip_ms": statistics.fmean(float(item["mobileclip_ms"]) for item in records),
            "mean_compression_ms": statistics.fmean(float(item["compression_ms"]) for item in records),
            "mean_generation_ms": statistics.fmean(float(item["generation_ms"]) for item in records),
            "generated_tokens_per_second": generated_tokens / generation_seconds if generation_seconds else 0.0,
            "peak_accelerator_memory_allocated": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "peak_accelerator_memory_reserved": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
        },
        "feature_diversity": diversity,
        "raw_measurements": records,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    report = benchmark(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
