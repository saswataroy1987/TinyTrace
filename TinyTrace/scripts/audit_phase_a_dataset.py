from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the pre-Phase-A and dense Phase-A QVHighlights annotations so "
            "the next retraining cycle starts from a verified dataset."
        )
    )
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("annotations/tinytrace_phase_a_v3_train.json"),
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=Path("annotations/tinytrace_phase_a_v3_val.json"),
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=3,
        help="How many high-signal samples to print from each split.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return payload


def _raw_bin_coverage(item: dict) -> int:
    times = item.get("times", [])
    bins = set()
    for row in times:
        if not isinstance(row, list) or len(row) != 1:
            continue
        value = row[0]
        if isinstance(value, (int, float)):
            bins.add(int(float(value) / 2.0))
    return len(bins)


def _dense_stats(items: list[dict]) -> dict[str, float]:
    positive_threshold = 3.0
    positive_bins = 0
    total_bins = 0
    max_scores: list[float] = []
    positive_runs: list[int] = []
    nonzero_bins: list[int] = []

    for item in items:
        scores = item["dense_saliency_scores"]
        total_bins += len(scores)
        positive_bins += sum(1 for score in scores if score >= positive_threshold)
        nonzero_bins.append(sum(1 for score in scores if score > 0.0))
        max_scores.append(max(scores))

        runs = 0
        in_run = False
        for score in scores:
            if score >= positive_threshold and not in_run:
                runs += 1
                in_run = True
            elif score < positive_threshold:
                in_run = False
        positive_runs.append(runs)

    return {
        "items": float(len(items)),
        "positive_bin_ratio": positive_bins / total_bins if total_bins else 0.0,
        "mean_nonzero_bins": statistics.mean(nonzero_bins) if nonzero_bins else 0.0,
        "mean_max_score": statistics.mean(max_scores) if max_scores else 0.0,
        "mean_positive_runs": statistics.mean(positive_runs) if positive_runs else 0.0,
    }


def _print_split(name: str, items: list[dict], show_samples: int) -> None:
    stats = _dense_stats(items)
    print(f"\n[{name}]")
    print(f"items: {int(stats['items'])}")
    print(f"positive_bin_ratio@>=3.0: {stats['positive_bin_ratio']:.6f}")
    print(f"mean_nonzero_bins: {stats['mean_nonzero_bins']:.2f}")
    print(f"mean_max_score: {stats['mean_max_score']:.3f}")
    print(f"mean_positive_runs: {stats['mean_positive_runs']:.2f}")

    ranked = sorted(
        items,
        key=lambda item: (
            max(item["dense_saliency_scores"]),
            sum(1 for score in item["dense_saliency_scores"] if score >= 3.0),
        ),
        reverse=True,
    )
    for item in ranked[:show_samples]:
        scores = item["dense_saliency_scores"]
        top_index = max(range(len(scores)), key=scores.__getitem__)
        print(
            f"- source_id={item['source_id']} top_bin={top_index} "
            f"top_window=[{2*top_index},{2*(top_index+1)}) "
            f"top_score={scores[top_index]:.2f} query={item['query']!r}"
        )


def main() -> None:
    args = parse_args()
    train_items = _load_json(args.train_json)
    val_items = _load_json(args.val_json)

    if args.raw_json is not None:
        raw_items = _load_json(args.raw_json)
        raw_coverage = [_raw_bin_coverage(item) for item in raw_items]
        print("[raw]")
        print(f"items: {len(raw_items)}")
        print(f"mean_covered_bins: {statistics.mean(raw_coverage):.2f}")
        print(f"max_covered_bins: {max(raw_coverage)}")

    _print_split("train", train_items, args.show_samples)
    _print_split("val", val_items, args.show_samples)


if __name__ == "__main__":
    main()
