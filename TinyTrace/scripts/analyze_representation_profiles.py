from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace import (
    CAPTION_TOKEN_LADDER,
    FRAME_COUNT_LADDER,
    TinyTraceConfig,
    aggregate_caption_budget,
    caption_budget_metadata,
    temporal_coverage_report,
)
from tinytrace.tokenizers import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Priority 4 temporal coverage and caption budgets without media acquisition."
    )
    parser.add_argument("--duration", type=float, action="append", default=[])
    parser.add_argument("--dataset-json", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def _duration_from_item(item: dict) -> tuple[float | None, str | None]:
    if isinstance(item.get("duration"), (int, float)):
        return float(item["duration"]), "duration"
    frame_times = item.get("frame_times")
    if isinstance(frame_times, list) and frame_times:
        return float(max(frame_times)) + 0.25, "frame_times"
    event_ends = [
        float(value)
        for event in item.get("events", [])
        for value in event.get("timestamp", [])
        if isinstance(value, (int, float))
    ]
    if event_ends:
        return max(event_ends) + 0.25, "event_extent_proxy"
    return None, None


def build_report(durations: list[float], dataset_items: list[dict]) -> dict[str, object]:
    inferred = []
    for index, item in enumerate(dataset_items):
        duration, source = _duration_from_item(item)
        if duration is not None and duration > 0:
            inferred.append({"sample_index": index, "duration_seconds": duration, "source": source})
            durations.append(duration)

    caption_profiles = {}
    events_by_sample = [item.get("events", []) for item in dataset_items]
    for budget in CAPTION_TOKEN_LADDER:
        config = TinyTraceConfig(
            max_caption_tokens=budget,
            max_generated_tokens=max(128, TinyTraceConfig(max_caption_tokens=budget, max_generated_tokens=512).required_generation_token_budget),
        )
        tokenizer = CharTokenizer(config.text_vocab_size)
        reports = [caption_budget_metadata(events, config, tokenizer) for events in events_by_sample]
        caption_profiles[str(budget)] = {
            "required_generation_tokens": config.required_generation_token_budget,
            **aggregate_caption_budget(reports),
        }

    return {
        "frame_count_ladder": list(FRAME_COUNT_LADDER),
        "caption_token_ladder": list(CAPTION_TOKEN_LADDER),
        "duration_sources": inferred,
        "temporal_coverage": [
            {"duration_seconds": duration, "profiles": temporal_coverage_report(duration)}
            for duration in durations
        ],
        "caption_profiles": caption_profiles,
        "notes": {
            "event_extent_proxy": "A proxy is used only when explicit duration/frame times are absent; it is not measured video duration.",
            "acquisition_performed": False,
        },
    }


def main() -> None:
    args = parse_args()
    items = []
    if args.dataset_json:
        payload = json.loads(Path(args.dataset_json).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise ValueError("dataset JSON must be a list of objects.")
        items = payload
    if not args.duration and not items:
        raise ValueError("Provide at least one --duration or --dataset-json.")
    report = build_report(list(args.duration), items)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
