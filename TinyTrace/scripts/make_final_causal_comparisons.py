"""Create an easy side-by-side ground-truth versus generated event report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace.phase_b_final_causal import parse_event_sequence


def _events(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No parseable `<EVENT> ... </EVENT>` block was generated."
    lines = []
    for number, event in enumerate(rows, 1):
        if event["valid_time_tokens"]:
            timing = f"{float(event['start_normalized']):.2f} to {float(event['end_normalized']):.2f}"
        else:
            timing = f"INVALID time tokens: {event['raw_start_token']} to {event['raw_end_token']}"
        lines.append(f"{number}. `{timing}`: {event['caption']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or args.audit_directory / "CAPTION_COMPARISONS.md"
    rows = []
    for path in sorted(args.audit_directory.glob("video_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(value)
    if not rows:
        raise ValueError(f"No video_*.json files in {args.audit_directory}")
    lines = ["# Ground Truth vs Generated Event Sequences", "", "This is a post-training read-only rendering of the saved audit outputs. `INVALID time tokens` are an important failure: the model emitted a T5 sentinel token instead of the required `<T000>` to `<T100>` boundary token.", ""]
    for row in rows:
        lines.extend([f"## {row['video_id']}", "", "### Ground Truth", _events(parse_event_sequence(str(row["reference_sequence"]))), "", "### Generated From Real Video Features", _events(parse_event_sequence(str(row["real"]["raw_sequence"]))), "", "### Generated From Shuffled Video Features", _events(parse_event_sequence(str(row["shuffled"]["raw_sequence"]))), ""])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "videos": len(rows)}))


if __name__ == "__main__":
    main()
