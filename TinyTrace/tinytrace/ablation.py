from __future__ import annotations

import json
from pathlib import Path

from .config import TinyTraceConfig
from .representation import CAPTION_TOKEN_LADDER, FRAME_COUNT_LADDER


def model_config_differences(
    baseline: TinyTraceConfig,
    candidate: TinyTraceConfig,
) -> dict[str, tuple[object, object]]:
    baseline_values = baseline.to_dict()
    candidate_values = candidate.to_dict()
    return {
        name: (baseline_values[name], candidate_values[name])
        for name in baseline_values
        if baseline_values[name] != candidate_values[name]
    }


def validate_representation_ablation(
    baseline: TinyTraceConfig,
    candidate: TinyTraceConfig,
    kind: str,
) -> dict[str, tuple[object, object]]:
    """Enforce the single-independent-variable rule for Priority 4."""
    differences = model_config_differences(baseline, candidate)
    if kind == "frame":
        if set(differences) != {"max_frames"}:
            raise ValueError(
                "A frame ablation must change only max_frames; changed fields: "
                f"{', '.join(sorted(differences)) or 'none'}."
            )
        baseline_index = FRAME_COUNT_LADDER.index(baseline.max_frames)
        if baseline_index + 1 >= len(FRAME_COUNT_LADDER):
            raise ValueError("The frame baseline is already the final supported ladder value.")
        expected = FRAME_COUNT_LADDER[baseline_index + 1]
        if candidate.max_frames != expected:
            raise ValueError(
                f"Frame ablations must be sequential: {baseline.max_frames} must be compared with {expected}."
            )
    elif kind == "caption":
        if baseline.max_caption_tokens != CAPTION_TOKEN_LADDER[0]:
            raise ValueError("Caption candidates must use the 20-token reference baseline.")
        if candidate.max_caption_tokens not in CAPTION_TOKEN_LADDER[1:]:
            raise ValueError("Caption candidate must use 48 or 64 tokens.")
        allowed = {"max_caption_tokens", "max_generated_tokens"}
        if not differences or not set(differences).issubset(allowed):
            raise ValueError(
                "A caption ablation may change only caption length and its dependent generation budget; "
                f"changed fields: {', '.join(sorted(differences)) or 'none'}."
            )
        if candidate.max_generated_tokens != candidate.required_generation_token_budget:
            raise ValueError(
                "Caption candidates must use the minimally required generation token budget."
            )
    else:
        raise ValueError("kind must be either 'frame' or 'caption'.")
    return differences


def summarize_run_artifacts(run_dir: str | Path) -> dict[str, object]:
    """Collect comparable quality/training/system fields from one completed run."""
    run_path = Path(run_dir)
    summary_path = run_path / "run_summary.json"
    history_path = run_path / "history.json"
    benchmark_path = run_path / "representation_benchmark.json"
    missing = [
        path.name
        for path in (summary_path, history_path, benchmark_path)
        if not path.is_file()
    ]
    result: dict[str, object] = {
        "run_dir": str(run_path),
        "complete_for_decision": not missing,
        "missing_artifacts": missing,
    }
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result.update(
            {
                "run_id": summary.get("run_id"),
                "status": summary.get("status"),
                "quality_metrics": summary.get("final_structured_metrics"),
                "generation_diagnostics": summary.get("final_generation_diagnostics"),
            }
        )
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if history:
            final = history[-1]
            train = final.get("train") or {}
            validation = final.get("validation") or {}
            result["training"] = {
                "elapsed_seconds": train.get("elapsed_seconds"),
                "frames_per_second": train.get("frames_per_second"),
                "peak_accelerator_memory_allocated": train.get(
                    "peak_accelerator_memory_allocated"
                ),
                "caption_budget": train.get("caption_budget"),
                "validation_caption_budget": validation.get("caption_budget"),
            }
    if benchmark_path.is_file():
        result["inference"] = json.loads(benchmark_path.read_text(encoding="utf-8"))
    return result


def decide_quality_efficiency_tradeoff(
    baseline: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, str]:
    """Require human-declared thresholds; never auto-adopt a representation."""
    complete = bool(baseline.get("complete_for_decision")) and bool(
        candidate.get("complete_for_decision")
    )
    return {
        "status": "awaiting_review" if complete else "incomplete",
        "decision": "not_automatically_selected",
        "reason": (
            "Artifacts are complete; compare predefined quality and efficiency guardrails."
            if complete
            else "Quality, training, and inference artifacts are required before a decision."
        ),
    }
