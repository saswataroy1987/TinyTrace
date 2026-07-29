from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace import JsonTinyTraceDataset, TinyTraceConfig, TinyTraceModel, tinytrace_collate_fn
from tinytrace.metrics import (
    QVH_NUM_BINS,
    evaluate_qvhighlights_mean_score_proxy,
    evaluate_qvhighlights_official,
    evaluate_saliency_collapse_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Phase-A-v3 TinyTrace checkpoint as direct 75-bin "
            "QVHighlights saliency predictions."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="TinyTrace checkpoint to evaluate.")
    parser.add_argument("--dataset-json", required=True, help="Prepared Phase-A-v3 split JSON.")
    parser.add_argument(
        "--official-ground-truth",
        default="",
        help=(
            "Optional official QVHighlights JSON/JSONL containing qid, "
            "relevant_clip_ids, and all three saliency_scores per relevant clip. "
            "Without this file only explicitly named mean-score proxy metrics are emitted."
        ),
    )
    parser.add_argument("--frame-cache-dir", default="", help="Optional uint8 frame cache.")
    parser.add_argument(
        "--visual-feature-cache-dir",
        default="",
        help="Optional frozen MobileCLIP FP16 feature cache.",
    )
    parser.add_argument(
        "--require-visual-feature-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail instead of decoding RGB when a frozen feature entry is missing.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("auto", "off"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-path", required=True, help="Destination JSON artifact.")
    return parser.parse_args()


def _load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        raise ValueError(f"Ground-truth file is empty: {source}")
    if stripped.startswith("["):
        payload = json.loads(text)
    else:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Ground truth must be a JSON array or JSONL stream of objects: {source}")
    return payload


def _canonical_qid(value: Any, *, context: str) -> Any:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{context} has invalid qid {value!r}")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{context} has an empty qid")
        try:
            return int(stripped)
        except ValueError:
            return stripped
    return value


def _qid_from_item(item: dict[str, Any], *, context: str) -> Any:
    for key in ("qid", "source_id", "id"):
        if key in item:
            return _canonical_qid(item[key], context=context)
    raise ValueError(f"{context} is missing qid/source_id/id")


def _select_official_ground_truth(
    items: list[dict[str, Any]], prediction_qids: list[Any]
) -> list[dict[str, Any]]:
    indexed: dict[Any, dict[str, Any]] = {}
    for index, raw_item in enumerate(items):
        qid = _qid_from_item(raw_item, context=f"official_ground_truth[{index}]")
        if qid in indexed:
            raise ValueError(f"Official ground truth contains duplicate qid {qid!r}")
        item = dict(raw_item)
        item["qid"] = qid
        indexed[qid] = item

    missing = [qid for qid in prediction_qids if qid not in indexed]
    if missing:
        raise ValueError(
            "Official ground truth is missing qids required by the evaluation split: "
            f"{missing[:12]}{' ...' if len(missing) > 12 else ''}"
        )
    # The official release normally contains more qids than the selected local
    # validation subset. Select the requested split explicitly, then let the
    # strict evaluator enforce exact qid and 75-bin agreement.
    return [indexed[qid] for qid in prediction_qids]


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 compatibility.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a dictionary payload.")
    if "config" not in payload or "model_state" not in payload:
        raise ValueError("Checkpoint is missing config or model_state.")
    return payload


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _autocast_context(device: torch.device, amp: str):
    if amp == "off" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative.")

    checkpoint = _load_checkpoint(args.checkpoint)
    config = TinyTraceConfig.from_dict(checkpoint["config"])
    if not config.phase_a_dense_saliency:
        raise ValueError("This evaluator accepts only phase_a_dense_saliency=true checkpoints.")
    if config.phase_a_bin_count != QVH_NUM_BINS or config.phase_a_bin_size_seconds != 2.0:
        raise ValueError(
            "Official QVHighlights evaluation requires exactly 75 direct 2-second bins."
        )

    device = _resolve_device(args.device)
    dataset = JsonTinyTraceDataset(
        args.dataset_json,
        config=config,
        frame_cache_dir=args.frame_cache_dir or None,
        allow_random_frames=False,
        validate_videos_on_init=True,
        strict_media_validation=True,
        visual_feature_cache_dir=args.visual_feature_cache_dir or None,
        require_visual_feature_cache=args.require_visual_feature_cache,
    )
    if not dataset:
        raise ValueError("Evaluation dataset is empty.")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=tinytrace_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # The saved state contains the exact visual tower. Avoid re-reading the
    # external MobileCLIP checkpoint during evaluation initialization.
    model = TinyTraceModel(config, load_pretrained_visual=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    predictions: list[dict[str, Any]] = []
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=device.type == "cuda")
        frame_times = batch["frame_times"].to(device, non_blocking=device.type == "cuda")
        frame_mask = batch["frame_mask"].to(device, non_blocking=device.type == "cuda")
        token_ids = batch["token_ids"].to(device, non_blocking=device.type == "cuda")
        prompt_lengths = batch["prompt_lengths"].to(
            device, non_blocking=device.type == "cuda"
        )
        patch_features = batch.get("visual_patch_features")
        if patch_features is not None:
            patch_features = patch_features.to(device, non_blocking=device.type == "cuda")

        with _autocast_context(device, args.amp):
            output = model(
                frames,
                frame_times,
                token_ids,
                frame_mask=frame_mask,
                visual_patch_features=patch_features,
                prompt_lengths=prompt_lengths,
            )
        if output.saliency_scores is None:
            raise RuntimeError("Dense Phase-A model returned no saliency scores.")

        predicted_batch = output.saliency_scores.detach().float().cpu().tolist()
        targets_batch = batch["saliency_targets"].float().tolist()
        for row, (predicted_scores, target_scores) in enumerate(
            zip(predicted_batch, targets_batch)
        ):
            qid = _canonical_qid(batch["source_id"][row], context="evaluation sample")
            predictions.append(
                {
                    "qid": qid,
                    "query": batch["query"][row],
                    "video_path": batch["video_path"][row],
                    "pred_saliency_scores": predicted_scores,
                    "qvh_mean_score_targets": target_scores,
                }
            )

    proxy_metrics = evaluate_qvhighlights_mean_score_proxy(predictions)
    diagnostics = evaluate_saliency_collapse_diagnostics(predictions)
    official_metrics: dict[str, float] = {}
    evaluation_mode = "mean_score_proxy_only"
    if args.official_ground_truth:
        official_source = _load_json_or_jsonl(args.official_ground_truth)
        selected_ground_truth = _select_official_ground_truth(
            official_source, [item["qid"] for item in predictions]
        )
        official_metrics = evaluate_qvhighlights_official(
            predictions, selected_ground_truth, num_bins=config.phase_a_bin_count
        )
        evaluation_mode = "official_three_annotator_and_mean_score_proxy"

    return {
        "protocol": "TinyTrace Phase-A-v3 direct 75x2-second saliency",
        "evaluation_mode": evaluation_mode,
        "official_metrics": official_metrics,
        "mean_score_proxy_metrics": proxy_metrics,
        "collapse_diagnostics": diagnostics,
        "sample_count": len(predictions),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_json": str(Path(args.dataset_json).resolve()),
        "official_ground_truth": (
            str(Path(args.official_ground_truth).resolve()) if args.official_ground_truth else None
        ),
        "warning": (
            None
            if args.official_ground_truth
            else "Official QVHighlights metrics require the original three-annotator labels; "
            "the local averaged labels support proxy metrics only."
        ),
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    artifact = evaluate(args)
    save_path = Path(args.save_path)
    if save_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation artifact: {save_path}"
        )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in artifact.items() if key != "predictions"}, indent=2))


if __name__ == "__main__":
    main()
