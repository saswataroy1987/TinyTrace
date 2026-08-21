"""Run real/shuffled/zero inference audit for a trained B1, B2, or C1 bridge."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_phase_b_v3_direct_mobileclip import _digest
from train_stage2_bridge_ablation import _audit_evaluation, _build, _zero_control
from tinytrace.phase_b_v2 import ActivityNetV2Dataset, Stage0Config
from tinytrace.phase_b_v3 import DirectMobileCLIPCaptionConfig


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("b1", "b2", "c1"), required=True)
    parser.add_argument("--stage0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("stage2_audit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable.")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "stage2_bridge_ablation" or payload.get("variant") != args.variant or payload.get("manifest_sha256") != _digest(args.manifest):
        raise ValueError("Checkpoint does not match this bridge-ablation audit.")
    config = DirectMobileCLIPCaptionConfig(**payload["model_config"])
    model = _build(config, args.variant)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    validation = ActivityNetV2Dataset(args.manifest, cache_root=args.cache_root, config=Stage0Config.from_json(args.stage0_config), split="val")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = _audit_evaluation(model, validation, args.audit_root, args.output_root, int(payload["epoch"]), device)
    _zero_control(model, validation, args.audit_root, args.output_root, int(payload["epoch"]), device)
    report = {"variant": args.variant, "checkpoint": str(args.checkpoint.resolve()), "epoch": payload["epoch"], "metrics": metrics, "visual_tokens_per_event": config.max_event_frames * config.patch_tokens if args.variant == "c1" else config.max_event_frames, "flan_visual_prefix_max_tokens": config.max_event_frames * config.patch_tokens if args.variant == "c1" else config.max_event_frames, "flan_instruction_tokens": len(model.tokenizer(config.instruction)["input_ids"]), "flan_config_n_positions": getattr(model.language_model.config, "n_positions", None), "audit_seconds": time.perf_counter() - started, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None}
    path = args.output_root / f"epoch-{int(payload['epoch']):04d}" / "audit_runtime.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
