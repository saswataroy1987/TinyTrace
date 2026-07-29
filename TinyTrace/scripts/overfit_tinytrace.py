from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tinytrace import (
    SyntheticTinyTraceDataset,
    TinyTraceConfig,
    TinyTraceModel,
    decode_event_sequence,
    tinytrace_collate_fn,
)
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the required TinyTrace synthetic overfit gate.")
    parser.add_argument("--config", default="configs/tinytrace_baseline.json")
    parser.add_argument("--output-dir", default="outputs-synthetic-overfit")
    parser.add_argument("--samples", type=int, default=4, choices=range(4, 9))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--success-loss", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    config = TinyTraceConfig.from_json(args.config)
    dataset = SyntheticTinyTraceDataset(config, size=args.samples, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.samples,
        shuffle=False,
        collate_fn=tinytrace_collate_fn,
    )
    batch = next(iter(loader))

    model = TinyTraceModel(config).to(device)
    frames = batch["frames"].to(device)
    frame_times = batch["frame_times"].to(device)
    frame_mask = batch["frame_mask"].to(device)
    token_ids = batch["token_ids"].to(device)
    label_types = batch["label_types"].to(device)

    model.eval()
    with torch.no_grad():
        # Cache only the frozen MobileCLIP output. Slot compression remains in
        # the trainable graph and is recomputed on every optimization step.
        visual_patch_features = model.visual_encoder.extract_patch_features(frames).detach()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=0.0,
    )

    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            frames,
            frame_times,
            token_ids,
            labels=token_ids,
            label_types=label_types,
            frame_mask=frame_mask,
            visual_patch_features=visual_patch_features,
        )
        if output.loss is None:
            raise RuntimeError("Overfit batch produced no training loss.")
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            max_norm=1.0,
        )
        optimizer.step()

        loss = float(output.loss.detach().cpu())
        history.append({"epoch": epoch, "loss": loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch} loss={loss:.6f}")

    text_tokenizer = CharTokenizer(config.text_vocab_size)
    time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
    score_tokenizer = NumericTokenizer(config.score_vocab, width=3)
    predictions = []
    exact_matches = 0
    model.eval()
    for index, sample in enumerate(dataset):
        prompt_ids = sample["token_ids"][: sample["prompt_length"]].unsqueeze(0).to(device)
        generated = model.generate(
            sample["frames"].unsqueeze(0).to(device),
            sample["frame_times"].unsqueeze(0).to(device),
            prompt_ids,
            max_new_tokens=config.max_generated_tokens,
            visual_patch_features=visual_patch_features[index : index + 1],
        )
        predicted_events = decode_event_sequence(
            generated[0, prompt_ids.size(1) :].tolist(),
            config,
            text_tokenizer,
            time_tokenizer,
            score_tokenizer,
        )
        is_exact = predicted_events == sample["events"]
        exact_matches += int(is_exact)
        predictions.append(
            {
                "sample_index": index,
                "exact_match": is_exact,
                "ground_truth": sample["events"],
                "predicted": predicted_events,
            }
        )

    final_loss = history[-1]["loss"]
    exact_rate = exact_matches / len(dataset)
    passed = final_loss <= args.success_loss and exact_matches == len(dataset)
    summary = {
        "passed": passed,
        "samples": len(dataset),
        "epochs": args.epochs,
        "seed": args.seed,
        "final_loss": final_loss,
        "success_loss": args.success_loss,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_rate,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(
        json.dumps(predictions, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {"model_state": model.state_dict(), "config": config.to_dict(), "summary": summary},
        output_dir / "tinytrace.pt",
    )

    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
