from __future__ import annotations

import argparse

import torch

from tinytrace import JsonTinyTraceDataset, SyntheticTinyTraceDataset, TinyTraceConfig, TinyTraceModel, decode_event_sequence
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="TinyTrace/outputs/tinytrace.pt")
    parser.add_argument("--dataset-json", type=str, default="")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain the TinyTrace configuration.")
    config = TinyTraceConfig.from_dict(checkpoint["config"])
    model = TinyTraceModel(config, load_pretrained_visual=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = JsonTinyTraceDataset(args.dataset_json, config=config) if args.dataset_json else SyntheticTinyTraceDataset(config=config, size=1)
    sample = dataset[args.sample_index]

    prompt_ids = sample["token_ids"][: sample["prompt_length"]].unsqueeze(0).to(device)
    generated = model.generate(
        sample["frames"].unsqueeze(0).to(device),
        sample["frame_times"].unsqueeze(0).to(device),
        prompt_ids,
        max_new_tokens=config.max_generated_tokens,
    )

    text_tokenizer = CharTokenizer(config.text_vocab_size)
    time_tokenizer = NumericTokenizer(config.time_vocab, width=6)
    score_tokenizer = NumericTokenizer(config.score_vocab, width=3)
    parsed = decode_event_sequence(
        generated[0].tolist()[prompt_ids.size(1) :],
        config,
        text_tokenizer,
        time_tokenizer,
        score_tokenizer,
    )

    print("ground_truth:", sample["events"])
    print("predicted:", parsed)


if __name__ == "__main__":
    main()
