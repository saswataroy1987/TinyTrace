from __future__ import annotations

from enum import IntEnum

from .config import TinyTraceConfig
from .tokenizers import CharTokenizer, NumericTokenizer


class LabelType(IntEnum):
    IGNORE = -1
    TEXT = 0
    TEXT_SYNC = 1
    TIME = 2
    SCORE = 3
    TIME_SYNC = 4
    SCORE_SYNC = 5
    HIGHLIGHT_BOUNDARY = 6


def caption_budget_metadata(
    events: list[dict],
    config: TinyTraceConfig,
    text_tokenizer: CharTokenizer,
) -> dict[str, object]:
    """Make target caption truncation explicit without changing serialization."""
    if not isinstance(events, list):
        raise ValueError("events must be a list.")
    details = []
    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"Event {event_index} must be an object.")
        caption = event.get("caption", "")
        if not isinstance(caption, str):
            raise ValueError(f"Event {event_index} caption must be a string when provided.")
        original = len(text_tokenizer.encode(caption))
        retained = min(original, config.max_caption_tokens)
        details.append(
            {
                "event_index": event_index,
                "original_tokens": original,
                "retained_tokens": retained,
                "truncated": original > retained,
            }
        )
    return {
        "max_caption_tokens": config.max_caption_tokens,
        "event_count": len(details),
        "truncated_event_count": sum(bool(item["truncated"]) for item in details),
        "original_caption_tokens": sum(int(item["original_tokens"]) for item in details),
        "retained_caption_tokens": sum(int(item["retained_tokens"]) for item in details),
        "events": details,
    }


def serialize_example(
    events: list[dict],
    instruction: str,
    config: TinyTraceConfig,
    text_tokenizer: CharTokenizer,
    time_tokenizer: NumericTokenizer,
    score_tokenizer: NumericTokenizer,
    task_mode: str = "caption",
) -> tuple[list[int], list[int], int]:
    """Build the canonical instruction + causal event target sequence."""
    if not isinstance(events, list):
        raise ValueError("events must be a list.")
    if len(events) > config.max_events:
        raise ValueError(
            f"Received {len(events)} events, but max_events={config.max_events}."
        )
    if task_mode not in {"caption", "highlight"}:
        raise ValueError("task_mode must be either 'caption' or 'highlight'.")
    if not isinstance(instruction, str):
        raise ValueError("instruction must be a string.")
    instruction_ids = [config.bos_token_id]
    instruction_ids.extend(text_tokenizer.encode(instruction)[: config.max_text_len])
    instruction_ids.append(config.video_token_id)
    prompt_length = len(instruction_ids)

    token_ids = list(instruction_ids)
    label_types = [LabelType.IGNORE] * prompt_length

    for event_index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValueError(f"Event {event_index} must be an object.")
        timestamps = event.get("timestamp")
        scores = event.get("score")
        caption = event.get("caption", "")
        if not isinstance(timestamps, list) or len(timestamps) != config.timestamp_value_count:
            raise ValueError(
                f"Event {event_index} must contain {config.timestamp_value_count} timestamp values."
            )
        if not isinstance(scores, list) or len(scores) != config.score_value_count:
            raise ValueError(f"Event {event_index} must contain {config.score_value_count} score value.")
        if task_mode == "caption" and (not isinstance(caption, str) or not caption.strip()):
            raise ValueError(f"Event {event_index} must contain a non-empty caption.")

        time_ids = [
            config.sync_token_id if token_id == 0 else config.time_token_base + token_id
            for token_id in time_tokenizer.encode([float(value) for value in timestamps])
        ]
        score_ids = [
            config.sync_token_id if token_id == 0 else config.score_token_base + token_id
            for token_id in score_tokenizer.encode([float(value) for value in scores])
        ]
        caption_ids = text_tokenizer.encode(caption)[: config.max_caption_tokens]
        caption_ids.append(config.sync_token_id)

        token_ids.extend(time_ids)
        label_types.extend(
            LabelType.TIME_SYNC if token_id == config.sync_token_id else LabelType.TIME
            for token_id in time_ids
        )
        token_ids.extend(score_ids)
        label_types.extend(
            (
                LabelType.HIGHLIGHT_BOUNDARY
                if task_mode == "highlight" and token_id == config.sync_token_id
                else LabelType.SCORE_SYNC
                if token_id == config.sync_token_id
                else LabelType.SCORE
            )
            for token_id in score_ids
        )
        if task_mode == "caption":
            token_ids.extend(caption_ids)
            label_types.extend(
                LabelType.TEXT_SYNC if token_id == config.sync_token_id else LabelType.TEXT
                for token_id in caption_ids
            )

    token_ids.append(config.eos_token_id)
    label_types.append(LabelType.TEXT)
    return token_ids, [int(label_type) for label_type in label_types], prompt_length
