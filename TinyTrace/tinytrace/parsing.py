from __future__ import annotations

from .config import TinyTraceConfig
from .tokenizers import CharTokenizer, NumericTokenizer


class EventParseError(ValueError):
    pass


def decode_event_sequence(
    token_ids: list[int],
    config: TinyTraceConfig,
    text_tokenizer: CharTokenizer,
    time_tokenizer: NumericTokenizer,
    score_tokenizer: NumericTokenizer,
    strict: bool = False,
    warnings: list[str] | None = None,
    task_mode: str = "caption",
) -> list[dict]:
    if task_mode not in {"caption", "highlight"}:
        raise ValueError("task_mode must be either 'caption' or 'highlight'.")
    events: list[dict] = []
    mode = "time"
    time_ids: list[int] = []
    score_ids: list[int] = []
    caption_ids: list[int] = []
    malformed = False

    def flush_event() -> None:
        nonlocal time_ids, score_ids, caption_ids, malformed
        if not time_ids and not score_ids and not caption_ids:
            return
        try:
            timestamps = time_tokenizer.decode_values(time_ids)
            scores = score_tokenizer.decode_values(score_ids)
            caption = text_tokenizer.decode(caption_ids).strip()
            valid = (
                not malformed
                and len(timestamps) == config.timestamp_value_count
                and len(scores) == config.score_value_count
                and (task_mode == "highlight" or bool(caption))
            )
        except (KeyError, ValueError, UnicodeError) as exc:
            if strict:
                raise EventParseError("Malformed numeric or caption tokens in generated event.") from exc
            valid = False
            timestamps, scores, caption = [], [], ""

        if not valid and strict:
            raise EventParseError(
                "Generated event does not contain a complete timestamp, score, and caption."
            )
        if not valid and warnings is not None:
            warnings.append(
                "Generated event does not contain a complete timestamp, score, and caption."
            )
        if valid:
            event = {"timestamp": timestamps, "score": scores}
            if task_mode == "caption":
                event["caption"] = caption
            events.append(event)
        time_ids = []
        score_ids = []
        caption_ids = []
        malformed = False

    def reject(message: str) -> None:
        nonlocal malformed
        if strict:
            raise EventParseError(message)
        if warnings is not None:
            warnings.append(message)
        malformed = True

    for token_id in token_ids:
        if not isinstance(token_id, int) or token_id < 0 or token_id >= config.total_token_vocab:
            reject(f"Generated token ID {token_id!r} is outside the TinyTrace vocabulary.")
            continue
        if token_id == config.eos_token_id:
            flush_event()
            break
        if token_id == config.sync_token_id:
            if mode == "time":
                mode = "score"
            elif mode == "score":
                if task_mode == "highlight":
                    flush_event()
                    mode = "time"
                else:
                    mode = "caption"
            else:
                flush_event()
                mode = "time"
            continue
        if mode == "time":
            if config.time_token_base <= token_id < config.score_token_base:
                time_ids.append(token_id - config.time_token_base)
            else:
                reject(f"Token {token_id} is not valid while parsing a timestamp.")
        elif mode == "score":
            if token_id >= config.score_token_base:
                score_ids.append(token_id - config.score_token_base)
            else:
                reject(f"Token {token_id} is not valid while parsing a score.")
        else:
            if token_id < config.text_vocab_size:
                caption_ids.append(token_id)
            else:
                reject(f"Token {token_id} is not valid while parsing a caption.")

    else:
        flush_event()

    return events
