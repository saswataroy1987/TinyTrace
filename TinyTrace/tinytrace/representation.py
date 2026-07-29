from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F

from .data import sample_uniform_frame_times


FRAME_COUNT_LADDER = (8, 12, 16, 24, 32)
CAPTION_TOKEN_LADDER = (20, 48, 64)


def temporal_coverage_report(
    duration: float,
    frame_counts: Iterable[int] = FRAME_COUNT_LADDER,
    *,
    safety_margin: float = 0.25,
    tokens_per_frame: int = 10,
) -> list[dict[str, object]]:
    """Describe uniform-sampling coverage without decoding or acquiring media."""
    counts = tuple(frame_counts)
    if not counts:
        raise ValueError("frame_counts cannot be empty.")
    if not isinstance(tokens_per_frame, int) or isinstance(tokens_per_frame, bool) or tokens_per_frame < 1:
        raise ValueError("tokens_per_frame must be a positive integer.")

    baseline = counts[0]
    if not isinstance(baseline, int) or isinstance(baseline, bool) or baseline < 1:
        raise ValueError("frame_counts must contain positive integers.")

    rows: list[dict[str, object]] = []
    for requested in counts:
        timestamps = sample_uniform_frame_times(duration, requested, safety_margin)
        spacings = timestamps[1:] - timestamps[:-1]
        valid_count = int(timestamps.numel())
        rows.append(
            {
                "requested_frames": requested,
                "valid_frames": valid_count,
                "padded_frames": requested - valid_count,
                "timestamps_seconds": [float(value) for value in timestamps.tolist()],
                "first_timestamp_seconds": float(timestamps[0]),
                "last_timestamp_seconds": float(timestamps[-1]),
                "mean_spacing_seconds": float(spacings.mean()) if spacings.numel() else None,
                "max_spacing_seconds": float(spacings.max()) if spacings.numel() else None,
                "visual_time_prefix_tokens": requested * tokens_per_frame,
                "relative_decoded_frame_work": requested / baseline,
            }
        )
    return rows


def _mean_pairwise_cosine(vectors: torch.Tensor) -> tuple[float | None, int]:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [count, channels].")
    count = vectors.size(0)
    if count < 2:
        return None, 0
    normalized = F.normalize(vectors.float(), dim=-1, eps=1e-12)
    similarities = normalized @ normalized.transpose(0, 1)
    mask = ~torch.eye(count, dtype=torch.bool, device=vectors.device)
    values = similarities[mask]
    return float(values.mean().cpu()), int(values.numel())


def visual_feature_diversity(
    patch_features: torch.Tensor,
    compressed_tokens: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
) -> dict[str, float | int | None]:
    """Report feature similarity across valid frames and compressed slots.

    Patch features must be ``[B, T, P, C]`` and compressed tokens
    ``[B, T, S, D]``. This function is diagnostic and does not add a loss.
    """
    if patch_features.ndim != 4:
        raise ValueError("patch_features must have shape [batch, frames, patches, channels].")
    if compressed_tokens.ndim != 4:
        raise ValueError("compressed_tokens must have shape [batch, frames, slots, channels].")
    if patch_features.shape[:2] != compressed_tokens.shape[:2]:
        raise ValueError("Patch features and compressed tokens must share batch/frame dimensions.")
    batch, frames = patch_features.shape[:2]
    if frame_mask is None:
        frame_mask = torch.ones(batch, frames, dtype=torch.bool, device=patch_features.device)
    if frame_mask.shape != (batch, frames):
        raise ValueError("frame_mask must have shape [batch, frames].")
    if frame_mask.dtype != torch.bool:
        raise ValueError("frame_mask must be boolean.")

    frame_similarities: list[float] = []
    frame_pairs = 0
    slot_similarities: list[float] = []
    slot_pairs = 0
    frame_summaries = patch_features.mean(dim=2)
    for batch_index in range(batch):
        valid = frame_mask[batch_index]
        value, pairs = _mean_pairwise_cosine(frame_summaries[batch_index, valid])
        if value is not None:
            frame_similarities.extend([value] * pairs)
            frame_pairs += pairs
        for tokens in compressed_tokens[batch_index, valid]:
            value, pairs = _mean_pairwise_cosine(tokens)
            if value is not None:
                slot_similarities.extend([value] * pairs)
                slot_pairs += pairs

    return {
        "mean_frame_patch_summary_cosine_similarity": (
            sum(frame_similarities) / len(frame_similarities) if frame_similarities else None
        ),
        "frame_similarity_pair_count": frame_pairs,
        "mean_compressed_slot_cosine_similarity": (
            sum(slot_similarities) / len(slot_similarities) if slot_similarities else None
        ),
        "slot_similarity_pair_count": slot_pairs,
        "valid_frame_count": int(frame_mask.sum().item()),
    }


def aggregate_caption_budget(reports: Iterable[dict[str, object]]) -> dict[str, float | int]:
    """Aggregate per-sample caption budget metadata for logs and reports."""
    sample_count = 0
    unavailable_sample_count = 0
    event_count = 0
    truncated_events = 0
    original_tokens = 0
    retained_tokens = 0
    for report in reports:
        sample_count += 1
        if report.get("available") is False:
            unavailable_sample_count += 1
        event_count += int(report.get("event_count", 0))
        truncated_events += int(report.get("truncated_event_count", 0))
        original_tokens += int(report.get("original_caption_tokens", 0))
        retained_tokens += int(report.get("retained_caption_tokens", 0))
    return {
        "sample_count": sample_count,
        "unavailable_sample_count": unavailable_sample_count,
        "event_count": event_count,
        "truncated_event_count": truncated_events,
        "truncated_event_rate": truncated_events / event_count if event_count else 0.0,
        "original_caption_tokens": original_tokens,
        "retained_caption_tokens": retained_tokens,
        "retained_caption_token_rate": retained_tokens / original_tokens if original_tokens else 1.0,
    }
