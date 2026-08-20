"""Masked temporal detector, matching, losses, and inference filtering for Phase B v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import PhaseBV2Config


def temporal_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    left, right = torch.maximum(a[..., 0], b[..., 0]), torch.minimum(a[..., 1], b[..., 1])
    intersection = (right - left).clamp_min(0)
    union = torch.maximum(a[..., 1], b[..., 1]) - torch.minimum(a[..., 0], b[..., 0])
    return intersection / union.clamp_min(1e-6)


def temporal_giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    iou = temporal_iou(a, b)
    union = torch.maximum(a[..., 1], b[..., 1]) - torch.minimum(a[..., 0], b[..., 0])
    enclosure = (torch.maximum(a[..., 1], b[..., 1]) - torch.minimum(a[..., 0], b[..., 0])).clamp_min(1e-6)
    return iou - (enclosure - union) / enclosure


def centre_duration_to_segment(values: torch.Tensor) -> torch.Tensor:
    """Convert normalized centre/duration predictions into finite ordered segments."""
    centre, duration = values.unbind(dim=-1)
    start, end = (centre - duration / 2).clamp(0, 1), (centre + duration / 2).clamp(0, 1)
    epsilon = torch.finfo(values.dtype).eps
    end = torch.maximum(end, start + epsilon).clamp_max(1)
    start = torch.minimum(start, end - epsilon).clamp_min(0)
    return torch.stack((start, end), dim=-1)


def _assignment(rows_to_columns_cost: torch.Tensor) -> list[tuple[int, int]]:
    """Kuhn-Munkres assignment where rows <= columns; kept dependency-free."""
    rows, columns = rows_to_columns_cost.shape
    costs = rows_to_columns_cost.detach().double().cpu().tolist()
    u, v, matched, path = [0.0] * (rows + 1), [0.0] * (columns + 1), [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, rows + 1):
        matched[0], column0 = row, 0
        minimum, used = [float("inf")] * (columns + 1), [False] * (columns + 1)
        while True:
            used[column0] = True
            row0, delta, column1 = matched[column0], float("inf"), 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced = costs[row0 - 1][column - 1] - u[row0] - v[column]
                if reduced < minimum[column]:
                    minimum[column], path[column] = reduced, column0
                if minimum[column] < delta:
                    delta, column1 = minimum[column], column
            for column in range(columns + 1):
                if used[column]:
                    u[matched[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if matched[column0] == 0:
                break
        while column0:
            previous = path[column0]
            matched[column0], column0 = matched[previous], previous
    return [(matched[column] - 1, column - 1) for column in range(1, columns + 1) if matched[column]]


def hungarian_match(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query and target indices for the minimum one-to-one assignment."""
    if cost.ndim != 2:
        raise ValueError("cost must have shape [queries, targets]")
    queries, targets = cost.shape
    empty = torch.empty(0, dtype=torch.long, device=cost.device)
    if not queries or not targets:
        return empty, empty
    if queries >= targets:
        pairs = _assignment(cost.transpose(0, 1))
        targets_out = torch.tensor([row for row, _ in pairs], dtype=torch.long, device=cost.device)
        queries_out = torch.tensor([column for _, column in pairs], dtype=torch.long, device=cost.device)
    else:
        pairs = _assignment(cost)
        queries_out = torch.tensor([row for row, _ in pairs], dtype=torch.long, device=cost.device)
        targets_out = torch.tensor([column for _, column in pairs], dtype=torch.long, device=cost.device)
    return queries_out, targets_out


class TemporalEventDetector(nn.Module):
    """Detector consuming only Stage 0's frozen MobileCLIP feature tensors."""

    def __init__(self, config: PhaseBV2Config) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(nn.Linear(config.feature_dim, config.d_model), nn.LayerNorm(config.d_model))
        self.time_embedding = nn.Sequential(nn.Linear(2, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(config.d_model, config.temporal_heads, config.d_model * 4, config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, config.temporal_layers)
        self.multiscale = nn.Conv1d(config.d_model, config.d_model, config.multiscale_kernel_size, padding=config.multiscale_kernel_size // 2, groups=config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(config.d_model, config.temporal_heads, config.d_model * 4, config.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.queries = nn.Embedding(config.event_queries, config.d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.event_decoder_layers)
        self.confidence = nn.Linear(config.d_model, 1)
        self.segment = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, 2))

    def encode(self, visual_features: torch.Tensor, frame_times: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if visual_features.ndim != 4 or tuple(visual_features.shape[2:]) != (self.config.patch_tokens, self.config.feature_dim):
            raise ValueError("visual_features must have shape [B,T,patch_tokens,feature_dim] matching config")
        if frame_times.shape != visual_features.shape[:2] or frame_mask.shape != visual_features.shape[:2]:
            raise ValueError("frame_times and frame_mask must have shape [B,T]")
        frame_features = visual_features.float().mean(dim=2)
        positions = torch.linspace(0, 1, frame_features.size(1), device=frame_features.device).expand(frame_features.size(0), -1)
        max_time = frame_times.masked_fill(~frame_mask, 0).amax(dim=1, keepdim=True).clamp_min(1e-6)
        temporal_input = self.projection(frame_features) + self.time_embedding(torch.stack((frame_times / max_time, positions), dim=-1).to(frame_features.dtype))
        encoded = self.encoder(temporal_input, src_key_padding_mask=~frame_mask)
        encoded = encoded + self.multiscale(encoded.transpose(1, 2)).transpose(1, 2)
        return encoded.masked_fill(~frame_mask.unsqueeze(-1), 0)

    def forward(self, visual_features: torch.Tensor, frame_times: torch.Tensor, frame_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encode(visual_features, frame_times, frame_mask)
        queries = self.queries.weight.unsqueeze(0).expand(memory.size(0), -1, -1)
        query_features = self.decoder(queries, memory, memory_key_padding_mask=~frame_mask)
        return {"temporal_features": memory, "query_features": query_features, "event_logits": self.confidence(query_features).squeeze(-1), "segments": centre_duration_to_segment(self.segment(query_features).sigmoid())}


@dataclass
class LocalizationLoss:
    total: torch.Tensor
    event: torch.Tensor
    l1: torch.Tensor
    giou: torch.Tensor
    matches: list[tuple[torch.Tensor, torch.Tensor]]


def localization_loss(outputs: dict[str, torch.Tensor], targets: torch.Tensor, target_mask: torch.Tensor, config: PhaseBV2Config) -> LocalizationLoss:
    logits, predicted = outputs["event_logits"], outputs["segments"]
    event_targets, matches, l1_terms, giou_terms = torch.zeros_like(logits), [], [], []
    for index in range(logits.size(0)):
        valid = target_mask[index].nonzero(as_tuple=False).flatten()
        if not valid.numel():
            matches.append((valid, valid))
            continue
        truth, current = targets[index, valid], predicted[index]
        cost = (config.matcher_class_cost * -logits[index].sigmoid()[:, None]
                + config.matcher_l1_cost * torch.cdist(current, truth, p=1)
                + config.matcher_giou_cost * (1 - temporal_iou(current[:, None], truth[None])))
        query_indices, local_target_indices = hungarian_match(cost)
        target_indices = valid[local_target_indices]
        matches.append((query_indices, target_indices))
        event_targets[index, query_indices] = 1
        l1_terms.append(F.l1_loss(current[query_indices], targets[index, target_indices]))
        giou_terms.append(1 - temporal_giou(current[query_indices], targets[index, target_indices]).mean())
    weight = torch.tensor([config.no_event_weight, 1.0], device=logits.device, dtype=logits.dtype)
    event = F.binary_cross_entropy_with_logits(logits, event_targets, weight=torch.where(event_targets > 0, weight[1], weight[0]))
    l1 = torch.stack(l1_terms).mean() if l1_terms else logits.new_zeros(())
    giou = torch.stack(giou_terms).mean() if giou_terms else logits.new_zeros(())
    total = config.loss_event_weight * event + config.loss_l1_weight * l1 + config.loss_giou_weight * giou
    return LocalizationLoss(total, event, l1, giou, matches)


def filter_events(segments: torch.Tensor, logits: torch.Tensor, threshold: float, overlap_threshold: float) -> list[dict[str, float]]:
    """Filter low-confidence/overlapping segments and return chronological predictions."""
    chosen: list[dict[str, float]] = []
    for index in logits.sigmoid().argsort(descending=True).tolist():
        score, candidate = float(logits[index].sigmoid()), segments[index]
        if score < threshold:
            break
        if any(float(temporal_iou(candidate, torch.tensor([item["start"], item["end"]], device=candidate.device))) > overlap_threshold for item in chosen):
            continue
        chosen.append({"start": float(candidate[0]), "end": float(candidate[1]), "score": score})
    return sorted(chosen, key=lambda item: (item["start"], item["end"]))
