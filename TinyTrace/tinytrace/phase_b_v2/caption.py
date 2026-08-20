"""Ground-truth event pooling and FLAN-T5 Small conditioning for Phase B v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .config import PhaseBV2Config


def pool_event_features(features: torch.Tensor, frame_times: torch.Tensor, frame_mask: torch.Tensor, segments: torch.Tensor, token_count: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool ordered, soft temporal tokens from each normalized event segment.

    Returns one pooled feature per segment and a validity mask. Segment bounds are
    normalized, while frame times are normalized independently for each video.
    """
    if features.ndim != 3 or segments.ndim != 3:
        raise ValueError("features must be [B,T,D] and segments must be [B,N,2]")
    if token_count < 1:
        raise ValueError("token_count must be positive.")
    maximum = frame_times.masked_fill(~frame_mask, 0).amax(dim=1, keepdim=True).clamp_min(1e-6)
    normalized_time = frame_times / maximum
    starts, ends = segments[..., 0], segments[..., 1]
    widths = (ends - starts).clamp_min(1e-4)
    positions = (torch.arange(token_count, device=features.device, dtype=features.dtype) + 0.5) / token_count
    centres = starts.unsqueeze(-1) + widths.unsqueeze(-1) * positions
    token_widths = (widths / token_count).unsqueeze(-1)
    distance = (normalized_time[:, None, None] - centres.unsqueeze(-1)).abs() / (token_widths.unsqueeze(-1) / 2)
    weights = (1 - distance).clamp_min(0) * frame_mask[:, None, None].to(features.dtype)
    # A narrow event between sampled frames still needs evidence: use its nearest valid frame.
    nearest = ((normalized_time[:, None, None] - centres.unsqueeze(-1)).abs() + (~frame_mask[:, None, None]).to(features.dtype) * 2).argmin(dim=-1)
    empty = weights.sum(dim=-1) == 0
    weights = weights.scatter_add(-1, nearest.unsqueeze(-1), empty.unsqueeze(-1).to(features.dtype))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    pooled = torch.einsum("bnkt,btd->bnkd", weights, features)
    if token_count == 1:
        pooled = pooled.squeeze(2)
    return pooled, segments[..., 1] > segments[..., 0]


@dataclass(frozen=True)
class CaptionBatch:
    conditioning: torch.Tensor
    conditioning_mask: torch.Tensor
    captions: list[str]
    batch_indices: torch.Tensor
    event_indices: torch.Tensor


def flatten_caption_events(event_features: torch.Tensor, event_mask: torch.Tensor, captions: list[list[str]], conditioning_tokens: int) -> CaptionBatch:
    rows: list[torch.Tensor] = []
    text: list[str] = []
    batch_indices: list[int] = []
    event_indices: list[int] = []
    if event_features.ndim not in {3, 4}:
        raise ValueError("event_features must be [B,N,D] or [B,N,K,D].")
    for batch_index in range(event_features.size(0)):
        for event_index in event_mask[batch_index].nonzero(as_tuple=False).flatten().tolist():
            rows.append(event_features[batch_index, event_index])
            text.append(captions[batch_index][event_index])
            batch_indices.append(batch_index)
            event_indices.append(event_index)
    if not rows:
        raise ValueError("Caption training batch has no valid events.")
    values = torch.stack(rows)
    if values.ndim == 2:
        values = values.unsqueeze(1).expand(-1, conditioning_tokens, -1)
    if values.size(1) != conditioning_tokens:
        raise ValueError("Event token count must equal conditioning_tokens.")
    return CaptionBatch(values, torch.ones(values.shape[:2], dtype=torch.long, device=values.device), text,
                        torch.tensor(batch_indices, device=values.device), torch.tensor(event_indices, device=values.device))


class FlanT5Captioner(nn.Module):
    """Visual bridge plus a pinned FLAN-T5 conditional generation model."""

    def __init__(self, config: PhaseBV2Config, model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.config, self.model, self.tokenizer = config, model, tokenizer
        hidden = int(model.config.d_model)
        self.bridge = nn.Sequential(nn.Linear(config.d_model, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.condition_positions = nn.Parameter(torch.empty(config.conditioning_tokens, hidden))
        nn.init.normal_(self.condition_positions, std=0.02)
        self._freeze_base_model()

    @classmethod
    def from_pretrained(cls, config: PhaseBV2Config) -> "FlanT5Captioner":
        try:
            from transformers import AutoTokenizer, T5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Phase B v2 captioning requires transformers and sentencepiece. Install TinyTrace/requirements.txt.") from exc
        kwargs = {"revision": config.flan_revision, "local_files_only": config.flan_local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(config.flan_model_name, **kwargs)
        model = T5ForConditionalGeneration.from_pretrained(config.flan_model_name, **kwargs)
        return cls(config, model, tokenizer)

    def _freeze_base_model(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        blocks = list(getattr(getattr(self.model, "decoder", None), "block", []))
        for block in blocks[-self.config.train_flan_decoder_layers:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        encoder_blocks = list(getattr(getattr(self.model, "encoder", None), "block", []))
        for block in encoder_blocks[-self.config.train_flan_encoder_layers:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def conditioning_tokens(self, event_features: torch.Tensor) -> torch.Tensor:
        if event_features.ndim == 2:
            event_features = event_features.unsqueeze(1).expand(-1, self.config.conditioning_tokens, -1)
        if event_features.ndim != 3 or event_features.size(1) != self.config.conditioning_tokens:
            raise ValueError("event_features must be [events,D] or [events,conditioning_tokens,D].")
        return self.bridge(event_features) + self.condition_positions.unsqueeze(0)

    def _tokenize(self, captions: list[str], device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        encoded = self.tokenizer(captions, padding=True, truncation=True, max_length=self.config.caption_max_tokens, return_tensors="pt")
        original = self.tokenizer(captions, padding=False, truncation=False, add_special_tokens=True)
        original_lengths = [len(item) for item in original["input_ids"]]
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        return {"labels": labels.to(device)}, {"caption_count": len(captions), "truncated_caption_count": sum(length > self.config.caption_max_tokens for length in original_lengths), "original_tokens": sum(original_lengths), "retained_tokens": int(encoded["attention_mask"].sum())}

    def forward(self, event_features: torch.Tensor, captions: list[str]) -> tuple[torch.Tensor, dict[str, int]]:
        conditioning = self.conditioning_tokens(event_features)
        mask = torch.ones(conditioning.shape[:2], dtype=torch.long, device=conditioning.device)
        labels, report = self._tokenize(captions, conditioning.device)
        output = self.model(inputs_embeds=conditioning, attention_mask=mask, labels=labels["labels"], return_dict=True)
        if output.loss is None:
            raise RuntimeError("FLAN-T5 did not return a caption loss.")
        return output.loss, report

    @torch.no_grad()
    def generate(self, event_features: torch.Tensor) -> list[str]:
        conditioning = self.conditioning_tokens(event_features)
        tokens = self.model.generate(inputs_embeds=conditioning, attention_mask=torch.ones(conditioning.shape[:2], dtype=torch.long, device=conditioning.device), max_new_tokens=self.config.generation_max_tokens, min_new_tokens=self.config.generation_min_tokens, eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=self.config.repetition_penalty, no_repeat_ngram_size=self.config.no_repeat_ngram_size)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
