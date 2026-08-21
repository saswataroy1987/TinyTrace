"""Direct event-aligned MobileCLIP patch features conditioned into FLAN-T5."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .config import DirectMobileCLIPCaptionConfig


def event_frame_indices(
    frame_times: torch.Tensor,
    frame_mask: torch.Tensor,
    segments: torch.Tensor,
    max_event_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordered cache indices for every normalized event segment."""
    if frame_times.ndim != 2 or frame_mask.shape != frame_times.shape or segments.ndim != 3:
        raise ValueError("frame_times/frame_mask must be [B,T] and segments [B,N,2].")
    batches, frames = frame_times.shape
    events = segments.size(1)
    indices_out = torch.zeros((batches, events, max_event_frames), dtype=torch.long, device=frame_times.device)
    selected_mask = torch.zeros((batches, events, max_event_frames), dtype=torch.bool, device=frame_times.device)
    maximum = frame_times.masked_fill(~frame_mask, 0).amax(dim=1).clamp_min(1e-6)
    normalized_times = frame_times / maximum[:, None]
    for batch_index in range(batches):
        valid_indices = frame_mask[batch_index].nonzero(as_tuple=False).flatten()
        if not valid_indices.numel():
            continue
        valid_times = normalized_times[batch_index, valid_indices]
        for event_index in range(events):
            start, end = segments[batch_index, event_index]
            if not bool(end > start):
                continue
            inside = valid_indices[(valid_times >= start) & (valid_times <= end)]
            if not inside.numel():
                midpoint = (start + end) / 2
                inside = valid_indices[(valid_times - midpoint).abs().argmin().view(1)]
            positions = torch.linspace(0, inside.numel() - 1, steps=min(max_event_frames, inside.numel()), device=frame_times.device)
            chosen = inside[positions.round().long()]
            indices_out[batch_index, event_index, : chosen.numel()] = chosen
            selected_mask[batch_index, event_index, : chosen.numel()] = True
    return indices_out, selected_mask


def select_event_patch_features(
    features: torch.Tensor,
    frame_times: torch.Tensor,
    frame_mask: torch.Tensor,
    segments: torch.Tensor,
    max_event_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ordered cached frames for every normalized event segment.

    The output preserves each selected frame's complete 64 MobileCLIP patch
    tokens. A narrow event without a sampled frame gets its nearest valid
    frame, which makes the selection total and explicitly auditable.
    """
    if features.ndim != 4 or segments.ndim != 3:
        raise ValueError("features must be [B,T,P,D] and segments must be [B,N,2].")
    if frame_times.shape != features.shape[:2] or frame_mask.shape != features.shape[:2]:
        raise ValueError("frame_times and frame_mask must be [B,T].")
    if max_event_frames < 1:
        raise ValueError("max_event_frames must be positive.")
    batches, _, patches, width = features.shape
    events = segments.size(1)
    selected = features.new_zeros((batches, events, max_event_frames, patches, width))
    indices_out, selected_mask = event_frame_indices(frame_times, frame_mask, segments, max_event_frames)
    for batch_index in range(batches):
        for event_index in range(events):
            count = int(selected_mask[batch_index, event_index].sum())
            if count:
                selected[batch_index, event_index, :count] = features[batch_index, indices_out[batch_index, event_index, :count]]
    return selected, selected_mask


class EventPatchResampler(nn.Module):
    """Cross-attention bridge from event patch tokens to FLAN hidden tokens."""

    def __init__(self, config: DirectMobileCLIPCaptionConfig, hidden_size: int) -> None:
        super().__init__()
        if hidden_size % config.visual_heads:
            raise ValueError("FLAN hidden size must be divisible by visual_heads.")
        self.config = config
        self.patch_projection = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, hidden_size))
        self.spatial_positions = nn.Parameter(torch.empty(config.patch_tokens, hidden_size))
        self.temporal_positions = nn.Parameter(torch.empty(config.max_event_frames, hidden_size))
        self.queries = nn.Parameter(torch.empty(config.visual_tokens, hidden_size))
        self.cross_attention = nn.MultiheadAttention(hidden_size, config.visual_heads, dropout=config.visual_dropout, batch_first=True)
        self.norm_query = nn.LayerNorm(hidden_size)
        self.norm_memory = nn.LayerNorm(hidden_size)
        self.norm_output = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(config.visual_dropout), nn.Linear(hidden_size * 4, hidden_size))
        self.dropout = nn.Dropout(config.visual_dropout)
        nn.init.normal_(self.spatial_positions, std=0.02)
        nn.init.normal_(self.temporal_positions, std=0.02)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, event_features: torch.Tensor, event_frame_mask: torch.Tensor) -> torch.Tensor:
        if event_features.ndim != 4:
            raise ValueError("event_features must be [events,frames,patches,features].")
        events, frames, patches, _ = event_features.shape
        if frames != self.config.max_event_frames or patches != self.config.patch_tokens:
            raise ValueError("Event feature dimensions do not match the v3 configuration.")
        if event_frame_mask.shape != (events, frames) or not bool(event_frame_mask.any(dim=1).all()):
            raise ValueError("Every flattened event must contain at least one selected frame.")
        memory = self.patch_projection(event_features.float())
        memory = memory + self.spatial_positions.view(1, 1, patches, -1) + self.temporal_positions.view(1, frames, 1, -1)
        memory = memory.reshape(events, frames * patches, -1)
        token_mask = event_frame_mask[:, :, None].expand(-1, -1, patches).reshape(events, frames * patches)
        query = self.queries.unsqueeze(0).expand(events, -1, -1)
        attended, _ = self.cross_attention(self.norm_query(query), self.norm_memory(memory), self.norm_memory(memory), key_padding_mask=~token_mask, need_weights=False)
        tokens = query + self.dropout(attended)
        return self.norm_output(tokens + self.dropout(self.feed_forward(tokens)))


class DirectMobileCLIPCaptionModel(nn.Module):
    """Stage 2 v3 captioner whose primary evidence is cached MobileCLIP patches."""

    def __init__(self, config: DirectMobileCLIPCaptionConfig, language_model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.config, self.language_model, self.tokenizer = config, language_model, tokenizer
        self.adapter = EventPatchResampler(config, int(language_model.config.d_model))
        self.temporal_context_bridge = nn.Sequential(
            nn.LayerNorm(config.stage1_context_dim),
            nn.Linear(config.stage1_context_dim, int(language_model.config.d_model)),
            nn.GELU(),
            nn.LayerNorm(int(language_model.config.d_model)),
        ) if config.use_stage1_temporal_context else None
        self._freeze_language_model()

    @classmethod
    def from_pretrained(cls, config: DirectMobileCLIPCaptionConfig) -> "DirectMobileCLIPCaptionModel":
        try:
            from transformers import AutoTokenizer, T5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError("Stage 2 v3 requires transformers and sentencepiece.") from exc
        kwargs = {"revision": config.flan_revision, "local_files_only": config.flan_local_files_only}
        return cls(config, T5ForConditionalGeneration.from_pretrained(config.flan_model_name, **kwargs), AutoTokenizer.from_pretrained(config.flan_model_name, **kwargs))

    def _freeze_language_model(self) -> None:
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False
        for block in list(self.language_model.encoder.block)[-self.config.train_flan_encoder_layers :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for block in list(self.language_model.decoder.block)[-self.config.train_flan_decoder_layers :]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def _instruction_embeddings(self, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer([self.config.instruction] * count, padding=True, return_tensors="pt")
        ids, mask = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
        return self.language_model.get_input_embeddings()(ids), mask

    def conditioning(self, event_features: torch.Tensor, event_frame_mask: torch.Tensor, temporal_context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.adapter(event_features, event_frame_mask)
        if self.temporal_context_bridge is not None:
            if temporal_context is None or temporal_context.ndim != 3 or temporal_context.shape != (visual.size(0), self.config.temporal_context_tokens, self.config.stage1_context_dim):
                raise ValueError("Configured Stage 1 context must be [events,temporal_context_tokens,stage1_context_dim].")
            visual = torch.cat((visual, self.temporal_context_bridge(temporal_context)), dim=1)
        elif temporal_context is not None:
            raise ValueError("Temporal context was passed although use_stage1_temporal_context=false.")
        instruction, instruction_mask = self._instruction_embeddings(visual.size(0), visual.device)
        visual_mask = torch.ones(visual.shape[:2], dtype=instruction_mask.dtype, device=visual.device)
        return torch.cat((visual, instruction), dim=1), torch.cat((visual_mask, instruction_mask), dim=1)

    def _labels(self, captions: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(captions, padding=True, truncation=True, max_length=self.config.caption_max_tokens, return_tensors="pt")
        labels = encoded["input_ids"].to(device)
        labels[encoded["attention_mask"].to(device) == 0] = -100
        return labels

    def forward(self, event_features: torch.Tensor, event_frame_mask: torch.Tensor, captions: list[str], temporal_context: torch.Tensor | None = None) -> torch.Tensor:
        inputs, mask = self.conditioning(event_features, event_frame_mask, temporal_context)
        output = self.language_model(inputs_embeds=inputs, attention_mask=mask, labels=self._labels(captions, inputs.device), return_dict=True)
        if output.loss is None:
            raise RuntimeError("FLAN-T5 did not return a caption loss.")
        return output.loss

    @torch.no_grad()
    def generate(self, event_features: torch.Tensor, event_frame_mask: torch.Tensor, temporal_context: torch.Tensor | None = None) -> list[str]:
        inputs, mask = self.conditioning(event_features, event_frame_mask, temporal_context)
        tokens = self.language_model.generate(inputs_embeds=inputs, attention_mask=mask, max_new_tokens=self.config.generation_max_tokens, min_new_tokens=self.config.generation_min_tokens, eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=self.config.repetition_penalty, no_repeat_ngram_size=self.config.no_repeat_ngram_size)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
