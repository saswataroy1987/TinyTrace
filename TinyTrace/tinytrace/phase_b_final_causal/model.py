"""Frozen-cache visual prefix and causal chronological FLAN-T5 target."""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn

from .config import FinalCausalConfig

EVENT, START, END, CAPTION, CLOSE, END_EVENTS = "<EVENT>", "<START>", "<END>", "<CAPTION>", "</EVENT>", "<END_EVENTS>"


def _time_token(index: int) -> str:
    return f"<T{index:03d}>"


def parse_event_sequence(text: str) -> list[dict[str, object]]:
    """Parse generated events while retaining malformed time fields for audit."""
    expression = re.compile(r"<EVENT>\s*<START>\s*(\S+)\s*<END>\s*(\S+)\s*<CAPTION>\s*(.*?)\s*</EVENT>", re.DOTALL)
    events = []
    for start, end, caption in expression.findall(text):
        start_match, end_match = re.fullmatch(r"<T(\d{3})>", start), re.fullmatch(r"<T(\d{3})>", end)
        events.append({"start_normalized": int(start_match.group(1)) / 100.0 if start_match else None, "end_normalized": int(end_match.group(1)) / 100.0 if end_match else None, "raw_start_token": start, "raw_end_token": end, "valid_time_tokens": bool(start_match and end_match), "caption": caption.strip()})
    return events


class SlotCompressor(nn.Module):
    """Eight learned cross-attention slots per frozen MobileCLIP frame."""

    def __init__(self, config: FinalCausalConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, hidden_size))
        self.slots = nn.Parameter(torch.empty(config.visual_slots_per_frame, hidden_size))
        self.cross_attention = nn.MultiheadAttention(hidden_size, config.visual_heads, dropout=config.visual_dropout, batch_first=True)
        self.norm_slots = nn.LayerNorm(hidden_size)
        self.norm_memory = nn.LayerNorm(hidden_size)
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))
        nn.init.normal_(self.slots, std=0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batches, frames, patch_count, _ = patches.shape
        memory = self.projection(patches.float()).reshape(batches * frames, patch_count, -1)
        queries = self.slots.unsqueeze(0).expand(batches * frames, -1, -1)
        attended, _ = self.cross_attention(self.norm_slots(queries), self.norm_memory(memory), self.norm_memory(memory), need_weights=False)
        return self.output(queries + attended).reshape(batches, frames, self.config.visual_slots_per_frame, -1)


class TimeTokenEncoder(nn.Module):
    """Learned six-token representation of each cached frame's normalized time."""

    def __init__(self, config: FinalCausalConfig, hidden_size: int) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.empty(config.time_tokens_per_frame, hidden_size))
        self.encoder = nn.Sequential(nn.Linear(6, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
        nn.init.normal_(self.tokens, std=0.02)

    def forward(self, normalized_times: torch.Tensor) -> torch.Tensor:
        values = normalized_times.unsqueeze(-1)
        features = torch.cat((values, values.square(), torch.sin(torch.pi * values), torch.cos(torch.pi * values), torch.sin(2 * torch.pi * values), torch.cos(2 * torch.pi * values)), dim=-1)
        return self.encoder(features).unsqueeze(-2) + self.tokens.view(1, 1, *self.tokens.shape)


class FinalCausalEventModel(nn.Module):
    """Autoregressively emits a single chronological event sequence per video."""

    def __init__(self, config: FinalCausalConfig, language_model: nn.Module, tokenizer: Any) -> None:
        super().__init__()
        self.config, self.language_model, self.tokenizer = config, language_model, tokenizer
        hidden = int(language_model.config.d_model)
        self.slot_compressor = SlotCompressor(config, hidden)
        self.time_encoder = TimeTokenEncoder(config, hidden)
        self._freeze_language_model()

    @property
    def structure_tokens(self) -> list[str]:
        return [EVENT, START, END, CAPTION, CLOSE, END_EVENTS, *[_time_token(index) for index in range(self.config.time_bins + 1)]]

    @classmethod
    def from_pretrained(cls, config: FinalCausalConfig) -> "FinalCausalEventModel":
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        kwargs = {"revision": config.flan_revision, "local_files_only": config.flan_local_files_only}
        tokenizer = AutoTokenizer.from_pretrained(config.flan_model_name, **kwargs)
        # T5 uses relative position bias rather than a fixed learned position
        # table. This is an explicit complete-event-sequence budget, not a
        # request to silently truncate the chronological target at 512 tokens.
        tokenizer.model_max_length = max(config.target_max_tokens, config.generation_max_tokens)
        language_model = T5ForConditionalGeneration.from_pretrained(config.flan_model_name, **kwargs)
        structure = [EVENT, START, END, CAPTION, CLOSE, END_EVENTS, *[_time_token(index) for index in range(config.time_bins + 1)]]
        tokenizer.add_special_tokens({"additional_special_tokens": structure})
        language_model.resize_token_embeddings(len(tokenizer))
        return cls(config, language_model, tokenizer)

    def _freeze_language_model(self) -> None:
        for parameter in self.language_model.parameters():
            parameter.requires_grad = False
        if self.config.train_full_flan_decoder:
            for parameter in self.language_model.decoder.parameters():
                parameter.requires_grad = True
        else:
            for block in list(self.language_model.decoder.block)[-1:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        for block in list(self.language_model.encoder.block)[-self.config.train_flan_encoder_layers :]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        # Newly added structured/time target embeddings must learn even with a frozen encoder.
        self.language_model.shared.weight.requires_grad = True

    def _instruction(self, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer([self.config.instruction] * count, padding=True, return_tensors="pt")
        ids, mask = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
        return self.language_model.get_input_embeddings()(ids), mask

    def select_video_frames(self, features: torch.Tensor, times: torch.Tensor, frame_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use only cached frames, preserving source order and never using labels."""
        batches, _, patches, width = features.shape
        output = features.new_zeros((batches, self.config.max_video_frames, patches, width))
        selected_times = times.new_zeros((batches, self.config.max_video_frames))
        selected_mask = torch.zeros((batches, self.config.max_video_frames), dtype=torch.bool, device=features.device)
        for batch in range(batches):
            valid = frame_mask[batch].nonzero(as_tuple=False).flatten()
            if not valid.numel():
                continue
            positions = torch.linspace(0, valid.numel() - 1, steps=min(valid.numel(), self.config.max_video_frames), device=features.device).round().long()
            indices = valid[positions]
            output[batch, :indices.numel()] = features[batch, indices]
            selected_times[batch, :indices.numel()] = times[batch, indices]
            selected_mask[batch, :indices.numel()] = True
        return output, selected_times, selected_mask

    def conditioning(self, features: torch.Tensor, times: torch.Tensor, frame_mask: torch.Tensor, durations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected, selected_times, selected_mask = self.select_video_frames(features, times, frame_mask)
        if not bool(selected_mask.any(dim=1).all()):
            raise ValueError("Every video must have at least one cached frame.")
        visual = self.slot_compressor(selected)
        normalized_times = (selected_times / durations[:, None].clamp_min(1e-6)).clamp(0, 1)
        temporal = self.time_encoder(normalized_times)
        per_frame = torch.cat((visual, temporal), dim=2)
        prefix = per_frame.flatten(1, 2)
        prefix_mask = selected_mask[:, :, None].expand(-1, -1, per_frame.size(2)).reshape(selected_mask.size(0), -1).long()
        instruction, instruction_mask = self._instruction(prefix.size(0), prefix.device)
        return torch.cat((prefix, instruction), dim=1), torch.cat((prefix_mask, instruction_mask), dim=1)

    def target_text(self, captions: list[list[str]], segments_seconds: torch.Tensor, event_mask: torch.Tensor, durations: torch.Tensor) -> list[str]:
        targets = []
        for batch, source in enumerate(captions):
            events = []
            for index in event_mask[batch].nonzero(as_tuple=False).flatten().tolist():
                start, end = segments_seconds[batch, index].tolist()
                events.append((float(start), float(end), str(source[index])))
            events.sort(key=lambda item: (item[0], item[1], item[2]))
            fields = []
            for start, end, caption in events:
                start_bin = round(100 * max(0.0, min(1.0, start / float(durations[batch]))))
                end_bin = round(100 * max(0.0, min(1.0, end / float(durations[batch]))))
                fields.append(f"{EVENT} {START} {_time_token(start_bin)} {END} {_time_token(end_bin)} {CAPTION} {caption} {CLOSE}")
            targets.append(" ".join([*fields, END_EVENTS]))
        return targets

    def _labels(self, targets: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(targets, padding=True, truncation=False, return_tensors="pt")
        if encoded["input_ids"].size(1) > self.config.target_max_tokens:
            raise ValueError(f"Chronological target length {encoded['input_ids'].size(1)} exceeds target_max_tokens={self.config.target_max_tokens}; refusing to truncate events.")
        labels = encoded["input_ids"].to(device)
        labels[encoded["attention_mask"].to(device) == 0] = -100
        return labels

    def forward(self, batch: dict[str, object]) -> tuple[torch.Tensor, list[str]]:
        inputs, mask = self.conditioning(batch["visual_features"], batch["frame_times"], batch["frame_mask"], batch["duration"])  # type: ignore[arg-type]
        targets = self.target_text(batch["captions"], batch["segments_seconds"], batch["event_mask"], batch["duration"])  # type: ignore[arg-type]
        result = self.language_model(inputs_embeds=inputs, attention_mask=mask, labels=self._labels(targets, inputs.device), return_dict=True)
        if result.loss is None:
            raise RuntimeError("FLAN-T5 did not produce chronological sequence loss.")
        return result.loss, targets

    @torch.no_grad()
    def generate(self, features: torch.Tensor, times: torch.Tensor, frame_mask: torch.Tensor, durations: torch.Tensor) -> list[str]:
        inputs, mask = self.conditioning(features, times, frame_mask, durations)
        tokens = self.language_model.generate(inputs_embeds=inputs, attention_mask=mask, max_new_tokens=self.config.generation_max_tokens, min_new_tokens=self.config.generation_min_tokens, eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=self.config.repetition_penalty, no_repeat_ngram_size=self.config.no_repeat_ngram_size)
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=False)
