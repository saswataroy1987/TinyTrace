from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TinyTraceConfig
from .vision import MobileCLIPSpatialEncoder, SlotCompressor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional capacity {self.pe.size(1)}."
            )
        return x + self.pe[:, : x.size(1)]


class LightweightVisualEncoder(nn.Module):
    def __init__(
        self,
        config: TinyTraceConfig,
        mobileclip_backbone: nn.Module | None = None,
        load_pretrained_visual: bool = True,
    ) -> None:
        super().__init__()
        self.mobileclip = MobileCLIPSpatialEncoder(
            config,
            backbone=mobileclip_backbone,
            load_pretrained=load_pretrained_visual,
        )
        self.compressor = SlotCompressor(
            input_dim=config.visual_hidden_dim,
            output_dim=config.d_model,
            num_slots=config.compressed_visual_tokens,
        )
        self.config = config

    def set_mobileclip_trainable(self, trainable: bool, strategy: str = "full") -> None:
        self.mobileclip.set_trainable(trainable, strategy=strategy)

    def extract_patch_features(self, frames: torch.Tensor) -> torch.Tensor:
        batch, num_frames, channels, height, width = frames.shape
        flattened = frames.reshape(batch * num_frames, channels, height, width)
        chunk_size = self.config.visual_encoder_chunk_size
        patches = torch.cat(
            [self.mobileclip(chunk) for chunk in flattened.split(chunk_size, dim=0)],
            dim=0,
        )
        return patches.view(batch, num_frames, patches.size(1), patches.size(2))

    def compress_patch_features(self, patch_features: torch.Tensor) -> torch.Tensor:
        if patch_features.ndim != 4:
            raise ValueError(
                "MobileCLIP patch features must have shape "
                "[batch, num_frames, num_patches, channels]."
            )
        batch, num_frames, num_patches, channels = patch_features.shape
        if channels != self.config.visual_hidden_dim:
            raise ValueError(
                f"Expected {self.config.visual_hidden_dim} MobileCLIP channels, received {channels}."
            )
        flattened = patch_features.reshape(batch * num_frames, num_patches, channels)
        compressed = self.compressor(flattened)
        return compressed.view(batch, num_frames, self.config.compressed_visual_tokens, -1)

    def forward(
        self,
        frames: torch.Tensor,
        patch_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patch_features is None:
            patch_features = self.extract_patch_features(frames)
        elif patch_features.shape[:2] != frames.shape[:2]:
            raise ValueError(
                "Cached MobileCLIP features do not match the frame batch/time dimensions: "
                f"{tuple(patch_features.shape[:2])} vs. {tuple(frames.shape[:2])}."
            )
        return self.compress_patch_features(patch_features)


class DecoderBlock(nn.Module):
    def __init__(self, config: TinyTraceConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.num_heads, dropout=config.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * config.mlp_ratio),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * config.mlp_ratio, config.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass
class TinyTraceOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    text_logits: torch.Tensor
    time_logits: torch.Tensor
    score_logits: torch.Tensor
    boundary_logits: torch.Tensor | None = None
    saliency_logits: torch.Tensor | None = None
    saliency_scores: torch.Tensor | None = None
    loss_components: dict[str, torch.Tensor] = field(default_factory=dict)
    weighted_loss_components: dict[str, torch.Tensor] = field(default_factory=dict)
    target_counts: dict[str, torch.Tensor] = field(default_factory=dict)


class TinyTraceModel(nn.Module):
    def __init__(
        self,
        config: TinyTraceConfig,
        mobileclip_backbone: nn.Module | None = None,
        load_pretrained_visual: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.visual_encoder = LightweightVisualEncoder(
            config,
            mobileclip_backbone=mobileclip_backbone,
            load_pretrained_visual=load_pretrained_visual,
        )
        self.text_embeddings = nn.Embedding(config.text_vocab_size, config.d_model)
        self.sync_embedding = nn.Parameter(torch.randn(config.d_model))
        self.time_embeddings = nn.Embedding(len(config.time_vocab), config.d_model)
        self.score_embeddings = nn.Embedding(len(config.score_vocab), config.d_model)
        self.token_type_embeddings = nn.Embedding(4, config.d_model)
        self.position = PositionalEncoding(config.d_model, max_len=config.max_position_embeddings)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)

        self.text_head = nn.Linear(config.d_model, config.text_vocab_size + 1)
        self.time_head = nn.Linear(config.d_model, len(config.time_vocab))
        self.score_head = nn.Linear(config.d_model, len(config.score_vocab))
        self.boundary_head = nn.Linear(config.d_model, 2)
        if config.phase_a_dense_saliency:
            self.phase_a_frame_projection = nn.Linear(config.d_model, config.d_model)
            self.phase_a_context_projection = nn.Linear(config.d_model, config.d_model)
            self.phase_a_bin_embeddings = nn.Embedding(config.phase_a_bin_count, config.d_model)
            self.phase_a_fusion = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
                nn.GELU(),
            )
            self.phase_a_saliency_head = nn.Linear(config.d_model, 1)

    def set_visual_encoder_trainable(self, trainable: bool, strategy: str = "full") -> None:
        self.visual_encoder.set_mobileclip_trainable(trainable, strategy=strategy)

    def _loss_component_weight(self, name: str) -> float:
        if name == "time":
            return self.config.time_loss_weight
        if name == "score":
            return self.config.score_loss_weight
        if name == "text":
            return self.config.caption_loss_weight
        if name in {"time_sync", "score_sync", "caption_sync"}:
            return self.config.sync_loss_weight
        if name == "boundary":
            return self.config.boundary_loss_weight
        if name == "saliency_regression":
            return self.config.saliency_regression_loss_weight
        if name == "saliency_relevance":
            return self.config.saliency_relevance_loss_weight
        if name == "saliency_ranking":
            return self.config.saliency_ranking_loss_weight
        raise KeyError(f"Unknown TinyTrace loss component: {name}")

    def _combine_loss_components(
        self,
        components: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        active_weights = {
            name: self._loss_component_weight(name)
            for name in components
            if self._loss_component_weight(name) > 0
        }
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            return None, {}
        # Each task loss is already mean-reduced.  A weighted average keeps the
        # gradient scale stable when a phase enables a different number of
        # heads; summing these means caused Phase A to clip every optimizer
        # step in the failed run.
        weighted = {
            name: value * self._loss_component_weight(name) / total_weight
            for name, value in components.items()
        }
        return sum(weighted.values()), weighted

    def _phase_a_saliency(
        self,
        hidden: torch.Tensor,
        token_ids: torch.Tensor,
        prompt_lengths: torch.Tensor | None,
        frame_mask: torch.Tensor | None,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.config.phase_a_dense_saliency:
            raise RuntimeError("The dense Phase-A saliency head is disabled in this config.")
        if prompt_lengths is None:
            prompt_lengths = token_ids.ne(self.config.pad_token_id).sum(dim=1)
        if prompt_lengths.shape != (token_ids.size(0),):
            raise ValueError("prompt_lengths must have shape [batch].")
        if (prompt_lengths < 1).any() or (prompt_lengths > token_ids.size(1)).any():
            raise ValueError("prompt_lengths must identify a non-empty prompt inside token_ids.")

        tokens_per_frame = self.config.compressed_visual_tokens + self.config.time_tokens_per_frame
        visual_length = num_frames * tokens_per_frame
        frame_hidden = hidden[:, :visual_length].reshape(
            hidden.size(0), num_frames, tokens_per_frame, self.config.d_model
        )[:, :, : self.config.compressed_visual_tokens].mean(dim=2)
        if frame_mask is None:
            frame_mask = torch.ones(
                hidden.size(0), num_frames, dtype=torch.bool, device=hidden.device
            )

        aligned_frames = []
        for sample_index in range(hidden.size(0)):
            valid_frames = frame_hidden[sample_index, frame_mask[sample_index].bool()]
            if valid_frames.size(0) < 1:
                raise ValueError("Each Phase-A sample must contain at least one valid frame.")
            aligned = self._deterministic_linear_resample(
                valid_frames,
                output_length=self.config.phase_a_bin_count,
            )
            aligned_frames.append(aligned)
        aligned_frame_hidden = torch.stack(aligned_frames, dim=0)

        prompt_last_indices = visual_length + prompt_lengths.to(hidden.device) - 1
        batch_indices = torch.arange(hidden.size(0), device=hidden.device)
        prompt_context = hidden[batch_indices, prompt_last_indices]
        bin_ids = torch.arange(self.config.phase_a_bin_count, device=hidden.device)
        fused = (
            self.phase_a_frame_projection(aligned_frame_hidden)
            + self.phase_a_context_projection(prompt_context).unsqueeze(1)
            + self.phase_a_bin_embeddings(bin_ids).unsqueeze(0)
        )
        fused = fused + self.phase_a_fusion(fused)
        logits = self.phase_a_saliency_head(fused).squeeze(-1)
        scores = self.config.phase_a_max_score * torch.sigmoid(logits)
        return logits, scores

    @staticmethod
    def _deterministic_linear_resample(
        sequence: torch.Tensor,
        *,
        output_length: int,
    ) -> torch.Tensor:
        """Align ``[time, channels]`` without CUDA's nondeterministic upsample backward.

        This is mathematically equivalent to one-dimensional linear
        interpolation with ``align_corners=True``. A fixed interpolation
        matrix makes both the forward and gradient use deterministic matrix
        multiplication when deterministic cuBLAS is configured.
        """

        if sequence.ndim != 2 or sequence.size(0) < 1:
            raise ValueError("sequence must have shape [positive_time, channels].")
        if output_length < 1:
            raise ValueError("output_length must be positive.")
        input_length = sequence.size(0)
        if input_length == output_length:
            return sequence

        positions = torch.linspace(
            0.0,
            float(input_length - 1),
            output_length,
            dtype=torch.float32,
            device=sequence.device,
        )
        lower = positions.floor().to(torch.long)
        upper = positions.ceil().to(torch.long).clamp(max=input_length - 1)
        upper_weight = positions - lower.to(positions.dtype)
        lower_weight = 1.0 - upper_weight
        interpolation = (
            F.one_hot(lower, num_classes=input_length).to(torch.float32)
            * lower_weight.unsqueeze(1)
            + F.one_hot(upper, num_classes=input_length).to(torch.float32)
            * upper_weight.unsqueeze(1)
        ).to(dtype=sequence.dtype)
        return interpolation @ sequence

    def _phase_a_loss_components(
        self,
        logits: torch.Tensor,
        scores: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if targets.shape != logits.shape or target_mask.shape != logits.shape:
            raise ValueError(
                "saliency_targets and saliency_target_mask must match [batch, phase_a_bin_count]."
            )
        mask = target_mask.bool()
        if not mask.any():
            raise ValueError("Dense Phase-A batches must contain at least one supervised bin.")
        if not torch.isfinite(targets[mask]).all():
            raise ValueError("saliency_targets must be finite in supervised bins.")
        if (targets[mask] < 0).any() or (targets[mask] > self.config.phase_a_max_score).any():
            raise ValueError("saliency_targets are outside the configured score range.")

        per_bin = F.smooth_l1_loss(scores, targets, reduction="none", beta=0.25)
        positive = mask & targets.gt(0)
        negative = mask & ~targets.gt(0)
        balanced_terms = []
        if positive.any():
            balanced_terms.append(per_bin[positive].mean())
        if negative.any():
            balanced_terms.append(per_bin[negative].mean())
        components: dict[str, torch.Tensor] = {
            "saliency_regression": sum(balanced_terms) / len(balanced_terms)
        }
        counts: dict[str, torch.Tensor] = {
            "saliency_regression": mask.sum().detach()
        }

        relevant = targets.ge(self.config.phase_a_positive_threshold).to(logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits[mask],
            relevant[mask],
            pos_weight=torch.tensor(
                self.config.saliency_positive_weight,
                dtype=logits.dtype,
                device=logits.device,
            ),
        )
        components["saliency_relevance"] = bce
        counts["saliency_relevance"] = mask.sum().detach()

        ranking_terms = []
        for sample_index in range(logits.size(0)):
            high = logits[sample_index][mask[sample_index] & relevant[sample_index].bool()]
            low = logits[sample_index][mask[sample_index] & ~relevant[sample_index].bool()]
            if high.numel() and low.numel():
                ranking_terms.append(F.softplus(-(high[:, None] - low[None, :])).mean())
        if ranking_terms:
            components["saliency_ranking"] = torch.stack(ranking_terms).mean()
            counts["saliency_ranking"] = torch.tensor(
                len(ranking_terms), dtype=torch.long, device=logits.device
            )
        return components, counts

    def _encode_frame_time_ids(self, frame_times: torch.Tensor) -> torch.Tensor:
        if frame_times.ndim != 2:
            raise ValueError("frame_times must have shape [batch, num_frames].")

        token_to_id = {token: index for index, token in enumerate(self.config.time_vocab)}
        rows: list[list[list[int]]] = []
        for sample_times in frame_times.detach().cpu().tolist():
            sample_rows = []
            for value in sample_times:
                formatted = format(float(value), "0>6.1f")
                if len(formatted) != self.config.time_tokens_per_frame:
                    raise ValueError(
                        f"Frame timestamp {value} cannot be represented as a "
                        f"{self.config.time_tokens_per_frame}-token TRACE timestamp."
                    )
                try:
                    sample_rows.append([token_to_id[character] for character in formatted])
                except KeyError as exc:
                    raise ValueError(f"Unsupported character in frame timestamp {formatted!r}.") from exc
            rows.append(sample_rows)
        return torch.tensor(rows, dtype=torch.long, device=frame_times.device)

    def build_visual_prefix(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        visual_patch_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        visual_tokens = self.visual_encoder(frames, patch_features=visual_patch_features)
        if visual_tokens.shape[:2] != frame_times.shape:
            raise ValueError(
                "Frame/time shape mismatch: visual encoder produced "
                f"{tuple(visual_tokens.shape[:2])}, frame_times has {tuple(frame_times.shape)}."
            )

        time_ids = self._encode_frame_time_ids(frame_times)
        time_tokens = self.time_embeddings(time_ids)
        per_frame_tokens = torch.cat([visual_tokens, time_tokens], dim=2)
        return per_frame_tokens.flatten(1, 2)

    def _build_key_padding_mask(
        self,
        token_ids: torch.Tensor,
        frame_mask: torch.Tensor | None,
        num_frames: int,
    ) -> torch.Tensor | None:
        if frame_mask is None:
            frame_mask = torch.ones(
                token_ids.size(0),
                num_frames,
                dtype=torch.bool,
                device=token_ids.device,
            )
        if frame_mask.shape != (token_ids.size(0), num_frames):
            raise ValueError(
                f"frame_mask must have shape {(token_ids.size(0), num_frames)}, "
                f"received {tuple(frame_mask.shape)}."
            )

        tokens_per_frame = self.config.compressed_visual_tokens + self.config.time_tokens_per_frame
        visual_padding = (~frame_mask.bool()).repeat_interleave(tokens_per_frame, dim=1)
        text_padding = token_ids.eq(self.config.pad_token_id)
        combined = torch.cat([visual_padding, text_padding], dim=1)
        return combined if combined.any() else None

    def _expected_phase_lengths(self) -> tuple[int, int]:
        time_len = (
            self.config.timestamp_value_count * 6
            + max(0, self.config.timestamp_value_count - 1)
        )
        score_len = (
            self.config.score_value_count * 3
            + max(0, self.config.score_value_count - 1)
        )
        return time_len, score_len

    def _numeric_format_mask(self, device: torch.device, mode: str, position: int, vocab_size: int) -> torch.Tensor:
        allowed = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        digit_slice = slice(2, 12)
        dot_idx = 12
        sep_idx = 1

        if mode == "time":
            if position in {0, 1, 2, 3, 5, 7, 8, 9, 10, 12}:
                allowed[digit_slice] = True
            elif position in {4, 11}:
                allowed[dot_idx] = True
            elif position == 6:
                allowed[sep_idx] = True
        else:
            if position in {0, 2}:
                allowed[digit_slice] = True
            elif position == 1:
                allowed[dot_idx] = True

        return allowed

    def _infer_token_type_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_types = torch.zeros_like(token_ids)
        token_types[token_ids == self.config.sync_token_id] = 3
        token_types[(token_ids >= self.config.time_token_base) & (token_ids < self.config.score_token_base)] = 1
        token_types[token_ids >= self.config.score_token_base] = 2
        return token_types

    def _decode_numeric_values(self, ids: list[int], vocab: tuple[str, ...]) -> list[float]:
        id_to_token = {index: token for index, token in enumerate(vocab)}
        values: list[float] = []
        current: list[str] = []
        width = 6 if vocab == self.config.time_vocab else 3

        def flush_current() -> None:
            nonlocal current
            if not current:
                return
            token_string = "".join(current)
            if len(token_string) != width or token_string.count(".") != 1:
                current = []
                return
            try:
                values.append(float(token_string))
            except ValueError:
                pass
            current = []

        for idx in ids:
            token = id_to_token[idx]
            if token == "<sep>":
                flush_current()
            elif token != "<sync>":
                current.append(token)
        flush_current()
        return values

    def _encode_numeric_values(self, values: list[float], mode: str) -> list[int]:
        width = 6 if mode == "time" else 3
        vocab = self.config.time_vocab if mode == "time" else self.config.score_vocab
        token_to_id = {token: index for index, token in enumerate(vocab)}
        encoded: list[int] = []
        for index, value in enumerate(values):
            for char in format(float(value), f"0>{width}.1f"):
                encoded.append(token_to_id[char])
            if index < len(values) - 1:
                encoded.append(token_to_id["<sep>"])
        encoded.append(token_to_id["<sync>"])
        return encoded

    def _constrain_time_ids(self, time_ids: list[int], clip_end: float) -> list[int]:
        try:
            values = self._decode_numeric_values(time_ids, self.config.time_vocab)
        except (KeyError, ValueError):
            # Generation artifacts must preserve malformed numeric output for
            # defensive parsing and diagnostics rather than crashing here.
            return time_ids
        if not values:
            return time_ids
        clipped = [min(max(value, 0.0), clip_end) for value in values]
        while len(clipped) < self.config.timestamp_value_count:
            clipped.append(clipped[-1] if clipped else 0.0)
        if len(clipped) >= 2:
            clipped[1] = max(clipped[0], clipped[1])
        return self._encode_numeric_values(clipped[: self.config.timestamp_value_count], mode="time")

    def embed_mixed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_type_ids = self._infer_token_type_ids(token_ids)
        embeddings = []
        for row, type_row in zip(token_ids, token_type_ids):
            row_embeddings = []
            for token_id, token_type in zip(row.tolist(), type_row.tolist()):
                if token_id == self.config.sync_token_id:
                    base_embedding = self.sync_embedding
                elif token_id >= self.config.score_token_base:
                    base_embedding = self.score_embeddings.weight[token_id - self.config.score_token_base]
                elif token_id >= self.config.time_token_base:
                    base_embedding = self.time_embeddings.weight[token_id - self.config.time_token_base]
                else:
                    base_embedding = self.text_embeddings.weight[token_id]
                row_embeddings.append(base_embedding + self.token_type_embeddings.weight[token_type])
            embeddings.append(torch.stack(row_embeddings, dim=0))
        return torch.stack(embeddings, dim=0)

    def forward(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        token_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        label_types: torch.Tensor | None = None,
        frame_mask: torch.Tensor | None = None,
        visual_patch_features: torch.Tensor | None = None,
        prompt_lengths: torch.Tensor | None = None,
        saliency_targets: torch.Tensor | None = None,
        saliency_target_mask: torch.Tensor | None = None,
    ) -> TinyTraceOutput:
        if (labels is None) != (label_types is None):
            raise ValueError("labels and label_types must either both be provided or both be omitted.")
        if labels is not None and labels.shape != token_ids.shape:
            raise ValueError("labels must have the same shape as token_ids.")
        if label_types is not None and label_types.shape != token_ids.shape:
            raise ValueError("label_types must have the same shape as token_ids.")
        if (saliency_targets is None) != (saliency_target_mask is None):
            raise ValueError(
                "saliency_targets and saliency_target_mask must either both be provided or both omitted."
            )
        visual_tokens = self.build_visual_prefix(
            frames,
            frame_times,
            visual_patch_features=visual_patch_features,
        )
        token_embeddings = self.embed_mixed_tokens(token_ids)
        x = torch.cat([visual_tokens, token_embeddings], dim=1)
        x = self.dropout(self.position(x))

        seq_len = x.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        key_padding_mask = self._build_key_padding_mask(token_ids, frame_mask, frames.size(1))
        for block in self.blocks:
            x = block(x, attn_mask, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        text_logits = self.text_head(x)
        time_logits = self.time_head(x)
        score_logits = self.score_head(x)
        boundary_logits = self.boundary_head(x)
        saliency_logits = None
        saliency_scores = None
        if self.config.phase_a_dense_saliency:
            saliency_logits, saliency_scores = self._phase_a_saliency(
                x,
                token_ids,
                prompt_lengths=prompt_lengths,
                frame_mask=frame_mask,
                num_frames=frames.size(1),
            )

        full_logits = torch.full(
            (x.size(0), x.size(1), self.config.total_token_vocab),
            float("-inf"),
            device=x.device,
        )
        full_logits[:, :, : self.config.text_vocab_size] = text_logits[:, :, : self.config.text_vocab_size]
        full_logits[:, :, self.config.sync_token_id] = text_logits[:, :, self.config.text_vocab_size]
        full_logits[:, :, self.config.time_token_base : self.config.score_token_base] = time_logits
        full_logits[:, :, self.config.score_token_base :] = score_logits

        loss = None
        loss_components: dict[str, torch.Tensor] = {}
        weighted_loss_components: dict[str, torch.Tensor] = {}
        target_counts: dict[str, torch.Tensor] = {}
        if saliency_targets is not None and saliency_target_mask is not None:
            if saliency_logits is None or saliency_scores is None:
                raise ValueError("Dense saliency targets require phase_a_dense_saliency=true.")
            phase_components, phase_counts = self._phase_a_loss_components(
                saliency_logits,
                saliency_scores,
                saliency_targets,
                saliency_target_mask,
            )
            loss_components.update(phase_components)
            target_counts.update(phase_counts)

        if labels is not None and label_types is not None:
            prompt_len = visual_tokens.size(1)
            hidden_text = text_logits[:, prompt_len:-1]
            hidden_time = time_logits[:, prompt_len:-1]
            hidden_score = score_logits[:, prompt_len:-1]

            target_tokens = labels[:, 1:]
            target_types = label_types[:, 1:]
            valid_mask = target_types >= 0

            text_mask = (target_types == 0) & valid_mask
            if text_mask.any():
                loss_components["text"] = F.cross_entropy(hidden_text[text_mask], target_tokens[text_mask])
                target_counts["text"] = text_mask.sum().detach()

            caption_sync_mask = (target_types == 1) & valid_mask
            if caption_sync_mask.any():
                sync_targets = torch.full_like(target_tokens[caption_sync_mask], self.config.text_vocab_size)
                loss_components["caption_sync"] = F.cross_entropy(hidden_text[caption_sync_mask], sync_targets)
                target_counts["caption_sync"] = caption_sync_mask.sum().detach()

            time_mask = (target_types == 2) & valid_mask
            if time_mask.any():
                loss_components["time"] = F.cross_entropy(
                    hidden_time[time_mask],
                    target_tokens[time_mask] - self.config.time_token_base,
                )
                target_counts["time"] = time_mask.sum().detach()

            time_sync_mask = (target_types == 4) & valid_mask
            if time_sync_mask.any():
                time_sync_targets = torch.zeros_like(target_tokens[time_sync_mask])
                loss_components["time_sync"] = F.cross_entropy(hidden_time[time_sync_mask], time_sync_targets)
                target_counts["time_sync"] = time_sync_mask.sum().detach()

            score_mask = (target_types == 3) & valid_mask
            if score_mask.any():
                loss_components["score"] = F.cross_entropy(
                    hidden_score[score_mask],
                    target_tokens[score_mask] - self.config.score_token_base,
                )
                target_counts["score"] = score_mask.sum().detach()

            score_sync_mask = ((target_types == 5) | (target_types == 6)) & valid_mask
            if score_sync_mask.any():
                score_sync_targets = torch.zeros_like(target_tokens[score_sync_mask])
                loss_components["score_sync"] = F.cross_entropy(hidden_score[score_sync_mask], score_sync_targets)
                target_counts["score_sync"] = score_sync_mask.sum().detach()

            # Boundary labels are attached to caption sync for caption tasks
            # and score sync for highlight-only tasks. A dedicated binary head
            # avoids comparing unrelated text/time logit scales.
            previous_types = label_types[:, :-1]
            boundary_mask = (
                ((previous_types == 1) | (previous_types == 6)) & valid_mask
            )
            if boundary_mask.any():
                boundary_tokens = target_tokens[boundary_mask]
                boundary_targets = torch.where(
                    boundary_tokens == self.config.eos_token_id,
                    torch.zeros_like(boundary_tokens),
                    torch.ones_like(boundary_tokens),
                )
                loss_components["boundary"] = F.cross_entropy(
                    boundary_logits[:, prompt_len:-1][boundary_mask],
                    boundary_targets,
                )
                target_counts["boundary"] = boundary_mask.sum().detach()

        loss, weighted_loss_components = self._combine_loss_components(loss_components)

        return TinyTraceOutput(
            loss=loss,
            logits=full_logits,
            text_logits=text_logits,
            time_logits=time_logits,
            score_logits=score_logits,
            boundary_logits=boundary_logits,
            saliency_logits=saliency_logits,
            saliency_scores=saliency_scores,
            loss_components=loss_components,
            weighted_loss_components=weighted_loss_components,
            target_counts=target_counts,
        )

    @torch.no_grad()
    def generate(
        self,
        frames: torch.Tensor,
        frame_times: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        frame_mask: torch.Tensor | None = None,
        visual_patch_features: torch.Tensor | None = None,
        return_metadata: bool = False,
        task_mode: str = "caption",
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        if task_mode not in {"caption", "highlight"}:
            raise ValueError("task_mode must be either 'caption' or 'highlight'.")
        if task_mode == "highlight" and self.config.phase_a_dense_saliency:
            raise ValueError(
                "Dense Phase A predicts all saliency bins in one forward pass; "
                "autoregressive generate() is not part of this protocol."
            )
        if prompt_ids.size(0) != 1:
            raise ValueError(
                "TinyTrace generation currently supports batch size 1. "
                "Per-sequence adaptive head state is not implemented yet."
            )
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        if max_new_tokens > self.config.max_generated_tokens:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} exceeds the configured limit "
                f"{self.config.max_generated_tokens}."
            )
        generated = prompt_ids.clone()
        mode = "time"
        phase_token_count = 0
        event_count = 0
        phase_start_index = generated.size(1)
        clip_end = float(frame_times.max().item())
        min_time_tokens, min_score_tokens = self._expected_phase_lengths()
        termination_reason = "max_tokens"
        forced_caption_termination = False
        for _ in range(max_new_tokens):
            output = self.forward(
                frames,
                frame_times,
                generated,
                frame_mask=frame_mask,
                visual_patch_features=visual_patch_features,
            )
            if mode == "boundary":
                if event_count >= self.config.max_events:
                    next_token = torch.full(
                        (generated.size(0), 1),
                        self.config.eos_token_id,
                        dtype=torch.long,
                        device=generated.device,
                    )
                else:
                    boundary_choice = torch.argmax(
                        output.boundary_logits[:, -1, :], dim=-1, keepdim=True
                    )
                    if int(boundary_choice.item()) == 0:
                        next_token = torch.full_like(boundary_choice, self.config.eos_token_id)
                    else:
                        time_logits = output.time_logits[:, -1, :].clone()
                        allowed = self._numeric_format_mask(
                            time_logits.device, "time", position=0,
                            vocab_size=time_logits.size(-1),
                        )
                        time_logits[:, ~allowed] = float("-inf")
                        best_time_id = torch.argmax(time_logits, dim=-1, keepdim=True)
                        next_token = best_time_id + self.config.time_token_base
                        mode = "time"
                        phase_token_count = 1

                generated = torch.cat([generated, next_token], dim=1)
                if next_token[0, 0].item() == self.config.eos_token_id:
                    termination_reason = "eos"
                    break
                phase_start_index = generated.size(1) - 1
                continue

            if mode == "time":
                next_logits = output.time_logits[:, -1, :].clone()
            elif mode == "score":
                next_logits = output.score_logits[:, -1, :].clone()
            else:
                next_logits = output.text_logits[:, -1, :].clone()

            if mode == "time":
                if phase_token_count == min_time_tokens:
                    next_token = torch.full(
                        (generated.size(0), 1),
                        self.config.sync_token_id,
                        dtype=torch.long,
                        device=generated.device,
                    )
                else:
                    next_logits[:, 0] = float("-inf")
                    allowed = self._numeric_format_mask(next_logits.device, "time", phase_token_count, next_logits.size(-1))
                    next_logits[:, ~allowed] = float("-inf")
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                    next_token = next_token + self.config.time_token_base
            elif mode == "score":
                if phase_token_count == min_score_tokens:
                    next_token = torch.full(
                        (generated.size(0), 1),
                        self.config.sync_token_id,
                        dtype=torch.long,
                        device=generated.device,
                    )
                else:
                    next_logits[:, 0] = float("-inf")
                    allowed = self._numeric_format_mask(next_logits.device, "score", phase_token_count, next_logits.size(-1))
                    next_logits[:, ~allowed] = float("-inf")
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                    next_token = next_token + self.config.score_token_base
            else:
                if phase_token_count == 0:
                    next_logits[:, self.config.eos_token_id] = float("-inf")
                if phase_token_count < self.config.min_caption_tokens:
                    next_logits[:, self.config.text_vocab_size] = float("-inf")
                if phase_token_count >= self.config.max_caption_tokens:
                    next_logits[:, : self.config.text_vocab_size] = float("-inf")
                    forced_caption_termination = True
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                next_token = torch.where(
                    next_token == self.config.text_vocab_size,
                    torch.full_like(next_token, self.config.sync_token_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=1)

            token_value = next_token[0, 0].item()
            if token_value == self.config.sync_token_id:
                if mode == "time":
                    time_phase = generated[0, phase_start_index:].tolist()
                    constrained = self._constrain_time_ids(
                        [
                            0 if token == self.config.sync_token_id else token - self.config.time_token_base
                            for token in time_phase
                        ],
                        clip_end=clip_end,
                    )
                    constrained_tokens = [
                        self.config.sync_token_id if token == 0 else token + self.config.time_token_base
                        for token in constrained
                    ]
                    generated = torch.cat(
                        [
                            generated[:, :phase_start_index],
                            torch.tensor(constrained_tokens, dtype=generated.dtype, device=generated.device).unsqueeze(0),
                        ],
                        dim=1,
                    )
                    mode = "score"
                elif mode == "score":
                    if task_mode == "highlight":
                        event_count += 1
                        mode = "boundary"
                    else:
                        mode = "caption"
                else:
                    event_count += 1
                    mode = "boundary"
                phase_token_count = 0
                phase_start_index = generated.size(1)
            elif token_value == self.config.eos_token_id:
                termination_reason = "eos"
                break
            else:
                phase_token_count += 1
        if not return_metadata:
            return generated
        return generated, {
            "termination_reason": termination_reason,
            "forced_caption_termination": forced_caption_termination,
            "generated_token_count": generated.size(1) - prompt_ids.size(1),
            "completed_event_count": event_count,
            "final_mode": mode,
        }
