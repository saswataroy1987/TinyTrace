"""Frame-structured alternatives to v3's flattened cross-attention resampler."""

from __future__ import annotations

import torch
from torch import nn

from tinytrace.phase_b_v3.config import DirectMobileCLIPCaptionConfig
from tinytrace.phase_b_v3.direct_caption import EventPatchResampler


class StructuredFramePoolBridge(nn.Module):
    """B1: one token per selected frame, retaining frame order until FLAN-T5."""

    def __init__(self, config: DirectMobileCLIPCaptionConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.patch_projection = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, hidden_size))
        self.frame_positions = nn.Parameter(torch.empty(config.max_event_frames, hidden_size))
        self.output_norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.frame_positions, std=0.02)

    def forward(self, event_features: torch.Tensor, event_frame_mask: torch.Tensor) -> torch.Tensor:
        if event_features.ndim != 4 or event_features.shape[1:3] != (self.config.max_event_frames, self.config.patch_tokens):
            raise ValueError("Expected [events,max_event_frames,patch_tokens,feature_dim].")
        if event_frame_mask.shape != event_features.shape[:2] or not bool(event_frame_mask.any(dim=1).all()):
            raise ValueError("Every event must contain at least one selected frame.")
        # The only spatial aggregation in B1: deterministic mean over a
        # single frame's 64 patches. Frames never attend across one another.
        frame_tokens = self.patch_projection(event_features.float()).mean(dim=2)
        frame_tokens = self.output_norm(frame_tokens + self.frame_positions.unsqueeze(0))
        return frame_tokens.masked_fill(~event_frame_mask.unsqueeze(-1), 0)


class TemporalFrameBridge(StructuredFramePoolBridge):
    """B2: B1 frame tokens plus one lightweight temporal self-attention layer."""

    def __init__(self, config: DirectMobileCLIPCaptionConfig, hidden_size: int) -> None:
        super().__init__(config, hidden_size)
        if hidden_size % config.visual_heads:
            raise ValueError("FLAN hidden size must be divisible by visual_heads.")
        layer = nn.TransformerEncoderLayer(hidden_size, config.visual_heads, hidden_size * 4, config.visual_dropout, batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, num_layers=1)
        self.temporal_norm = nn.LayerNorm(hidden_size)

    def forward(self, event_features: torch.Tensor, event_frame_mask: torch.Tensor) -> torch.Tensor:
        frames = super().forward(event_features, event_frame_mask)
        encoded = self.temporal(frames, src_key_padding_mask=~event_frame_mask)
        return self.temporal_norm(encoded).masked_fill(~event_frame_mask.unsqueeze(-1), 0)


class NoResamplerBridge(nn.Module):
    """C1: retain every projected patch token until FLAN-T5 consumes it."""

    def __init__(self, config: DirectMobileCLIPCaptionConfig, hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.patch_projection = nn.Sequential(nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, hidden_size))
        self.spatial_positions = nn.Parameter(torch.empty(config.patch_tokens, hidden_size))
        self.temporal_positions = nn.Parameter(torch.empty(config.max_event_frames, hidden_size))
        self.output_norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.spatial_positions, std=0.02)
        nn.init.normal_(self.temporal_positions, std=0.02)

    def forward(self, event_features: torch.Tensor, event_frame_mask: torch.Tensor) -> torch.Tensor:
        if event_features.ndim != 4 or event_features.shape[1:3] != (self.config.max_event_frames, self.config.patch_tokens):
            raise ValueError("Expected [events,max_event_frames,patch_tokens,feature_dim].")
        if event_frame_mask.shape != event_features.shape[:2] or not bool(event_frame_mask.any(dim=1).all()):
            raise ValueError("Every event must contain at least one selected frame.")
        tokens = self.patch_projection(event_features.float())
        tokens = self.output_norm(tokens + self.spatial_positions.view(1, 1, self.config.patch_tokens, -1) + self.temporal_positions.view(1, self.config.max_event_frames, 1, -1))
        return tokens.reshape(tokens.size(0), self.config.max_event_frames * self.config.patch_tokens, -1)


def build_bridge(name: str, config: DirectMobileCLIPCaptionConfig, hidden_size: int) -> nn.Module:
    if name == "b0":
        return EventPatchResampler(config, hidden_size)
    if name == "b1":
        return StructuredFramePoolBridge(config, hidden_size)
    if name == "b2":
        return TemporalFrameBridge(config, hidden_size)
    if name == "c1":
        return NoResamplerBridge(config, hidden_size)
    raise ValueError("bridge name must be b0, b1, b2, or c1.")
