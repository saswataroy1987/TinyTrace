"""Caption model wrapper that preserves valid-frame masks for B1/B2 tokens."""

from __future__ import annotations

import torch

from tinytrace.phase_b_v3.direct_caption import DirectMobileCLIPCaptionModel


class BridgeAblationCaptionModel(DirectMobileCLIPCaptionModel):
    """Use the original v3 language policy with bridge-specific visual masks."""

    def __init__(self, *args, bridge_name: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bridge_name = bridge_name

    def conditioning(self, event_features: torch.Tensor, event_frame_mask: torch.Tensor, temporal_context: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.adapter(event_features, event_frame_mask)
        if self.bridge_name == "b0":
            visual_mask = torch.ones(visual.shape[:2], dtype=torch.long, device=visual.device)
        elif self.bridge_name == "c1":
            visual_mask = event_frame_mask[:, :, None].expand(-1, -1, self.config.patch_tokens).reshape(event_frame_mask.size(0), -1).to(dtype=torch.long)
        else:
            visual_mask = event_frame_mask.to(dtype=torch.long)
        if self.temporal_context_bridge is not None or temporal_context is not None:
            raise ValueError("Stage 1 context is intentionally outside bridge ablations.")
        instruction, instruction_mask = self._instruction_embeddings(visual.size(0), visual.device)
        return torch.cat((visual, instruction), dim=1), torch.cat((visual_mask, instruction_mask), dim=1)
