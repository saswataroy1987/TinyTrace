from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .caption import FlanT5Captioner, flatten_caption_events, pool_event_features
from .config import PhaseBV2Config
from .temporal import LocalizationLoss, TemporalEventDetector, filter_events, localization_loss


class PhaseBV2Model(nn.Module):
    """Stage-aware V2 model; MobileCLIP is intentionally absent because inputs are cached."""

    def __init__(self, config: PhaseBV2Config, captioner: FlanT5Captioner | None = None) -> None:
        super().__init__()
        self.config = config
        self.detector = TemporalEventDetector(config)
        self.captioner = captioner
        if config.stage in {"caption", "joint"} and captioner is None:
            raise ValueError("Caption/joint-stage model requires a FLAN-T5 captioner.")

    @classmethod
    def for_language_stage(cls, config: PhaseBV2Config) -> "PhaseBV2Model":
        if config.stage not in {"caption", "joint"}:
            raise ValueError("Language-stage construction requires config.stage='caption' or 'joint'.")
        return cls(config, FlanT5Captioner.from_pretrained(config))

    def freeze_temporal_encoder(self) -> None:
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        self.detector.eval()

    def forward_localization(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], LocalizationLoss]:
        outputs = self.detector(batch["visual_features"], batch["frame_times"], batch["frame_mask"])
        loss = localization_loss(outputs, batch["segments"], batch["event_mask"], self.config)
        return outputs, loss

    def forward_caption(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, int]]:
        if self.captioner is None:
            raise RuntimeError("Captioner is not initialized for this model.")
        temporal = self.detector.encode(batch["visual_features"], batch["frame_times"], batch["frame_mask"])
        pooled, valid = pool_event_features(temporal, batch["frame_times"], batch["frame_mask"], batch["segments"], self.config.conditioning_tokens)
        active_mask = batch["event_mask"] & valid
        flattened = flatten_caption_events(pooled, active_mask, batch["captions"], self.config.conditioning_tokens)
        return self.captioner(flattened.conditioning, flattened.captions)

    def forward_joint(self, batch: dict[str, Any], ground_truth_segment_ratio: float | None = None) -> tuple[LocalizationLoss, torch.Tensor, dict[str, int]]:
        """Train localization and captions together on one-to-one matched queries.

        Caption targets stay aligned to the matcher target index. During the
        planned transition, each matched query pools either its target segment
        or its own predicted segment; this ratio is saved in the run config.
        """
        if self.captioner is None:
            raise RuntimeError("Captioner is not initialized for this model.")
        outputs, local = self.forward_localization(batch)
        ratio = self.config.joint_ground_truth_segment_ratio if ground_truth_segment_ratio is None else ground_truth_segment_ratio
        if not 0 <= ratio <= 1:
            raise ValueError("ground_truth_segment_ratio must be in [0, 1].")
        selected = torch.zeros_like(outputs["segments"])
        selected_mask = torch.zeros(outputs["segments"].shape[:2], dtype=torch.bool, device=outputs["segments"].device)
        captions: list[list[str]] = [["" for _ in range(outputs["segments"].size(1))] for _ in range(outputs["segments"].size(0))]
        for batch_index, (query_indices, target_indices) in enumerate(local.matches):
            if not query_indices.numel():
                continue
            use_truth = torch.rand(query_indices.numel(), device=selected.device) < ratio
            predicted = outputs["segments"][batch_index, query_indices]
            truth = batch["segments"][batch_index, target_indices]
            selected[batch_index, query_indices] = torch.where(use_truth[:, None], truth, predicted)
            selected_mask[batch_index, query_indices] = True
            for query_index, target_index in zip(query_indices.tolist(), target_indices.tolist()):
                captions[batch_index][query_index] = batch["captions"][batch_index][target_index]
        pooled, valid = pool_event_features(outputs["temporal_features"], batch["frame_times"], batch["frame_mask"], selected, self.config.conditioning_tokens)
        flattened = flatten_caption_events(pooled, selected_mask & valid, captions, self.config.conditioning_tokens)
        caption_loss, report = self.captioner(flattened.conditioning, flattened.captions)
        return local, caption_loss, report

    @torch.no_grad()
    def generate_ground_truth_events(self, batch: dict[str, Any]) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        if self.captioner is None:
            raise RuntimeError("Captioner is not initialized for this model.")
        temporal = self.detector.encode(batch["visual_features"], batch["frame_times"], batch["frame_mask"])
        pooled, valid = pool_event_features(temporal, batch["frame_times"], batch["frame_mask"], batch["segments"], self.config.conditioning_tokens)
        flattened = flatten_caption_events(pooled, batch["event_mask"] & valid, batch["captions"], self.config.conditioning_tokens)
        return self.captioner.generate(flattened.conditioning), flattened.batch_indices, flattened.event_indices

    @torch.no_grad()
    def predict_events(self, batch: dict[str, Any], *, threshold: float, overlap_threshold: float) -> list[list[dict[str, object]]]:
        """Run the final detector-to-caption graph without ground-truth events."""
        if self.captioner is None:
            raise RuntimeError("Captioner is not initialized for this model.")
        outputs = self.detector(batch["visual_features"], batch["frame_times"], batch["frame_mask"])
        results: list[list[dict[str, object]]] = []
        for batch_index in range(outputs["segments"].size(0)):
            events = filter_events(outputs["segments"][batch_index], outputs["event_logits"][batch_index], threshold, overlap_threshold)
            if not events:
                results.append([])
                continue
            segments = torch.tensor([[item["start"], item["end"]] for item in events], dtype=outputs["segments"].dtype, device=outputs["segments"].device).unsqueeze(0)
            pooled, _ = pool_event_features(outputs["temporal_features"][batch_index : batch_index + 1], batch["frame_times"][batch_index : batch_index + 1], batch["frame_mask"][batch_index : batch_index + 1], segments, self.config.conditioning_tokens)
            captions = self.captioner.generate(pooled[0])
            results.append([{**event, "caption": caption} for event, caption in zip(events, captions)])
        return results
