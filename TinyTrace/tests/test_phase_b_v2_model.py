from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from tinytrace.phase_b_v2.caption import FlanT5Captioner, pool_event_features
from tinytrace.phase_b_v2.config import PhaseBV2Config
from tinytrace.phase_b_v2.model import PhaseBV2Model
from tinytrace.phase_b_v2.metrics import localization_metrics, matched_caption_metrics
from tinytrace.phase_b_v2.temporal import centre_duration_to_segment, hungarian_match, temporal_iou


class FakeTokenizer:
    eos_token_id = 1

    def __call__(self, captions, padding, truncation, max_length=None, return_tensors=None, add_special_tokens=True):
        ids = [[2 + index for index, _ in enumerate(text.split())][:max_length] + ([self.eos_token_id] if add_special_tokens else []) for text in captions]
        width = max(map(len, ids)) if ids else 1
        input_ids = torch.tensor([row + [0] * (width - len(row)) for row in ids])
        mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": mask}

    def batch_decode(self, tokens, skip_special_tokens=True):
        return ["generated caption" for _ in range(tokens.size(0))]


class FakeT5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=16)
        self.decoder = nn.Module()
        self.decoder.block = nn.ModuleList([nn.Linear(16, 16), nn.Linear(16, 16)])
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([nn.Linear(16, 16), nn.Linear(16, 16)])
        self.output = nn.Linear(16, 1)

    def forward(self, inputs_embeds, attention_mask, labels, return_dict):
        return SimpleNamespace(loss=self.output(inputs_embeds).mean() + labels[labels != -100].float().mean() * 0)

    def generate(self, inputs_embeds, attention_mask, **kwargs):
        return torch.tensor([[2, 1]] * inputs_embeds.size(0), device=inputs_embeds.device)


class PhaseBV2ModelTests(unittest.TestCase):
    def config(self, **overrides) -> PhaseBV2Config:
        return PhaseBV2Config(d_model=32, temporal_heads=4, temporal_layers=1, event_decoder_layers=1, event_queries=3, **overrides)

    def batch(self) -> dict[str, object]:
        return {"visual_features": torch.randn(2, 4, 64, 1024), "frame_times": torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 0.0]]), "frame_mask": torch.tensor([[True, True, True, True], [True, True, False, False]]), "segments": torch.tensor([[[0.1, 0.4], [0.5, 0.9]], [[0.2, 0.8], [0.0, 0.0]]]), "event_mask": torch.tensor([[True, True], [True, False]]), "captions": [["one event", "another event"], ["last event", ""]]}

    def test_assignment_and_segment_conversion_are_valid(self) -> None:
        queries, targets = hungarian_match(torch.tensor([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]))
        self.assertEqual(set(zip(queries.tolist(), targets.tolist())), {(0, 1), (1, 0)})
        segments = centre_duration_to_segment(torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
        self.assertTrue(torch.all(segments[:, 1] > segments[:, 0]))
        self.assertTrue(torch.all((segments >= 0) & (segments <= 1)))

    def test_detector_forward_loss_and_padding_mask(self) -> None:
        model = PhaseBV2Model(self.config())
        outputs, loss = model.forward_localization(self.batch())
        self.assertEqual(outputs["segments"].shape, (2, 3, 2))
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        self.assertIsNotNone(model.detector.projection[0].weight.grad)

    def test_event_pooling_uses_segment_and_nearest_valid_frame(self) -> None:
        features = torch.tensor([[[1.0], [3.0], [100.0]]])
        times = torch.tensor([[0.0, 1.0, 0.0]])
        mask = torch.tensor([[True, True, False]])
        pooled, valid = pool_event_features(features, times, mask, torch.tensor([[[0.4, 0.6]]]))
        self.assertTrue(valid.item())
        self.assertLess(float(pooled.item()), 10.0)

    def test_event_pooling_preserves_ordered_conditioning_tokens(self) -> None:
        features = torch.tensor([[[1.0], [3.0], [5.0], [7.0]]])
        times = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, True, True, True]])
        pooled, valid = pool_event_features(features, times, mask, torch.tensor([[[0.0, 1.0]]]), token_count=2)
        self.assertTrue(valid.item())
        self.assertEqual(tuple(pooled.shape), (1, 1, 2, 1))
        self.assertLess(float(pooled[0, 0, 0, 0]), float(pooled[0, 0, 1, 0]))

    def test_caption_bridge_uses_ignore_index_and_final_decoder_policy(self) -> None:
        config = self.config(stage="caption", conditioning_tokens=3, train_flan_encoder_layers=1, train_flan_decoder_layers=1, caption_max_tokens=2)
        captioner = FlanT5Captioner(config, FakeT5(), FakeTokenizer())
        model = PhaseBV2Model(config, captioner)
        model.freeze_temporal_encoder()
        loss, report = model.forward_caption(self.batch())
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(report["truncated_caption_count"], 1)
        self.assertTrue(any(parameter.requires_grad for parameter in captioner.model.decoder.block[-1].parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in captioner.model.decoder.block[0].parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in captioner.model.encoder.block[-1].parameters()))
        loss.backward()
        self.assertIsNotNone(captioner.bridge[0].weight.grad)

    def test_joint_path_keeps_match_caption_alignment_and_predicts_without_targets(self) -> None:
        config = self.config(stage="joint", conditioning_tokens=2, train_flan_decoder_layers=1)
        model = PhaseBV2Model(config, FlanT5Captioner(config, FakeT5(), FakeTokenizer()))
        local, caption, _ = model.forward_joint(self.batch(), ground_truth_segment_ratio=0.0)
        self.assertTrue(torch.isfinite(local.total + caption))
        events = model.predict_events(self.batch(), threshold=0.0, overlap_threshold=1.0)
        self.assertEqual(len(events), 2)
        self.assertTrue(all("caption" in event for video in events for event in video))

    def test_metrics_accept_variable_event_counts_across_batches(self) -> None:
        predictions = [[{"start": 0.1, "end": 0.4, "score": 0.9, "caption": "first event"}], [{"start": 0.5, "end": 0.8, "score": 0.8, "caption": "second event"}]]
        targets = [torch.tensor([[0.1, 0.4], [0.6, 0.9]]), torch.tensor([[0.5, 0.8]])]
        temporal = localization_metrics(predictions, targets, [10.0, 20.0])
        captions = matched_caption_metrics(predictions, targets, [["first event", "missed"], ["second event"]])
        self.assertEqual(temporal["target_events"], 3.0)
        self.assertEqual(temporal["matched_events"], 2.0)
        self.assertEqual(captions["caption_count"], 2.0)


if __name__ == "__main__":
    unittest.main()
