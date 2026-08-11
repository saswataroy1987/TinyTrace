from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tinytrace.config import TinyTraceConfig
from tinytrace.model import TinyTraceModel

from test_vision import FakeMobileCLIPBackbone


class PhaseADenseModelTests(unittest.TestCase):
    def make_config(self) -> TinyTraceConfig:
        return TinyTraceConfig(
            image_size=16,
            max_frames=4,
            d_model=24,
            num_layers=1,
            num_heads=4,
            max_text_len=16,
            max_caption_tokens=0,
            min_caption_tokens=0,
            max_events=1,
            max_generated_tokens=1,
            phase_a_dense_saliency=True,
            phase_a_bin_count=5,
            visual_encoder_chunk_size=2,
        )

    def test_128_frame_dense_contract_fits_position_budget(self) -> None:
        config = TinyTraceConfig(
            max_frames=128,
            max_text_len=256,
            max_caption_tokens=0,
            min_caption_tokens=0,
            max_generated_tokens=1,
            phase_a_dense_saliency=True,
        )

        self.assertEqual(config.maximum_training_sequence_length, 1538)
        self.assertEqual(config.maximum_inference_sequence_length, 1538)
        self.assertLessEqual(config.maximum_training_sequence_length, 2048)

    def test_deterministic_resample_matches_linear_interpolation_and_backpropagates(self) -> None:
        source = torch.randn(128, 24, requires_grad=True)
        expected = F.interpolate(
            source.transpose(0, 1).unsqueeze(0),
            size=75,
            mode="linear",
            align_corners=True,
        ).squeeze(0).transpose(0, 1)
        actual = TinyTraceModel._deterministic_linear_resample(source, output_length=75)

        # The native kernel and explicit matrix use slightly different
        # floating-point evaluation orders, while representing the same
        # align-corners interpolation weights.
        self.assertTrue(torch.allclose(actual, expected, atol=1e-4, rtol=1e-5))
        actual.square().mean().backward()
        self.assertIsNotNone(source.grad)
        self.assertTrue(torch.isfinite(source.grad).all())

    def test_dense_head_returns_finite_scores_and_only_dense_losses(self) -> None:
        config = self.make_config()
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        prompt = torch.tensor(
            [[config.bos_token_id, ord("q"), config.video_token_id]], dtype=torch.long
        )
        labels = prompt.clone()
        label_types = torch.full_like(prompt, -1)
        targets = torch.tensor([[0.0, 0.0, 3.7, 0.0, 2.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)

        output = model(
            torch.rand(1, 4, 3, 16, 16),
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
            prompt,
            labels=labels,
            label_types=label_types,
            prompt_lengths=torch.tensor([3]),
            saliency_targets=targets,
            saliency_target_mask=mask,
        )

        self.assertEqual(output.saliency_scores.shape, (1, 5))
        self.assertTrue(torch.isfinite(output.saliency_scores).all())
        self.assertTrue((output.saliency_scores >= 0).all())
        self.assertTrue((output.saliency_scores <= 4).all())
        self.assertEqual(
            set(output.loss_components),
            {"saliency_regression", "saliency_relevance", "saliency_ranking"},
        )
        self.assertIsNotNone(output.loss)
        output.loss.backward()
        self.assertIsNotNone(model.phase_a_saliency_head.weight.grad)
        self.assertIsNone(model.time_head.weight.grad)
        self.assertIsNone(model.score_head.weight.grad)
        self.assertIsNone(model.boundary_head.weight.grad)

    def test_dense_phase_refuses_autoregressive_highlight_generation(self) -> None:
        config = self.make_config()
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        with self.assertRaisesRegex(ValueError, "one forward pass"):
            model.generate(
                torch.rand(1, 4, 3, 16, 16),
                torch.arange(4, dtype=torch.float32).unsqueeze(0),
                torch.tensor([[config.bos_token_id, config.video_token_id]]),
                max_new_tokens=1,
                task_mode="highlight",
            )

    def test_score_aware_ranking_prefers_correct_logit_ordering(self) -> None:
        config = self.make_config()
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        targets = torch.tensor([[0.0, 1.0, 4.0, 2.0, 0.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        ordered_logits = torch.tensor([[0.1, 0.4, 1.2, 0.7, -0.2]])
        reversed_logits = torch.tensor([[1.2, 0.7, 0.1, 0.4, -0.2]])
        ordered_scores = config.phase_a_max_score * torch.sigmoid(ordered_logits)
        reversed_scores = config.phase_a_max_score * torch.sigmoid(reversed_logits)

        ordered_components, _ = model._phase_a_loss_components(
            ordered_logits,
            ordered_scores,
            targets,
            mask,
        )
        reversed_components, _ = model._phase_a_loss_components(
            reversed_logits,
            reversed_scores,
            targets,
            mask,
        )

        self.assertLess(
            float(ordered_components["saliency_ranking"]),
            float(reversed_components["saliency_ranking"]),
        )


if __name__ == "__main__":
    unittest.main()
