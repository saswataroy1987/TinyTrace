import unittest

import torch
import torch.nn as nn

from tinytrace.config import TinyTraceConfig
from tinytrace.model import TinyTraceModel
from tinytrace.vision import MobileCLIPSpatialEncoder, SlotCompressor


class FakeMobileCLIPBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 64, kernel_size=1)
        self.conv_exp = nn.Conv2d(64, 1024, kernel_size=1)

    def forward_embeddings(self, frames: torch.Tensor) -> torch.Tensor:
        return self.stem(frames)

    def forward_tokens(self, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.adaptive_avg_pool2d(features, (8, 8))

class MobileCLIPSpatialEncoderTests(unittest.TestCase):
    def test_spatial_tokens_have_expected_shape_and_backbone_is_frozen(self) -> None:
        config = TinyTraceConfig()
        backbone = FakeMobileCLIPBackbone()
        encoder = MobileCLIPSpatialEncoder(config, backbone=backbone)

        tokens = encoder(torch.rand(2, 3, 96, 96))

        self.assertEqual(tokens.shape, (2, 64, 1024))
        self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))

    def test_preprocess_preserves_aspect_then_center_crops_without_normalization(self) -> None:
        config = TinyTraceConfig(image_size=32)
        encoder = MobileCLIPSpatialEncoder(config, backbone=FakeMobileCLIPBackbone())
        frames = torch.ones(1, 3, 20, 40)

        processed = encoder.preprocess(frames)

        self.assertEqual(processed.shape, (1, 3, 32, 32))
        self.assertTrue(torch.equal(processed, torch.ones_like(processed)))

    def test_frozen_backbone_stays_in_eval_mode(self) -> None:
        encoder = MobileCLIPSpatialEncoder(TinyTraceConfig(), backbone=FakeMobileCLIPBackbone())

        encoder.train()

        self.assertFalse(encoder.backbone.training)

    def test_partial_unfreeze_produces_only_conv_exp_gradients(self) -> None:
        backbone = FakeMobileCLIPBackbone()
        encoder = MobileCLIPSpatialEncoder(TinyTraceConfig(), backbone=backbone)
        encoder.set_trainable(True, strategy="conv_exp")
        encoder.train()

        encoder(torch.rand(1, 3, 32, 32)).sum().backward()

        self.assertIsNone(backbone.stem.weight.grad)
        self.assertIsNotNone(backbone.conv_exp.weight.grad)
        self.assertFalse(backbone.training)
        self.assertTrue(backbone.conv_exp.training)

    def test_slot_compressor_shape(self) -> None:
        compressor = SlotCompressor(input_dim=1024, output_dim=192, num_slots=4)

        compressed = compressor(torch.rand(2, 64, 1024))

        self.assertEqual(compressed.shape, (2, 4, 192))

    def test_model_builds_interleaved_visual_and_discrete_time_prefix(self) -> None:
        config = TinyTraceConfig(max_frames=2)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        frames = torch.rand(1, 2, 3, 96, 96)
        frame_times = torch.tensor([[0.0, 12.5]])

        prefix = model.build_visual_prefix(frames, frame_times)
        time_ids = model._encode_frame_time_ids(frame_times)

        self.assertEqual(prefix.shape, (1, 20, config.d_model))
        self.assertEqual(time_ids.shape, (1, 2, 6))
        decoded = "".join(config.time_vocab[index] for index in time_ids[0, 1].tolist())
        self.assertEqual(decoded, "0012.5")

    def test_positional_encoding_rejects_runtime_overflow(self) -> None:
        config = TinyTraceConfig(
            max_frames=1,
            max_events=1,
            max_generated_tokens=40,
            max_position_embeddings=100,
        )
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        with self.assertRaisesRegex(ValueError, "exceeds positional capacity"):
            model.position(torch.zeros(1, 101, config.d_model))

    def test_generation_rejects_runtime_budget_overflow(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        with self.assertRaisesRegex(ValueError, "configured limit"):
            model.generate(
                torch.rand(1, 1, 3, 16, 16),
                torch.zeros(1, 1),
                torch.tensor([[config.bos_token_id, config.video_token_id]]),
                max_new_tokens=config.max_generated_tokens + 1,
            )

    def test_cached_mobileclip_features_preserve_prefix(self) -> None:
        config = TinyTraceConfig(max_frames=2)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone()).eval()
        frames = torch.rand(1, 2, 3, 96, 96)
        frame_times = torch.tensor([[0.0, 1.0]])

        patch_features = model.visual_encoder.extract_patch_features(frames)
        direct = model.build_visual_prefix(frames, frame_times)
        cached = model.build_visual_prefix(
            frames,
            frame_times,
            visual_patch_features=patch_features,
        )

        self.assertTrue(torch.equal(direct, cached))


if __name__ == "__main__":
    unittest.main()
