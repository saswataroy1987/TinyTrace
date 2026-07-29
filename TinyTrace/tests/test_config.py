import unittest
from pathlib import Path

from tinytrace.config import TinyTraceConfig
from tinytrace.tokenizers import NumericTokenizer


class TinyTraceConfigTests(unittest.TestCase):
    def test_default_config_satisfies_sequence_contract(self) -> None:
        config = TinyTraceConfig()

        self.assertLessEqual(
            config.maximum_training_sequence_length,
            config.max_position_embeddings,
        )
        self.assertGreater(config.maximum_event_token_count, config.max_caption_tokens)

    def test_baseline_json_preserves_documented_eight_frame_reference(self) -> None:
        baseline = TinyTraceConfig.from_json(
            Path(__file__).resolve().parents[1] / "configs" / "tinytrace_baseline.json"
        )

        self.assertEqual(TinyTraceConfig().max_frames, 8)
        self.assertEqual(baseline.max_frames, 8)

    def test_decoder_dimensions_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by num_heads"):
            TinyTraceConfig(d_model=192, num_heads=5)

        with self.assertRaisesRegex(ValueError, "must be even"):
            TinyTraceConfig(d_model=191, num_heads=1)

    def test_caption_limits_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            TinyTraceConfig(min_caption_tokens=21, max_caption_tokens=20)

    def test_numeric_vocabulary_order_is_frozen(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_vocab"):
            TinyTraceConfig(time_vocab=("<sync>", "<sep>", "1", "0"))

    def test_preprocessing_statistics_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            TinyTraceConfig(mobileclip_image_std=(1.0, 0.0, 1.0))

    def test_positional_capacity_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds positional capacity"):
            TinyTraceConfig(max_frames=32, max_position_embeddings=128)

    def test_checkpoint_digest_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "64-character"):
            TinyTraceConfig(mobileclip_checkpoint_sha256="not-a-sha256")

    def test_generation_budget_covers_all_configured_events(self) -> None:
        config = TinyTraceConfig()

        self.assertEqual(
            config.required_generation_token_budget,
            config.max_events * config.maximum_event_token_count + 1,
        )
        self.assertGreaterEqual(
            config.max_generated_tokens,
            config.required_generation_token_budget,
        )

        with self.assertRaisesRegex(ValueError, "maximum structured output"):
            TinyTraceConfig(
                max_generated_tokens=config.required_generation_token_budget - 1,
            )

    def test_inference_sequence_must_fit_positional_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "inference sequence length"):
            TinyTraceConfig(
                max_position_embeddings=300,
                max_generated_tokens=200,
                max_events=1,
            )

    def test_loss_weights_must_be_finite_and_nonnegative(self) -> None:
        for invalid_weight in (-0.1, float("inf"), float("nan")):
            with self.subTest(invalid_weight=invalid_weight):
                with self.assertRaisesRegex(ValueError, "time_loss_weight"):
                    TinyTraceConfig(time_loss_weight=invalid_weight)

    def test_legacy_preprocessing_configuration_is_migrated(self) -> None:
        legacy_normalized = TinyTraceConfig().to_dict()
        legacy_normalized.pop("mobileclip_apply_normalization")

        restored_normalized = TinyTraceConfig.from_dict(legacy_normalized)
        restored_official = TinyTraceConfig.from_dict({"max_frames": 8})

        self.assertTrue(restored_normalized.mobileclip_apply_normalization)
        self.assertFalse(restored_official.mobileclip_apply_normalization)

    def test_invalid_configuration_payload_types_fail_cleanly(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload must be an object"):
            TinyTraceConfig.from_dict([])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "mobileclip_model_name"):
            TinyTraceConfig(mobileclip_model_name=None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "channel values"):
            TinyTraceConfig(mobileclip_image_mean=None)  # type: ignore[arg-type]

    def test_numeric_tokenizer_rejects_unrepresentable_values(self) -> None:
        tokenizer = NumericTokenizer(TinyTraceConfig().time_vocab, width=6)

        for value in (-1.0, 10000.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite|fixed width"):
                    tokenizer.encode([value])


if __name__ == "__main__":
    unittest.main()
