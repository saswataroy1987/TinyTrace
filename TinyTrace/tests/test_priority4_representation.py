import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.analyze_representation_profiles import build_report
from scripts.train_tinytrace import run_epoch
from test_vision import FakeMobileCLIPBackbone
from tinytrace.ablation import (
    decide_quality_efficiency_tradeoff,
    summarize_run_artifacts,
    validate_representation_ablation,
)
from tinytrace.config import TinyTraceConfig
from tinytrace.data import SyntheticTinyTraceDataset, sample_uniform_frame_times, tinytrace_collate_fn
from tinytrace.model import TinyTraceModel
from tinytrace.representation import (
    CAPTION_TOKEN_LADDER,
    FRAME_COUNT_LADDER,
    aggregate_caption_budget,
    temporal_coverage_report,
    visual_feature_diversity,
)
from tinytrace.serialization import caption_budget_metadata, serialize_example
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer
from tinytrace.training import TrainingConfig, build_named_optimizer


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


class Priority4RepresentationTests(unittest.TestCase):
    def test_frame_profiles_change_only_frame_count(self) -> None:
        reference = TinyTraceConfig.from_json(CONFIG_ROOT / "tinytrace_frames_08.json")
        for count in FRAME_COUNT_LADDER[1:]:
            candidate = TinyTraceConfig.from_json(CONFIG_ROOT / f"tinytrace_frames_{count:02d}.json")
            differences = {
                name
                for name, value in reference.to_dict().items()
                if value != candidate.to_dict()[name]
            }
            self.assertEqual(differences, {"max_frames"})

    def test_caption_profiles_change_only_budget_fields(self) -> None:
        reference = TinyTraceConfig.from_json(CONFIG_ROOT / "tinytrace_caption_020.json")
        self.assertEqual(reference.max_generated_tokens, 128)
        for budget, generation_budget in ((48, 202), (64, 250)):
            candidate = TinyTraceConfig.from_json(
                CONFIG_ROOT / f"tinytrace_caption_{budget:03d}.json"
            )
            differences = {
                name
                for name, value in reference.to_dict().items()
                if value != candidate.to_dict()[name]
            }
            self.assertEqual(differences, {"max_caption_tokens", "max_generated_tokens"})
            self.assertEqual(candidate.max_generated_tokens, generation_budget)
            self.assertEqual(candidate.max_generated_tokens, candidate.required_generation_token_budget)

    def test_representation_safety_limits_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame.*safety limit"):
            TinyTraceConfig(max_frames=129)
        with self.assertRaisesRegex(ValueError, "caption.*safety limit"):
            TinyTraceConfig(max_caption_tokens=65, max_generated_tokens=512)
        with self.assertRaisesRegex(ValueError, "generated.*safety limit"):
            TinyTraceConfig(max_generated_tokens=513)
        with self.assertRaisesRegex(ValueError, "Unknown TinyTrace config fields"):
            TinyTraceConfig.from_dict({"MAX_SUPPORTED_FRAMES": 64})

    def test_sampler_and_coverage_cover_short_and_boundary_durations(self) -> None:
        ordinary = sample_uniform_frame_times(10.0, 8)
        boundary = sample_uniform_frame_times(1.2500001, 12)
        short = sample_uniform_frame_times(0.1, 32)

        self.assertEqual(float(ordinary[0]), 0.0)
        self.assertAlmostEqual(float(ordinary[-1]), 9.75)
        self.assertTrue(torch.all(ordinary[1:] > ordinary[:-1]))
        self.assertTrue(torch.equal(boundary, sample_uniform_frame_times(1.2500001, 12)))
        self.assertEqual(short.tolist(), [0.0])

        report = temporal_coverage_report(0.1)
        self.assertEqual([row["requested_frames"] for row in report], list(FRAME_COUNT_LADDER))
        self.assertTrue(all(row["valid_frames"] == 1 for row in report))
        self.assertEqual([row["visual_time_prefix_tokens"] for row in report], [80, 120, 160, 240, 320])
        self.assertEqual([row["relative_decoded_frame_work"] for row in report], [1, 1.5, 2, 3, 4])

    def test_every_frame_profile_has_expected_patch_and_prefix_shapes(self) -> None:
        for count in FRAME_COUNT_LADDER:
            config = TinyTraceConfig(image_size=8, max_frames=count)
            model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone()).eval()
            frames = torch.rand(1, count, 3, 8, 8)
            times = torch.linspace(0, 1, count).unsqueeze(0)

            patches = model.visual_encoder.extract_patch_features(frames)
            compressed = model.visual_encoder.compress_patch_features(patches)
            prefix = model.build_visual_prefix(frames, times, visual_patch_features=patches)

            self.assertEqual(patches.shape, (1, count, 64, 1024))
            self.assertEqual(compressed.shape, (1, count, 4, config.d_model))
            self.assertEqual(prefix.shape, (1, count * 10, config.d_model))

    def test_feature_diversity_respects_frame_mask(self) -> None:
        patches = torch.tensor(
            [[[[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]]]
        )
        slots = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]]
        )
        report = visual_feature_diversity(
            patches,
            slots,
            torch.tensor([[True, True, False]]),
        )

        self.assertAlmostEqual(report["mean_frame_patch_summary_cosine_similarity"], 1.0)
        self.assertAlmostEqual(report["mean_compressed_slot_cosine_similarity"], 0.0)
        self.assertEqual(report["valid_frame_count"], 2)
        with self.assertRaisesRegex(ValueError, "frame_mask"):
            visual_feature_diversity(patches, slots, torch.ones(1, 3))

    def test_caption_truncation_is_explicit_and_serialization_remains_compatible(self) -> None:
        config = TinyTraceConfig(
            max_events=1,
            max_caption_tokens=5,
            min_caption_tokens=1,
            max_generated_tokens=25,
        )
        text = CharTokenizer(config.text_vocab_size)
        time = NumericTokenizer(config.time_vocab, width=6)
        score = NumericTokenizer(config.score_vocab, width=3)
        events = [{"timestamp": [0.0, 1.0], "score": [2.0], "caption": "abcdefgh"}]

        metadata = caption_budget_metadata(events, config, text)
        serialized = serialize_example(events, "find", config, text, time, score)

        self.assertEqual(metadata["truncated_event_count"], 1)
        self.assertEqual(metadata["retained_caption_tokens"], 5)
        self.assertEqual(len(serialized), 3)
        self.assertEqual(len(serialized[0]), len(serialized[1]))

    def test_dataset_collation_and_epoch_logs_include_caption_budget(self) -> None:
        config = TinyTraceConfig(
            image_size=8,
            max_frames=1,
            max_events=1,
            max_caption_tokens=5,
            min_caption_tokens=1,
            max_generated_tokens=25,
            d_model=24,
            num_heads=4,
            num_layers=1,
        )
        dataset = SyntheticTinyTraceDataset(config, size=1, seed=7)
        batch = tinytrace_collate_fn([dataset[0]])
        self.assertEqual(batch["caption_budget"][0]["truncated_event_count"], 1)

        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        optimizer = build_named_optimizer(model, TrainingConfig(learning_rate=1e-4))
        metrics, _ = run_epoch(
            model,
            DataLoader(dataset, batch_size=1, collate_fn=tinytrace_collate_fn),
            torch.device("cpu"),
            optimizer,
            gradient_clip=1.0,
            epoch=1,
            log_every=0,
            split="train",
        )
        self.assertEqual(metrics["caption_budget"]["truncated_event_count"], 1)

    def test_caption_aggregation_handles_empty_and_multiple_reports(self) -> None:
        empty = aggregate_caption_budget([])
        self.assertEqual(empty["truncated_event_rate"], 0.0)
        self.assertEqual(empty["retained_caption_token_rate"], 1.0)
        combined = aggregate_caption_budget(
            [
                {
                    "event_count": 2,
                    "truncated_event_count": 1,
                    "original_caption_tokens": 10,
                    "retained_caption_tokens": 8,
                },
                {
                    "event_count": 1,
                    "truncated_event_count": 0,
                    "original_caption_tokens": 4,
                    "retained_caption_tokens": 4,
                },
            ]
        )
        self.assertEqual(combined["event_count"], 3)
        self.assertAlmostEqual(combined["truncated_event_rate"], 1 / 3)

    def test_collation_remains_compatible_with_legacy_samples(self) -> None:
        legacy = {
            "frames": torch.rand(1, 3, 8, 8),
            "frame_times": torch.zeros(1),
            "token_ids": torch.tensor([1, 3, 2]),
            "label_types": torch.tensor([-1, -1, 0]),
            "events": [],
            "instruction": "test",
        }
        batch = tinytrace_collate_fn([legacy])
        self.assertFalse(batch["caption_budget"][0]["available"])
        aggregate = aggregate_caption_budget(batch["caption_budget"])
        self.assertEqual(aggregate["unavailable_sample_count"], 1)

    def test_ablation_validation_enforces_single_variable_and_sequence(self) -> None:
        frame8 = TinyTraceConfig.from_json(CONFIG_ROOT / "tinytrace_frames_08.json")
        frame12 = TinyTraceConfig.from_json(CONFIG_ROOT / "tinytrace_frames_12.json")
        self.assertEqual(set(validate_representation_ablation(frame8, frame12, "frame")), {"max_frames"})
        with self.assertRaisesRegex(ValueError, "sequential"):
            validate_representation_ablation(
                frame8,
                TinyTraceConfig.from_json(CONFIG_ROOT / "tinytrace_frames_16.json"),
                "frame",
            )
        with self.assertRaisesRegex(ValueError, "only max_frames"):
            validate_representation_ablation(frame8, TinyTraceConfig(max_frames=12, dropout=0.1), "frame")

        caption20 = TinyTraceConfig.from_json(CONFIG_ROOT / "tinytrace_caption_020.json")
        caption48 = TinyTraceConfig.from_json(CONFIG_ROOT / "tinytrace_caption_048.json")
        validate_representation_ablation(caption20, caption48, "caption")
        with self.assertRaisesRegex(ValueError, "minimally required"):
            validate_representation_ablation(
                caption20,
                TinyTraceConfig(max_caption_tokens=48, max_generated_tokens=203),
                "caption",
            )

    def test_analysis_uses_existing_inputs_and_marks_duration_proxy(self) -> None:
        report = build_report(
            [10.0],
            [{"events": [{"timestamp": [0.0, 2.0], "score": [1.0], "caption": "caption"}]}],
        )
        self.assertFalse(report["notes"]["acquisition_performed"])
        self.assertEqual(report["duration_sources"][0]["source"], "event_extent_proxy")
        self.assertEqual(set(report["caption_profiles"]), {str(value) for value in CAPTION_TOKEN_LADDER})

    def test_run_summary_requires_quality_training_and_inference_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_summary.json").write_text(
                json.dumps({"run_id": "run", "status": "completed", "final_structured_metrics": {"temporal_mean_iou": 0.5}}),
                encoding="utf-8",
            )
            (root / "history.json").write_text(
                json.dumps([{"train": {"elapsed_seconds": 1.0}, "validation": {}}]),
                encoding="utf-8",
            )
            incomplete = summarize_run_artifacts(root)
            self.assertFalse(incomplete["complete_for_decision"])
            self.assertEqual(decide_quality_efficiency_tradeoff(incomplete, incomplete)["status"], "incomplete")

            (root / "representation_benchmark.json").write_text(
                json.dumps({"measurement": {"median_end_to_end_ms": 10.0}}),
                encoding="utf-8",
            )
            complete = summarize_run_artifacts(root)
            self.assertTrue(complete["complete_for_decision"])
            decision = decide_quality_efficiency_tradeoff(complete, complete)
            self.assertEqual(decision["status"], "awaiting_review")
            self.assertEqual(decision["decision"], "not_automatically_selected")


if __name__ == "__main__":
    unittest.main()
