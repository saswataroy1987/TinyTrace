import json
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch

from tinytrace.config import TinyTraceConfig
from tinytrace.data import (
    JsonTinyTraceDataset,
    SyntheticTinyTraceDataset,
    sample_uniform_frame_times,
    tinytrace_collate_fn,
)
from tinytrace.model import TinyTraceModel
from tinytrace.tokenizers import NumericTokenizer

from test_vision import FakeMobileCLIPBackbone


class StabilityTests(unittest.TestCase):
    def test_uniform_frame_sampling_is_deterministic_and_bounded(self) -> None:
        first = sample_uniform_frame_times(duration=10.0, requested_frames=4)
        second = sample_uniform_frame_times(duration=10.0, requested_frames=4)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first.shape, (4,))
        self.assertEqual(first[0].item(), 0.0)
        self.assertAlmostEqual(first[-1].item(), 9.75)
        self.assertTrue((first[1:] > first[:-1]).all())

    def test_short_video_sampling_avoids_duplicate_frames(self) -> None:
        frame_times = sample_uniform_frame_times(duration=0.1, requested_frames=8)

        self.assertEqual(frame_times.tolist(), [0.0])

    def test_uniform_frame_sampling_rejects_invalid_inputs(self) -> None:
        for duration in (0.0, -1.0, float("nan")):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "duration"):
                    sample_uniform_frame_times(duration=duration, requested_frames=8)

        with self.assertRaisesRegex(ValueError, "requested_frames"):
            sample_uniform_frame_times(duration=1.0, requested_frames=0)

    def test_batched_decoder_uses_n_minus_one_endpoint_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "samples.json"
            annotations.write_text("[]", encoding="utf-8")
            dataset = JsonTinyTraceDataset(
                annotations,
                TinyTraceConfig(image_size=2, max_frames=4),
            )
            expected_bytes = bytes(4 * 2 * 2 * 3)
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=expected_bytes, stderr=b""
            )
            with patch("tinytrace.data.subprocess.run", return_value=completed) as run:
                frames = dataset._read_uniform_frames_batched(
                    "video.mp4", duration=10.0, num_frames=4
                )

            command = run.call_args.args[0]
            filter_value = command[command.index("-vf") + 1]
            self.assertIn("fps=0.307692307692", filter_value)
            self.assertEqual(frames.shape, (4, 3, 2, 2))

    def test_json_dataset_decodes_lazily_and_reuses_frame_cache(self) -> None:
        class CountingDataset(JsonTinyTraceDataset):
            decode_calls = 0

            def _load_video_frames(self, video_path: str, num_frames: int):
                self.decode_calls += 1
                return (
                    torch.ones(num_frames, 3, self.config.image_size, self.config.image_size),
                    torch.arange(num_frames, dtype=torch.float32),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "samples.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "video_path": str(root / "video.mp4"),
                            "num_frames": 2,
                            "events": [],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            dataset = CountingDataset(
                annotations,
                TinyTraceConfig(image_size=16, max_frames=2),
                frame_cache_dir=root / "cache",
                allow_random_frames=False,
            )

            self.assertEqual(dataset.decode_calls, 0)
            first = dataset[0]
            self.assertEqual(dataset.decode_calls, 1)
            second = dataset[0]
            self.assertEqual(dataset.decode_calls, 1)
            self.assertTrue(torch.equal(first["frames"], second["frames"]))
            self.assertEqual(len(list((root / "cache").glob("*.pt"))), 1)

    def test_real_json_dataset_rejects_missing_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            annotations = Path(directory) / "samples.json"
            annotations.write_text(json.dumps([{"events": []}]), encoding="utf-8")
            dataset = JsonTinyTraceDataset(
                annotations,
                TinyTraceConfig(image_size=16, max_frames=1),
                allow_random_frames=False,
            )

            with self.assertRaisesRegex(ValueError, "random fallback frames are disabled"):
                dataset[0]

    def test_invalid_frame_cache_is_regenerated(self) -> None:
        class CountingDataset(JsonTinyTraceDataset):
            decode_calls = 0

            def _load_video_frames(self, video_path: str, num_frames: int):
                self.decode_calls += 1
                return (
                    torch.ones(num_frames, 3, self.config.image_size, self.config.image_size),
                    torch.arange(num_frames, dtype=torch.float32),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "samples.json"
            annotations.write_text(
                json.dumps([{"video_path": str(root / "video.mp4"), "num_frames": 2, "events": []}]),
                encoding="utf-8",
            )
            dataset = CountingDataset(
                annotations,
                TinyTraceConfig(image_size=16, max_frames=2),
                frame_cache_dir=root / "cache",
                allow_random_frames=False,
            )

            dataset[0]
            cache_path = next((root / "cache").glob("*.pt"))
            torch.save({"format_version": -1, "frames": torch.zeros(1)}, cache_path)

            regenerated = dataset[0]

            self.assertEqual(dataset.decode_calls, 2)
            self.assertEqual(regenerated["frames"].shape, (2, 3, 16, 16))

    def test_concurrent_frame_cache_writers_leave_one_valid_entry(self) -> None:
        barrier = threading.Barrier(2)

        class ConcurrentDataset(JsonTinyTraceDataset):
            def _load_video_frames(self, video_path: str, num_frames: int):
                barrier.wait(timeout=5)
                return (
                    torch.ones(num_frames, 3, self.config.image_size, self.config.image_size),
                    torch.arange(num_frames, dtype=torch.float32),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations = root / "samples.json"
            annotations.write_text(
                json.dumps([{"video_path": str(root / "video.mp4"), "num_frames": 2, "events": []}]),
                encoding="utf-8",
            )
            dataset = ConcurrentDataset(
                annotations,
                TinyTraceConfig(image_size=16, max_frames=2),
                frame_cache_dir=root / "cache",
                allow_random_frames=False,
            )

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: dataset[0], range(2)))

            cache_files = list((root / "cache").glob("*.pt"))
            temporary_files = list((root / "cache").glob("*.tmp"))
            self.assertEqual(len(cache_files), 1)
            self.assertEqual(temporary_files, [])
            self.assertTrue(torch.equal(results[0]["frames"], results[1]["frames"]))
            self.assertIsNotNone(dataset._read_frame_cache(cache_files[0], num_frames=2))

    def test_json_dataset_rejects_non_object_samples_and_excess_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.json"
            malformed.write_text(json.dumps([1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be an object"):
                JsonTinyTraceDataset(malformed, TinyTraceConfig(max_frames=2))

            excessive = root / "excessive.json"
            excessive.write_text(json.dumps([{"num_frames": 3, "events": []}]), encoding="utf-8")
            dataset = JsonTinyTraceDataset(excessive, TinyTraceConfig(max_frames=2))
            with self.assertRaisesRegex(ValueError, "configured max_frames=2"):
                dataset[0]

    def test_optional_video_validation_skips_probe_failures(self) -> None:
        class ProbeFailureDataset(JsonTinyTraceDataset):
            @staticmethod
            def _probe_video_duration(video_path: str) -> float:
                raise subprocess.CalledProcessError(1, ["ffprobe", video_path])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_video = root / "invalid.mp4"
            invalid_video.touch()
            annotations = root / "samples.json"
            annotations.write_text(
                json.dumps(
                    [
                        {"video_path": str(invalid_video), "events": []},
                        {"num_frames": 1, "events": []},
                    ]
                ),
                encoding="utf-8",
            )

            dataset = ProbeFailureDataset(
                annotations,
                TinyTraceConfig(max_frames=2),
                validate_videos_on_init=True,
            )

            self.assertEqual(len(dataset), 1)

    def test_strict_dense_phase_a_rejects_media_shorter_than_bin_window(self) -> None:
        class TruncatedDenseDataset(JsonTinyTraceDataset):
            @staticmethod
            def _probe_video_duration(video_path: str) -> float:
                return 4.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "truncated.mp4"
            video.touch()
            annotations = root / "phase-a.json"
            annotations.write_text(
                json.dumps(
                    [
                        {
                            "source_id": 1,
                            "video_path": str(video),
                            "instruction": "Find the action",
                            "query": "action",
                            "task_mode": "highlight",
                            "duration_seconds": 6.0,
                            "saliency_bin_count": 3,
                            "saliency_bin_size_seconds": 2.0,
                            "dense_saliency_scores": [0.0, 3.0, 0.0],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            config = TinyTraceConfig(
                max_frames=2,
                max_caption_tokens=0,
                min_caption_tokens=0,
                max_events=1,
                max_generated_tokens=1,
                phase_a_dense_saliency=True,
                phase_a_bin_count=3,
                phase_a_bin_size_seconds=2.0,
            )

            with self.assertRaisesRegex(ValueError, "shorter than the required 5.500s"):
                TruncatedDenseDataset(
                    annotations,
                    config,
                    validate_videos_on_init=True,
                    strict_media_validation=True,
                )

    def test_relative_precomputed_frames_are_resolved_and_variable_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_path = root / "frames.pt"
            torch.save(torch.ones(1, 3, 8, 8), frames_path)
            annotations = root / "samples.json"
            annotations.write_text(
                json.dumps([{"frames_path": "frames.pt", "num_frames": 2, "events": []}]),
                encoding="utf-8",
            )

            dataset = JsonTinyTraceDataset(
                annotations,
                TinyTraceConfig(image_size=8, max_frames=2),
                allow_random_frames=False,
            )
            sample = dataset[0]

            self.assertEqual(sample["frames"].shape, (1, 3, 8, 8))
            self.assertEqual(sample["frame_times"].tolist(), [0.0])

    def test_numeric_tokenizer_round_trip(self) -> None:
        config = TinyTraceConfig()
        tokenizer = NumericTokenizer(config.time_vocab, width=6)

        encoded = tokenizer.encode([0.0, 12.5])

        self.assertEqual(tokenizer.decode_sequence(encoded), [0.0, 12.5])

    def test_variable_frame_collation_adds_padding_mask(self) -> None:
        def sample(frame_count: int, token_count: int) -> dict:
            return {
                "frames": torch.rand(frame_count, 3, 16, 16),
                "frame_times": torch.arange(frame_count, dtype=torch.float32),
                "token_ids": torch.arange(1, token_count + 1),
                "label_types": torch.zeros(token_count, dtype=torch.long),
                "events": [],
                "instruction": "test",
            }

        batch = tinytrace_collate_fn([sample(2, 4), sample(4, 6)])

        self.assertEqual(batch["frames"].shape, (2, 4, 3, 16, 16))
        self.assertEqual(batch["frame_times"].shape, (2, 4))
        self.assertEqual(
            batch["frame_mask"].tolist(),
            [[True, True, False, False], [True, True, True, True]],
        )

    def test_attention_padding_mask_covers_each_padded_frame_group(self) -> None:
        config = TinyTraceConfig(max_frames=2)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())
        token_ids = torch.tensor([[config.bos_token_id, config.video_token_id, config.pad_token_id]])
        frame_mask = torch.tensor([[True, False]])

        mask = model._build_key_padding_mask(token_ids, frame_mask, num_frames=2)

        tokens_per_frame = config.compressed_visual_tokens + config.time_tokens_per_frame
        self.assertFalse(mask[0, :tokens_per_frame].any())
        self.assertTrue(mask[0, tokens_per_frame : 2 * tokens_per_frame].all())
        self.assertTrue(mask[0, -1])

    def test_generation_rejects_unsupported_batches(self) -> None:
        config = TinyTraceConfig(max_frames=1)
        model = TinyTraceModel(config, mobileclip_backbone=FakeMobileCLIPBackbone())

        with self.assertRaisesRegex(ValueError, "batch size 1"):
            model.generate(
                torch.rand(2, 1, 3, 16, 16),
                torch.zeros(2, 1),
                torch.tensor([[1, 3], [1, 3]]),
                max_new_tokens=1,
            )

    def test_synthetic_samples_have_distinct_deterministic_visual_patterns(self) -> None:
        config = TinyTraceConfig(image_size=32, max_frames=2)
        first = SyntheticTinyTraceDataset(config, size=2, seed=7)
        second = SyntheticTinyTraceDataset(config, size=2, seed=7)

        self.assertTrue(torch.equal(first[0]["frames"], second[0]["frames"]))
        self.assertFalse(torch.equal(first[0]["frames"], first[1]["frames"]))
        self.assertEqual(first[0]["instruction"], first[1]["instruction"])


if __name__ == "__main__":
    unittest.main()
