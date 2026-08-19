from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.prepare_phase_b_activitynet import _build_events, _convert_split
from tinytrace.config import TinyTraceConfig
from tinytrace.model import TinyTraceModel
from tinytrace.training import load_model_state_compat

from test_vision import FakeMobileCLIPBackbone


class ActivityNetPhaseBTests(unittest.TestCase):
    def test_build_events_pairs_sentences_with_timestamps(self) -> None:
        events = _build_events(
            sentences=["first step", "second step"],
            timestamps=[[5.0, 10.0], [1.0, 3.0]],
            max_events=6,
            default_score=1.0,
        )

        self.assertEqual(
            events,
            [
                {"timestamp": [1.0, 3.0], "score": [1.0], "caption": "second step"},
                {"timestamp": [5.0, 10.0], "score": [1.0], "caption": "first step"},
            ],
        )

    def test_convert_split_skips_missing_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos_root = root / "videos"
            videos_root.mkdir()
            (videos_root / "v_keep.mp4").write_bytes(b"fake")
            payload = {
                "v_keep": {
                    "duration": 12.0,
                    "sentences": ["keep this"],
                    "timestamps": [[0.0, 2.0]],
                },
                "v_missing": {
                    "duration": 9.0,
                    "sentences": ["missing"],
                    "timestamps": [[1.0, 3.0]],
                },
            }
            items, summary = _convert_split(
                payload=payload,
                split_name="train",
                video_index={"v_keep": videos_root / "v_keep.mp4"},
                videos_root=videos_root,
                max_events=6,
                default_score=1.0,
            )

        self.assertEqual(summary["kept"], 1)
        self.assertEqual(summary["skipped_missing_video"], 1)
        self.assertEqual(items[0]["video_path"], "videos/v_keep.mp4")
        self.assertEqual(items[0]["task_mode"], "caption")

    def test_convert_split_skips_invalid_videos_when_validator_rejects_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos_root = root / "videos"
            videos_root.mkdir()
            video_path = videos_root / "v_bad.mp4"
            video_path.write_bytes(b"not-a-video")
            payload = {
                "v_bad": {
                    "duration": 12.0,
                    "sentences": ["bad video"],
                    "timestamps": [[0.0, 2.0]],
                }
            }
            items, summary = _convert_split(
                payload=payload,
                split_name="train",
                video_index={"v_bad": video_path},
                videos_root=videos_root,
                max_events=6,
                default_score=1.0,
                media_validator=lambda _path: "unreadable video",
            )

        self.assertEqual(items, [])
        self.assertEqual(summary["kept"], 0)
        self.assertEqual(summary["skipped_invalid_video"], 1)

    def test_convert_split_skips_malformed_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos_root = root / "videos"
            videos_root.mkdir()
            video_path = videos_root / "v_bad_annotation.mp4"
            video_path.write_bytes(b"placeholder")
            payload = {
                "v_bad_annotation": {
                    "duration": 12.0,
                    "sentences": "not a list",
                    "timestamps": [[0.0, 2.0]],
                }
            }
            items, summary = _convert_split(
                payload=payload,
                split_name="train",
                video_index={"v_bad_annotation": video_path},
                videos_root=videos_root,
                max_events=6,
                default_score=1.0,
            )

        self.assertEqual(items, [])
        self.assertEqual(summary["skipped_invalid_annotation"], 1)

    def test_phase_a_checkpoint_can_warm_start_phase_b_compatibly(self) -> None:
        phase_a_config = TinyTraceConfig(
            image_size=16,
            max_frames=4,
            d_model=24,
            num_layers=1,
            num_heads=4,
            max_text_len=16,
            max_caption_tokens=0,
            min_caption_tokens=0,
            max_events=2,
            max_generated_tokens=1,
            phase_a_dense_saliency=True,
            phase_a_bin_count=5,
            visual_encoder_chunk_size=2,
        )
        phase_b_config = TinyTraceConfig(
            image_size=16,
            max_frames=4,
            d_model=24,
            num_layers=1,
            num_heads=4,
            max_text_len=16,
            max_caption_tokens=12,
            min_caption_tokens=1,
            max_events=2,
            max_generated_tokens=63,
            phase_a_dense_saliency=False,
            visual_encoder_chunk_size=2,
        )
        phase_a_model = TinyTraceModel(phase_a_config, mobileclip_backbone=FakeMobileCLIPBackbone())
        phase_b_model = TinyTraceModel(phase_b_config, mobileclip_backbone=FakeMobileCLIPBackbone())

        summary = load_model_state_compat(phase_b_model, phase_a_model.state_dict())

        self.assertGreater(summary["matched_parameter_keys"], 0)
        self.assertTrue(any(key.startswith("phase_a_") for key in summary["unexpected_keys"] + list(summary["mismatched_shapes"])))
        self.assertTrue(torch.isfinite(phase_b_model.text_head.weight).all())


if __name__ == "__main__":
    unittest.main()
