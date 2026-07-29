from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tinytrace.config import TinyTraceConfig
from tinytrace.data import JsonTinyTraceDataset, tinytrace_collate_fn


class PhaseAFeatureCacheTests(unittest.TestCase):
    def test_fp16_patch_cache_bypasses_rgb_decode_and_collates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            annotation = root / "phase_a.json"
            annotation.write_text(
                json.dumps(
                    [
                        {
                            "source_id": 1,
                            "video_path": str(video),
                            "instruction": "Find highlights for query: action",
                            "query": "action",
                            "task_mode": "highlight",
                            "dense_saliency_scores": [0.0, 3.0, 0.0],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            config = TinyTraceConfig(
                max_frames=2,
                max_text_len=48,
                max_caption_tokens=0,
                min_caption_tokens=0,
                max_events=1,
                max_generated_tokens=1,
                phase_a_dense_saliency=True,
                phase_a_bin_count=3,
            )
            dataset = JsonTinyTraceDataset(
                annotation,
                config,
                allow_random_frames=False,
                visual_feature_cache_dir=root / "features",
            )
            features = torch.rand(2, 64, config.visual_hidden_dim)
            times = torch.tensor([0.0, 1.0])

            cache_path = dataset.write_visual_feature_cache(0, features, times)
            stored = torch.load(cache_path, map_location="cpu", weights_only=True)
            self.assertEqual(stored["patch_features"].dtype, torch.float16)

            strict_cached = JsonTinyTraceDataset(
                annotation,
                config,
                allow_random_frames=False,
                visual_feature_cache_dir=root / "features",
                require_visual_feature_cache=True,
            )
            sample = strict_cached[0]
            batch = tinytrace_collate_fn([sample])

            self.assertEqual(sample["frames"].shape, (2, 3, 1, 1))
            self.assertEqual(sample["visual_patch_features"].dtype, torch.float32)
            self.assertEqual(
                batch["visual_patch_features"].shape,
                (1, 2, 64, config.visual_hidden_dim),
            )
            self.assertEqual(batch["saliency_targets"].tolist(), [[0.0, 3.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
