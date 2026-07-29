from __future__ import annotations

import unittest

import torch

from tinytrace import TinyTraceConfig, decode_event_sequence, serialize_example
from tinytrace.metrics import evaluate_event_predictions
from tinytrace.serialization import LabelType
from tinytrace.tokenizers import CharTokenizer, NumericTokenizer


class QVHighlightsPhaseATests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = TinyTraceConfig(max_events=2, max_caption_tokens=0, min_caption_tokens=0)
        self.text = CharTokenizer(self.config.text_vocab_size)
        self.time = NumericTokenizer(self.config.time_vocab, width=6)
        self.score = NumericTokenizer(self.config.score_vocab, width=3)

    def test_highlight_serialization_masks_caption_and_marks_boundaries(self) -> None:
        events = [
            {"timestamp": [2.0, 4.0], "score": [3.7]},
            {"timestamp": [8.0, 10.0], "score": [4.0]},
        ]
        token_ids, label_types, prompt_length = serialize_example(
            events,
            "Find highlights for query: a person cooks",
            self.config,
            self.text,
            self.time,
            self.score,
            task_mode="highlight",
        )
        self.assertEqual(label_types.count(LabelType.TEXT), 1)  # final EOS only
        self.assertEqual(label_types.count(LabelType.HIGHLIGHT_BOUNDARY), 2)
        decoded = decode_event_sequence(
            token_ids[prompt_length:],
            self.config,
            self.text,
            self.time,
            self.score,
            task_mode="highlight",
        )
        self.assertEqual(decoded, events)

    def test_qvhighlights_metrics_reward_correct_ranked_clip(self) -> None:
        scores = [0.0] * 75
        scores[2] = 4.0
        sample = {
            "task_mode": "highlight",
            "source_id": 1,
            "video_path": "clip_0.0_10.0.mp4",
            "ground_truth": [],
            "predicted": [],
            "pred_saliency_scores": scores,
            "qvh_mean_score_targets": scores,
        }
        metrics = evaluate_event_predictions([sample])
        self.assertEqual(metrics["qvh_mean_score_proxy_Good_mAP"], 100.0)
        self.assertEqual(metrics["qvh_mean_score_proxy_Good_Hit1"], 100.0)
        self.assertNotIn("qvh_mAP", metrics)


if __name__ == "__main__":
    unittest.main()
