from __future__ import annotations

import unittest

from tinytrace.metrics import (
    QVH_NUM_BINS,
    evaluate_event_predictions,
    evaluate_qvhighlights,
    evaluate_qvhighlights_mean_score_proxy,
    evaluate_qvhighlights_official,
)


def _scores(*values: tuple[int, float]) -> list[float]:
    result = [0.0] * QVH_NUM_BINS
    for index, value in values:
        result[index] = value
    return result


def _annotator_scores(*values: tuple[int, list[float]]) -> list[list[float]]:
    result = [[0.0, 0.0, 0.0] for _ in range(QVH_NUM_BINS)]
    for index, value in values:
        result[index] = value
    return result


class PhaseADenseMetricsTests(unittest.TestCase):
    def test_official_metrics_use_all_three_annotators_and_percent_units(self) -> None:
        prediction = {
            "qid": 17,
            "pred_saliency_scores": _scores((0, 2.0), (1, 1.0)),
        }
        ground_truth = {
            "qid": 17,
            "qvh_saliency_scores": _annotator_scores(
                (0, [4.0, 3.0, 2.0]),
                (1, [2.0, 4.0, 4.0]),
            ),
        }

        metrics = evaluate_qvhighlights_official([prediction], [ground_truth])

        self.assertEqual(metrics["HL-min-Fair-mAP"], 100.0)
        self.assertEqual(metrics["HL-min-Good-mAP"], 83.33)
        self.assertEqual(metrics["HL-min-VeryGood-mAP"], 66.67)
        self.assertEqual(metrics["HL-min-Fair-Hit1"], 100.0)
        self.assertEqual(metrics["HL-min-Good-Hit1"], 100.0)
        self.assertEqual(metrics["HL-min-VeryGood-Hit1"], 100.0)

    def test_ap_is_tie_aware_but_official_hit1_uses_first_argmax(self) -> None:
        # The relevant bin is tied with an irrelevant bin. TRACE-style AP treats
        # the score group as one operating point (AP=1/2). Official TRACE Hit1
        # retains numpy's first-argmax behavior and therefore selects bin zero.
        prediction = {
            "source_id": "query-a",
            "pred_saliency_scores": _scores((0, 1.0), (1, 1.0)),
        }
        ground_truth = {
            "source_id": "query-a",
            "relevant_clip_ids": [1],
            "saliency_scores": [[4.0, 4.0, 4.0]],
        }

        metrics = evaluate_qvhighlights_official([prediction], [ground_truth])

        self.assertEqual(metrics["HL-min-VeryGood-mAP"], 50.0)
        self.assertEqual(metrics["HL-min-VeryGood-Hit1"], 0.0)

    def test_official_evaluator_rejects_qid_and_bin_mismatches(self) -> None:
        prediction = {"qid": 1, "pred_saliency_scores": [0.0] * QVH_NUM_BINS}
        dense_ground_truth = {
            "qid": 1,
            "qvh_saliency_scores": _annotator_scores(),
        }

        with self.assertRaisesRegex(ValueError, "qids must match exactly"):
            evaluate_qvhighlights_official(
                [prediction],
                [{**dense_ground_truth, "qid": 2}],
            )
        with self.assertRaisesRegex(ValueError, "exactly 75"):
            evaluate_qvhighlights_official(
                [{"qid": 1, "pred_saliency_scores": [0.0] * 74}],
                [dense_ground_truth],
            )
        with self.assertRaisesRegex(ValueError, "exactly 3 annotator"):
            evaluate_qvhighlights_official(
                [prediction],
                [{"qid": 1, "qvh_saliency_scores": [[0.0, 0.0]] * QVH_NUM_BINS}],
            )
        with self.assertRaisesRegex(ValueError, r"outside \[0, 74\]"):
            evaluate_qvhighlights_official(
                [prediction],
                [{"qid": 1, "relevant_clip_ids": [75], "saliency_scores": [[4, 4, 4]]}],
            )

    def test_mean_score_metric_is_explicitly_a_proxy(self) -> None:
        sample = {
            "task_mode": "highlight",
            "qid": 91,
            "pred_saliency_scores": _scores((3, 2.0)),
            "qvh_mean_score_targets": _scores((3, 4.0)),
        }

        proxy = evaluate_qvhighlights_mean_score_proxy([sample])
        combined = evaluate_qvhighlights([sample])

        self.assertEqual(proxy["qvh_mean_score_proxy_mAP"], 100.0)
        self.assertEqual(proxy["qvh_mean_score_proxy_Hit1"], 100.0)
        self.assertEqual(proxy["qvh_mean_score_proxy_tie_averaged_Hit1"], 100.0)
        self.assertEqual(proxy["qvh_mean_score_proxy_constant_mAP"], 1.33)
        self.assertEqual(proxy["qvh_mean_score_proxy_constant_Hit1"], 0.0)
        self.assertEqual(combined["qvh_mean_score_proxy_mAP"], 100.0)
        self.assertNotIn("qvh_mAP", combined)
        self.assertNotIn("qvh_HIT_at_1", combined)

    def test_caption_event_metrics_do_not_claim_qvh_metrics(self) -> None:
        sample = {
            "task_mode": "caption",
            "ground_truth": [
                {"timestamp": [2.0, 4.0], "score": [3.0], "caption": "a jump"}
            ],
            "predicted": [
                {"timestamp": [2.0, 4.0], "score": [3.0], "caption": "a jump"}
            ],
        }

        metrics = evaluate_event_predictions([sample])

        self.assertEqual(metrics["temporal_mean_iou"], 1.0)
        self.assertEqual(metrics["caption_exact_match"], 1.0)
        self.assertFalse(any(key.startswith("qvh_") for key in metrics))
        self.assertFalse(any(key.startswith("HL-min-") for key in metrics))


if __name__ == "__main__":
    unittest.main()
