from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


QVH_NUM_BINS = 75
QVH_ANNOTATORS = 3
QVH_THRESHOLDS = ((2.0, "Fair"), (3.0, "Good"), (4.0, "VeryGood"))


def temporal_iou(first: list[float], second: list[float]) -> float:
    if len(first) < 2 or len(second) < 2:
        return 0.0
    start = max(float(first[0]), float(second[0]))
    end = min(float(first[1]), float(second[1]))
    intersection = max(0.0, end - start)
    union = max(float(first[1]), float(second[1])) - min(float(first[0]), float(second[0]))
    if union <= 0:
        return 0.0
    return intersection / union


def _best_ious(ground_truth: list[dict], predicted: list[dict]) -> list[float]:
    if not ground_truth:
        return []
    values = []
    for gt_event in ground_truth:
        gt_ts = gt_event.get("timestamp", [])
        if not predicted:
            values.append(0.0)
            continue
        values.append(
            max(
                temporal_iou(gt_ts, pred_event.get("timestamp", []))
                for pred_event in predicted
            )
        )
    return values


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return float(sum((value - mean) ** 2 for value in values) / len(values))


def _video_duration_from_path(video_path: str) -> float | None:
    parts = Path(video_path).stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return max(0.0, float(parts[-1]) - float(parts[-2]))
    except ValueError:
        return None


def _events_to_clip_scores(
    events: list[dict],
    duration: float,
    clip_length: float = 2.0,
) -> list[float]:
    """Convert legacy event predictions to clip scores.

    This helper remains available for caption/event analysis. Phase-A highlight
    evaluation intentionally does not call it: Phase A must emit one score for
    each QVHighlights clip directly instead of reconstructing scores from
    generated timestamp strings.
    """

    clip_count = max(1, int(math.ceil(duration / clip_length)))
    totals = [0.0] * clip_count
    counts = [0] * clip_count
    for event in events:
        timestamps = event.get("timestamp", [])
        scores = event.get("score", [])
        if len(timestamps) != 2 or not scores:
            continue
        start = min(max(float(timestamps[0]), 0.0), duration)
        end = min(max(float(timestamps[1]), start), duration)
        first = min(int(start / clip_length), clip_count - 1)
        last = min(int(end / clip_length), clip_count - 1)
        for index in range(first, last + 1):
            totals[index] += float(scores[0])
            counts[index] += 1
    return [total / count if count else 0.0 for total, count in zip(totals, counts)]


def _qid(item: dict[str, Any], *, context: str) -> Hashable:
    qid = item.get("qid", item.get("source_id"))
    if qid is None:
        raise ValueError(f"{context} is missing required qid/source_id")
    if isinstance(qid, bool) or not isinstance(qid, Hashable):
        raise ValueError(f"{context} has an invalid qid/source_id: {qid!r}")
    return qid


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite numeric value, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return result


def _dense_vector(value: Any, *, context: str, num_bins: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context} must be a sequence of exactly {num_bins} scores")
    if len(value) != num_bins:
        raise ValueError(f"{context} must contain exactly {num_bins} scores, got {len(value)}")
    return [
        _finite_float(score, context=f"{context}[{index}]")
        for index, score in enumerate(value)
    ]


def _dense_annotator_scores(
    item: dict[str, Any],
    *,
    context: str,
    num_bins: int,
) -> list[list[float]]:
    """Read official QVHighlights labels as a dense ``[bins, 3]`` matrix.

    TinyTrace artifacts use ``qvh_saliency_scores`` for dense labels. The
    official release's sparse ``relevant_clip_ids`` + ``saliency_scores`` form
    is accepted as well so this function can validate the source annotations
    without an intermediate conversion.
    """

    if "qvh_saliency_scores" in item:
        raw_scores = item["qvh_saliency_scores"]
        sparse_ids: Any = None
    elif "gt_saliency_scores" in item:
        raw_scores = item["gt_saliency_scores"]
        sparse_ids = None
    elif "saliency_scores" in item:
        raw_scores = item["saliency_scores"]
        sparse_ids = item.get("relevant_clip_ids")
    else:
        raise ValueError(
            f"{context} is missing official three-annotator saliency scores "
            "(`qvh_saliency_scores` or `relevant_clip_ids` + `saliency_scores`)"
        )

    if isinstance(raw_scores, (str, bytes)) or not isinstance(raw_scores, Sequence):
        raise ValueError(f"{context} saliency scores must be a sequence")

    if sparse_ids is None:
        if len(raw_scores) != num_bins:
            raise ValueError(
                f"{context} dense saliency scores must contain exactly {num_bins} bins, "
                f"got {len(raw_scores)}"
            )
        rows_and_ids = enumerate(raw_scores)
    else:
        if isinstance(sparse_ids, (str, bytes)) or not isinstance(sparse_ids, Sequence):
            raise ValueError(f"{context}.relevant_clip_ids must be a sequence")
        if len(sparse_ids) != len(raw_scores):
            raise ValueError(
                f"{context} has {len(sparse_ids)} relevant clip ids but "
                f"{len(raw_scores)} saliency rows"
            )
        seen: set[int] = set()
        indexed_rows: list[tuple[int, Any]] = []
        for row_index, (raw_id, row) in enumerate(zip(sparse_ids, raw_scores)):
            if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                raise ValueError(
                    f"{context}.relevant_clip_ids[{row_index}] must be an integer"
                )
            if raw_id < 0 or raw_id >= num_bins:
                raise ValueError(
                    f"{context}.relevant_clip_ids[{row_index}]={raw_id} is outside "
                    f"[0, {num_bins - 1}]"
                )
            if raw_id in seen:
                raise ValueError(f"{context} contains duplicate relevant clip id {raw_id}")
            seen.add(raw_id)
            indexed_rows.append((raw_id, row))
        rows_and_ids = indexed_rows

    dense = [[0.0] * QVH_ANNOTATORS for _ in range(num_bins)]
    for bin_index, row in rows_and_ids:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise ValueError(f"{context} saliency row {bin_index} must be a sequence")
        if len(row) != QVH_ANNOTATORS:
            raise ValueError(
                f"{context} saliency row {bin_index} must contain exactly "
                f"{QVH_ANNOTATORS} annotator scores, got {len(row)}"
            )
        parsed = [
            _finite_float(score, context=f"{context}.saliency_scores[{bin_index}][{worker}]")
            for worker, score in enumerate(row)
        ]
        if any(score < 0.0 or score > 4.0 for score in parsed):
            raise ValueError(
                f"{context} saliency row {bin_index} contains a score outside [0, 4]"
            )
        dense[bin_index] = parsed
    return dense


def _index_by_qid(items: list[dict], *, context: str) -> dict[Hashable, dict]:
    indexed: dict[Hashable, dict] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{context}[{index}] must be an object")
        qid = _qid(item, context=f"{context}[{index}]")
        if qid in indexed:
            raise ValueError(f"{context} contains duplicate qid/source_id {qid!r}")
        indexed[qid] = item
    return indexed


def _tie_aware_average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Return TRACE-style interpolated AP while treating score ties as a group.

    TRACE's ``get_ap`` uses ``precision_recall_curve`` and interpolates the
    precision envelope. Grouping equal scores explicitly reproduces that
    behavior without making the result depend on the order of equal-scored
    clips (or requiring scikit-learn at training time).
    """

    if len(scores) != len(labels):
        raise ValueError("Prediction and ground-truth vectors must have equal length")
    positives = sum(bool(label) for label in labels)
    if positives == 0:
        return 0.0
    if positives == len(labels):
        return 1.0

    score_groups: dict[float, list[bool]] = {}
    for score, label in zip(scores, labels):
        score_groups.setdefault(float(score), []).append(bool(label))

    precisions: list[float] = []
    recall_increased: list[bool] = []
    true_positives = 0
    seen = 0
    for score in sorted(score_groups, reverse=True):
        group = score_groups[score]
        group_positives = sum(group)
        true_positives += group_positives
        seen += len(group)
        precisions.append(true_positives / seen)
        recall_increased.append(group_positives > 0)

    # Interpolate p(r) with the best precision obtainable at equal or greater
    # recall, matching TRACE's forward envelope over sklearn's reversed curve.
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    recalled_precisions = [
        precision
        for precision, increased in zip(precisions, recall_increased)
        if increased
    ]
    return _mean(recalled_precisions)


def _official_hit1(scores: Sequence[float], labels_by_bin: Sequence[Sequence[bool]]) -> float:
    """Return TRACE/official Hit@1 (first argmax, then max over annotators)."""

    if not scores:
        return 0.0
    top_index = max(range(len(scores)), key=scores.__getitem__)
    return float(any(labels_by_bin[top_index]))


def _tie_averaged_hit1(
    scores: Sequence[float],
    labels_by_bin: Sequence[Sequence[bool]],
) -> float:
    """Average Hit@1 over all bins tied for the top score.

    This is a collapse diagnostic, not the official metric. In particular, a
    flat prediction receives the fraction of tied bins that are positive
    instead of benefiting from an arbitrary ordering or an optimistic maximum.
    """

    if not scores:
        return 0.0
    maximum = max(scores)
    tied_hits = [
        float(any(labels_by_bin[index]))
        for index, score in enumerate(scores)
        if score == maximum
    ]
    return _mean(tied_hits)


# Kept as a private compatibility name for downstream tests/imports.
def _average_precision(scores: list[float], labels: list[bool]) -> float:
    return _tie_aware_average_precision(scores, labels)


def evaluate_qvhighlights_official(
    submission: list[dict],
    ground_truth: list[dict] | None = None,
    *,
    num_bins: int = QVH_NUM_BINS,
) -> dict[str, float]:
    """Evaluate exact three-annotator QVHighlights saliency metrics.

    Predictions must be *direct* dense ``pred_saliency_scores`` vectors. There
    is deliberately no length truncation, zero-padding, qid intersection, or
    event-to-clip conversion: each prediction and label must refer to the same
    set of qids and contain exactly 75 two-second bins for the 150-second QVH
    clips. Returned values are percentages, as in the official/TRACE evaluator.

    ``ground_truth=None`` allows combined prediction artifacts containing both
    ``pred_saliency_scores`` and dense ``qvh_saliency_scores``.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    if not submission:
        return {}
    ground_truth = submission if ground_truth is None else ground_truth
    predictions_by_qid = _index_by_qid(submission, context="submission")
    ground_truth_by_qid = _index_by_qid(ground_truth, context="ground_truth")
    prediction_qids = set(predictions_by_qid)
    ground_truth_qids = set(ground_truth_by_qid)
    if prediction_qids != ground_truth_qids:
        missing = sorted((repr(qid) for qid in ground_truth_qids - prediction_qids))
        unexpected = sorted((repr(qid) for qid in prediction_qids - ground_truth_qids))
        raise ValueError(
            "QVHighlights qids must match exactly; "
            f"missing predictions={missing}, unexpected predictions={unexpected}"
        )

    parsed: list[tuple[list[float], list[list[float]]]] = []
    for qid, prediction in predictions_by_qid.items():
        if "pred_saliency_scores" not in prediction:
            raise ValueError(f"submission qid {qid!r} is missing pred_saliency_scores")
        predicted_scores = _dense_vector(
            prediction["pred_saliency_scores"],
            context=f"submission qid {qid!r}.pred_saliency_scores",
            num_bins=num_bins,
        )
        annotator_scores = _dense_annotator_scores(
            ground_truth_by_qid[qid],
            context=f"ground_truth qid {qid!r}",
            num_bins=num_bins,
        )
        parsed.append((predicted_scores, annotator_scores))

    metrics: dict[str, float] = {}
    for threshold, label_name in QVH_THRESHOLDS:
        average_precisions: list[float] = []
        hits: list[float] = []
        for predicted_scores, annotator_scores in parsed:
            binary_by_bin = [
                [score >= threshold for score in row]
                for row in annotator_scores
            ]
            for worker in range(QVH_ANNOTATORS):
                worker_labels = [row[worker] for row in binary_by_bin]
                average_precisions.append(
                    _tie_aware_average_precision(predicted_scores, worker_labels)
                )
            hits.append(_official_hit1(predicted_scores, binary_by_bin))
        metrics[f"HL-min-{label_name}-mAP"] = round(100.0 * _mean(average_precisions), 2)
        metrics[f"HL-min-{label_name}-Hit1"] = round(100.0 * _mean(hits), 2)
    return metrics


def evaluate_qvhighlights_mean_score_proxy(
    samples: list[dict],
    *,
    positive_score: float = 3.0,
    num_bins: int = QVH_NUM_BINS,
) -> dict[str, float]:
    """Evaluate a clearly named single-label proxy for Phase-A diagnostics.

    ``qvh_mean_score_targets`` contains one aggregated score per clip. This is
    useful for fast training-time feedback but is not the official QVHighlights
    metric because it discards annotator disagreement. Values are percentages.
    """

    if not samples:
        return {}
    _finite_float(positive_score, context="positive_score")
    indexed = _index_by_qid(samples, context="samples")
    parsed: list[tuple[list[float], list[float]]] = []
    for qid, sample in indexed.items():
        if "pred_saliency_scores" not in sample:
            raise ValueError(f"sample qid {qid!r} is missing pred_saliency_scores")
        if "qvh_mean_score_targets" not in sample:
            raise ValueError(f"sample qid {qid!r} is missing qvh_mean_score_targets")
        predicted_scores = _dense_vector(
            sample["pred_saliency_scores"],
            context=f"sample qid {qid!r}.pred_saliency_scores",
            num_bins=num_bins,
        )
        target_scores = _dense_vector(
            sample["qvh_mean_score_targets"],
            context=f"sample qid {qid!r}.qvh_mean_score_targets",
            num_bins=num_bins,
        )
        if any(score < 0.0 or score > 4.0 for score in target_scores):
            raise ValueError(f"sample qid {qid!r} has qvh_mean_score_targets outside [0, 4]")
        parsed.append((predicted_scores, target_scores))

    metrics: dict[str, float] = {}
    for threshold, label_name in QVH_THRESHOLDS:
        average_precisions: list[float] = []
        hits: list[float] = []
        tie_averaged_hits: list[float] = []
        constant_average_precisions: list[float] = []
        constant_hits: list[float] = []
        for predicted_scores, target_scores in parsed:
            labels = [score >= threshold for score in target_scores]
            labels_by_bin = [[label] for label in labels]
            average_precisions.append(
                _tie_aware_average_precision(predicted_scores, labels)
            )
            hits.append(_official_hit1(predicted_scores, labels_by_bin))
            tie_averaged_hits.append(
                _tie_averaged_hit1(predicted_scores, labels_by_bin)
            )
            constant_scores = [0.0] * num_bins
            constant_average_precisions.append(
                _tie_aware_average_precision(constant_scores, labels)
            )
            constant_hits.append(_official_hit1(constant_scores, labels_by_bin))
        prefix = f"qvh_mean_score_proxy_{label_name}"
        metrics[f"{prefix}_mAP"] = round(100.0 * _mean(average_precisions), 2)
        metrics[f"{prefix}_Hit1"] = round(100.0 * _mean(hits), 2)
        metrics[f"{prefix}_tie_averaged_Hit1"] = round(
            100.0 * _mean(tie_averaged_hits), 2
        )
        metrics[f"{prefix}_constant_mAP"] = round(
            100.0 * _mean(constant_average_precisions), 2
        )
        metrics[f"{prefix}_constant_Hit1"] = round(
            100.0 * _mean(constant_hits), 2
        )

    # Backwards-compatible explicit proxy aliases for callers that request a
    # non-standard threshold. Standard Phase-A monitoring uses the named Good
    # metric above.
    matching_name = next(
        (name for threshold, name in QVH_THRESHOLDS if threshold == positive_score),
        None,
    )
    if matching_name is not None:
        prefix = f"qvh_mean_score_proxy_{matching_name}"
        metrics["qvh_mean_score_proxy_mAP"] = metrics[f"{prefix}_mAP"]
        metrics["qvh_mean_score_proxy_Hit1"] = metrics[f"{prefix}_Hit1"]
        metrics["qvh_mean_score_proxy_tie_averaged_Hit1"] = metrics[
            f"{prefix}_tie_averaged_Hit1"
        ]
        metrics["qvh_mean_score_proxy_constant_mAP"] = metrics[
            f"{prefix}_constant_mAP"
        ]
        metrics["qvh_mean_score_proxy_constant_Hit1"] = metrics[
            f"{prefix}_constant_Hit1"
        ]
    return metrics


def evaluate_saliency_collapse_diagnostics(
    samples: list[dict],
    *,
    num_bins: int = QVH_NUM_BINS,
) -> dict[str, float]:
    """Summarize flat or query-independent dense saliency predictions."""

    if not samples:
        return {}
    parsed = []
    for index, sample in enumerate(samples):
        if "pred_saliency_scores" not in sample:
            raise ValueError(f"samples[{index}] is missing pred_saliency_scores")
        parsed.append(
            _dense_vector(
                sample["pred_saliency_scores"],
                context=f"samples[{index}].pred_saliency_scores",
                num_bins=num_bins,
            )
        )
    within_query_variances = [_variance(scores) for scores in parsed]
    across_query_variances = [
        _variance([scores[bin_index] for scores in parsed])
        for bin_index in range(num_bins)
    ]
    top_bins = [max(range(num_bins), key=scores.__getitem__) for scores in parsed]
    return {
        "qvh_pred_within_query_variance": _mean(within_query_variances),
        "qvh_pred_across_query_variance": _mean(across_query_variances),
        "qvh_pred_flat_query_rate": _mean(
            float(variance == 0.0) for variance in within_query_variances
        ),
        "qvh_pred_unique_argmax_ratio": len(set(top_bins)) / len(top_bins),
    }


def evaluate_qvhighlights(
    samples: list[dict],
    positive_score: float = 3.0,
    *,
    num_bins: int = QVH_NUM_BINS,
) -> dict[str, float]:
    """Evaluate available direct Phase-A outputs without mislabeling proxies.

    Official metrics are emitted only when dense three-annotator labels are
    present. Aggregated mean-score labels produce explicitly named proxy
    metrics. Legacy timestamp-event predictions are intentionally ignored here.
    """

    highlight_samples = [sample for sample in samples if sample.get("task_mode") == "highlight"]
    direct_samples = [sample for sample in highlight_samples if "pred_saliency_scores" in sample]
    if not direct_samples:
        return {}
    if len(direct_samples) != len(highlight_samples):
        raise ValueError(
            "Every highlight sample must contain direct pred_saliency_scores; "
            "mixed dense/event-only evaluation is not allowed"
        )

    metrics: dict[str, float] = {}
    official_presence = [
        any(
            key in sample
            for key in ("qvh_saliency_scores", "gt_saliency_scores", "saliency_scores")
        )
        for sample in direct_samples
    ]
    if any(official_presence):
        if not all(official_presence):
            raise ValueError(
                "Official QVHighlights labels must be present for every highlight sample"
            )
        metrics.update(evaluate_qvhighlights_official(direct_samples, num_bins=num_bins))

    proxy_presence = ["qvh_mean_score_targets" in sample for sample in direct_samples]
    if any(proxy_presence):
        if not all(proxy_presence):
            raise ValueError("qvh_mean_score_targets must be present for every highlight sample")
        metrics.update(
            evaluate_qvhighlights_mean_score_proxy(
                direct_samples,
                positive_score=positive_score,
                num_bins=num_bins,
            )
        )
    metrics.update(evaluate_saliency_collapse_diagnostics(direct_samples, num_bins=num_bins))
    return metrics


def evaluate_event_predictions(samples: list[dict]) -> dict[str, float]:
    best_ious: list[float] = []
    top1_ious: list[float] = []
    score_errors: list[float] = []
    caption_exact_hits: list[float] = []
    event_count_errors: list[float] = []

    for sample in samples:
        ground_truth = list(sample.get("ground_truth", []))
        predicted = list(sample.get("predicted", []))
        event_count_errors.append(abs(len(predicted) - len(ground_truth)))
        best_ious.extend(_best_ious(ground_truth, predicted))

        if ground_truth:
            top1_ious.append(
                temporal_iou(
                    ground_truth[0].get("timestamp", []),
                    predicted[0].get("timestamp", []) if predicted else [],
                )
            )
            if predicted:
                gt_score = float(ground_truth[0].get("score", [0.0])[0])
                pred_score = float(predicted[0].get("score", [0.0])[0])
                score_errors.append(abs(pred_score - gt_score))
                gt_caption = str(ground_truth[0].get("caption", "")).strip().lower()
                pred_caption = str(predicted[0].get("caption", "")).strip().lower()
                caption_exact_hits.append(float(gt_caption == pred_caption and bool(gt_caption)))
            else:
                score_errors.append(abs(float(ground_truth[0].get("score", [0.0])[0])))
                caption_exact_hits.append(0.0)

    metrics = {
        "temporal_mean_iou": _mean(best_ious),
        "r1_iou_0.3": _mean(1.0 if value >= 0.3 else 0.0 for value in top1_ious),
        "r1_iou_0.5": _mean(1.0 if value >= 0.5 else 0.0 for value in top1_ious),
        "score_mae": _mean(score_errors),
        "caption_exact_match": _mean(caption_exact_hits),
        "event_count_mae": _mean(event_count_errors),
    }
    metrics.update(evaluate_qvhighlights(samples))
    return metrics
