from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import torch

from .temporal import temporal_iou


def localization_metrics(predicted: list[list[dict[str, float]]], targets: list[torch.Tensor], durations: list[float], threshold: float = 0.5) -> dict[str, float]:
    """Evaluate per-video target rows without assuming batch padding is global."""
    if len(predicted) != len(targets) or len(predicted) != len(durations):
        raise ValueError("Predictions, target rows, and durations must have equal lengths.")
    matches = predictions_count = targets_count = 0
    ious: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    for index, rows in enumerate(predicted):
        truth = targets[index].detach().cpu()
        targets_count += len(truth)
        used: set[int] = set()
        predictions_count += len(rows)
        for item in rows:
            candidate = torch.tensor([item["start"], item["end"]])
            if not len(truth):
                continue
            values = temporal_iou(candidate.unsqueeze(0), truth).tolist()
            best = max(range(len(values)), key=values.__getitem__)
            if values[best] < threshold or best in used:
                continue
            used.add(best)
            matches += 1
            ious.append(values[best])
            start_errors.append(abs(float(candidate[0] - truth[best, 0])) * durations[index])
            end_errors.append(abs(float(candidate[1] - truth[best, 1])) * durations[index])
    precision = matches / predictions_count if predictions_count else 0.0
    recall = matches / targets_count if targets_count else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "mean_iou": sum(ious) / len(ious) if ious else 0.0, "start_mae_seconds": sum(start_errors) / len(start_errors) if start_errors else 0.0, "end_mae_seconds": sum(end_errors) / len(end_errors) if end_errors else 0.0, "matched_events": float(matches), "target_events": float(targets_count), "predicted_events": float(predictions_count)}


def _tokens(text: str) -> list[str]:
    return [piece for piece in text.lower().split() if piece]


def caption_metrics(predictions: Iterable[str], references: Iterable[str]) -> dict[str, float]:
    """Dependency-free corpus BLEU-1, METEOR-style unigram F, and CIDEr-style TF-IDF cosine.

    These are explicit lightweight proxies for per-epoch checkpoint selection;
    final reporting may additionally use the official ActivityNet tooling.
    """
    pairs = list(zip(predictions, references))
    if not pairs:
        return {"bleu1": 0.0, "meteor_unigram": 0.0, "cider_unigram": 0.0, "caption_count": 0.0, "empty_predictions": 0.0}
    document_frequency: Counter[str] = Counter()
    references_tokens = [_tokens(reference) for _, reference in pairs]
    for row in references_tokens:
        document_frequency.update(set(row))
    bleu_precision = meteor = cider = 0.0
    empty = 0
    documents = len(pairs)
    for (prediction, _), reference in zip(pairs, references_tokens):
        candidate = _tokens(prediction)
        if not candidate:
            empty += 1
            continue
        overlap = sum((Counter(candidate) & Counter(reference)).values())
        precision, recall = overlap / len(candidate), overlap / len(reference) if reference else 0.0
        bleu_precision += precision
        meteor += 10 * precision * recall / (recall + 9 * precision) if precision + recall else 0.0
        vocabulary = set(candidate) | set(reference)
        weights = {word: math.log((documents + 1) / (document_frequency[word] + 1)) + 1 for word in vocabulary}
        candidate_count, reference_count = Counter(candidate), Counter(reference)
        dot = sum(candidate_count[word] * reference_count[word] * weights[word] ** 2 for word in vocabulary)
        candidate_norm = math.sqrt(sum((candidate_count[word] * weights[word]) ** 2 for word in vocabulary))
        reference_norm = math.sqrt(sum((reference_count[word] * weights[word]) ** 2 for word in vocabulary))
        cider += dot / (candidate_norm * reference_norm) if candidate_norm and reference_norm else 0.0
    return {"bleu1": bleu_precision / documents, "meteor_unigram": meteor / documents, "cider_unigram": cider / documents, "caption_count": float(documents), "empty_predictions": float(empty)}


def matched_caption_metrics(predicted: list[list[dict[str, object]]], targets: list[torch.Tensor], captions: list[list[str]], threshold: float = 0.5) -> dict[str, float]:
    """Score only one-to-one temporally matched predicted captions."""
    generated: list[str] = []
    references: list[str] = []
    total_targets = 0
    if len(predicted) != len(targets) or len(predicted) != len(captions):
        raise ValueError("Predictions, target rows, and captions must have equal lengths.")
    for batch_index, rows in enumerate(predicted):
        truth = targets[batch_index].detach().cpu()
        total_targets += len(truth)
        used: set[int] = set()
        for item in rows:
            candidate = torch.tensor([float(item["start"]), float(item["end"])])
            if not len(truth):
                continue
            scores = temporal_iou(candidate.unsqueeze(0), truth).tolist()
            target_index = max(range(len(scores)), key=scores.__getitem__)
            if scores[target_index] < threshold or target_index in used:
                continue
            used.add(target_index)
            generated.append(str(item.get("caption", "")))
            references.append(captions[batch_index][target_index])
    return {**caption_metrics(generated, references), "temporal_match_coverage": len(references) / total_targets if total_targets else 0.0}
