from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int


def evaluate_threshold(
    scores: list[float],
    labels: list[int],
    threshold: float,
) -> ThresholdMetrics:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length.")
    if not scores:
        raise ValueError("scores must not be empty.")

    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        actual = bool(label)
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ThresholdMetrics(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        true_negatives=tn,
    )


def choose_threshold_by_f1(scores: list[float], labels: list[int]) -> ThresholdMetrics:
    candidates = sorted({0.0, 1.0, *scores})
    best = evaluate_threshold(scores=scores, labels=labels, threshold=candidates[0])
    for threshold in candidates[1:]:
        metrics = evaluate_threshold(scores=scores, labels=labels, threshold=threshold)
        if (metrics.f1, metrics.precision, metrics.recall) > (
            best.f1,
            best.precision,
            best.recall,
        ):
            best = metrics
    return best
