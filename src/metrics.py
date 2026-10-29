"""Shared classification metrics with explicit invalid-output handling.

``macro_f1`` is the primary four-class macro-F1 with invalid generative outputs
counted as errors: an invalid prediction can create a false negative for the
true class but never a false positive for a valid class. ``macro_f1_valid_only``
is reported alongside it so format failures cannot hide inside a clinical class.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from src.label_policy import LABELS
except ImportError:
    from label_policy import LABELS

INVALID = "__INVALID__"
PREDICTION_LABELS = tuple(LABELS) + (INVALID,)
CURRENT_INDEX = LABELS.index("CURRENT")


def normalize_prediction(value: Any) -> str:
    """Return a valid label, INVALID, or raise for a bad ground-truth label."""
    text = str(value).strip().upper()
    if text in LABELS:
        return text
    return INVALID


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Rows = four true classes, columns = four labels + INVALID."""
    truth_index = {label: i for i, label in enumerate(LABELS)}
    prediction_index = {label: i for i, label in enumerate(PREDICTION_LABELS)}
    matrix = np.zeros((len(LABELS), len(PREDICTION_LABELS)), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        if truth not in truth_index:
            raise ValueError(f"Invalid ground-truth label {truth!r}")
        matrix[truth_index[truth], prediction_index[prediction]] += 1
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    scores = []
    for index in range(len(LABELS)):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def current_scores(matrix: np.ndarray) -> tuple[float | None, float | None]:
    true_positive = matrix[CURRENT_INDEX, CURRENT_INDEX]
    false_positive = matrix[:, CURRENT_INDEX].sum() - true_positive
    false_negative = matrix[CURRENT_INDEX, :].sum() - true_positive
    recall = None if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
    precision = None if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
    return recall, precision


def per_class_report(matrix: np.ndarray) -> dict[str, dict[str, float | int]]:
    report: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(LABELS):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = 0.0 if true_positive + false_positive == 0 else true_positive / (true_positive + false_positive)
        recall = 0.0 if true_positive + false_negative == 0 else true_positive / (true_positive + false_negative)
        f1 = 0.0 if 2 * true_positive + false_positive + false_negative == 0 else (
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
        )
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
    return report


def metrics_payload(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=object)
    predictions = np.asarray(y_pred, dtype=object)
    if len(truth) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length")
    matrix = confusion(truth, predictions)
    valid_mask = predictions != INVALID
    valid_matrix = confusion(truth[valid_mask], predictions[valid_mask]) if valid_mask.any() else np.zeros_like(matrix)
    recall, precision = current_scores(matrix)
    return {
        "n": int(len(truth)),
        "n_invalid": int((~valid_mask).sum()),
        "invalid_rate": float((~valid_mask).mean()) if len(truth) else 0.0,
        "coverage": float(valid_mask.mean()) if len(truth) else 0.0,
        "macro_f1": macro_f1(matrix),
        "macro_f1_valid_only": macro_f1(valid_matrix) if valid_mask.any() else None,
        "accuracy_all": float(matrix.trace() / len(truth)) if len(truth) else 0.0,
        "current_recall": recall,
        "current_precision": precision,
        "per_class": per_class_report(matrix),
        "confusion_matrix_rows_true_cols_pred": matrix.tolist(),
        "prediction_columns": list(PREDICTION_LABELS),
        "note": "macro_f1 counts invalid outputs as errors; macro_f1_valid_only excludes them so format failures stay visible",
    }
