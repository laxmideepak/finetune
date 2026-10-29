"""Paired patient-cluster bootstrap noninferiority comparison.

Pre-registered deployment rule (replaces "95% CI overlap"):

    ΔF1 = candidate macro-F1 − reference macro-F1
    lower one-sided 95% bound of ΔF1 must be > −f1_margin   (default 0.01)
    lower bound of ΔCURRENT-recall must be > −recall_margin (default 0.02)
    lower bound of ΔCURRENT-precision must be > −precision_margin (default 0.02)

Patients, not rows, are the resampling unit. Model/checkpoint selection runs on
dev; the selected comparison is confirmed once on the locked test with
``--split test --confirm-locked-test``.

Predictions CSV columns:
    example_id, patient_uid, split, label_policy_version,
    y_true, y_pred_candidate, y_pred_reference

Invalid generative outputs may appear as ``__INVALID__``; they count as errors
and are never silently converted to UNKNOWN.

Usage:
    python src/compare_models.py --pred-csv dev_predictions.csv --split dev
    python src/compare_models.py --pred-csv test_predictions.csv --split test \
        --confirm-locked-test
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.label_policy import LABELS, POLICY_VERSION
    from src.metrics import INVALID, PREDICTION_LABELS, confusion as _confusion, current_scores as _current_scores, macro_f1 as _macro_f1
except ImportError:
    from label_policy import LABELS, POLICY_VERSION
    from metrics import INVALID, PREDICTION_LABELS, confusion as _confusion, current_scores as _current_scores, macro_f1 as _macro_f1

_REQUIRED = (
    "example_id",
    "patient_uid",
    "split",
    "label_policy_version",
    "y_true",
    "y_pred_candidate",
    "y_pred_reference",
)
_CURRENT_INDEX = LABELS.index("CURRENT")


def _delta(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    candidate_recall, candidate_precision = _current_scores(candidate)
    reference_recall, reference_precision = _current_scores(reference)
    return {
        "delta_macro_f1": _macro_f1(candidate) - _macro_f1(reference),
        "delta_current_recall": (
            None if candidate_recall is None or reference_recall is None
            else candidate_recall - reference_recall
        ),
        "delta_current_precision": (
            None if candidate_precision is None or reference_precision is None
            else candidate_precision - reference_precision
        ),
    }


def compare_predictions(
    frame: pd.DataFrame,
    *,
    split: str = "dev",
    n_bootstrap: int = 2000,
    seed: int = 20261028,
    f1_margin: float = 0.01,
    current_recall_margin: float = 0.02,
    current_precision_margin: float = 0.02,
) -> dict[str, Any]:
    """Return point estimates, lower bounds, and the pre-registered pass/fail."""
    missing = [column for column in _REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if split not in ("dev", "test"):
        raise ValueError("split must be 'dev' or 'test'")
    data = frame.copy()
    for column in _REQUIRED:
        data[column] = data[column].astype(str).str.strip()
    data = data[data["split"] == split].copy()
    if data.empty:
        raise ValueError(f"No rows for split {split!r}")
    if data["example_id"].duplicated().any():
        raise ValueError("Duplicate example_id in prediction rows")
    if not set(data["label_policy_version"]) == {POLICY_VERSION}:
        found = sorted(set(data["label_policy_version"]))
        raise ValueError(f"label_policy_version mismatch: found {found}, expected {POLICY_VERSION!r}")
    bad_truth = sorted(set(data["y_true"]) - set(LABELS))
    if bad_truth:
        raise ValueError(f"Invalid y_true labels: {bad_truth}")
    bad_prediction = sorted(
        (set(data["y_pred_candidate"]) | set(data["y_pred_reference"])) - set(PREDICTION_LABELS)
    )
    if bad_prediction:
        raise ValueError(f"Invalid prediction labels: {bad_prediction}")

    for margin_name, margin in (
        ("f1_margin", f1_margin),
        ("current_recall_margin", current_recall_margin),
        ("current_precision_margin", current_precision_margin),
    ):
        if margin < 0:
            raise ValueError(f"{margin_name} must be >= 0")

    clusters = sorted(data["patient_uid"].unique().tolist())
    cluster_index = {cluster: i for i, cluster in enumerate(clusters)}
    candidate_by_cluster = np.zeros((len(clusters), len(LABELS), len(PREDICTION_LABELS)), dtype=np.int64)
    reference_by_cluster = np.zeros_like(candidate_by_cluster)
    for cluster, rows in data.groupby("patient_uid"):
        index = cluster_index[cluster]
        candidate_by_cluster[index] += _confusion(rows["y_true"].to_numpy(), rows["y_pred_candidate"].to_numpy())
        reference_by_cluster[index] += _confusion(rows["y_true"].to_numpy(), rows["y_pred_reference"].to_numpy())

    candidate_total = candidate_by_cluster.sum(axis=0)
    reference_total = reference_by_cluster.sum(axis=0)
    point = _delta(candidate_total, reference_total)

    rng = np.random.default_rng(seed)
    draws_f1: list[float] = []
    draws_recall: list[float] = []
    draws_precision: list[float] = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(clusters), size=len(clusters))
        candidate = candidate_by_cluster[sample].sum(axis=0)
        reference = reference_by_cluster[sample].sum(axis=0)
        delta = _delta(candidate, reference)
        draws_f1.append(float(delta["delta_macro_f1"]))
        if delta["delta_current_recall"] is not None:
            draws_recall.append(float(delta["delta_current_recall"]))
        if delta["delta_current_precision"] is not None:
            draws_precision.append(float(delta["delta_current_precision"]))

    lower_f1 = float(np.percentile(draws_f1, 5))
    lower_recall = float(np.percentile(draws_recall, 5)) if draws_recall else None
    lower_precision = float(np.percentile(draws_precision, 5)) if draws_precision else None

    f1_pass = lower_f1 > -f1_margin
    recall_pass = lower_recall is not None and lower_recall > -current_recall_margin
    precision_pass = lower_precision is not None and lower_precision > -current_precision_margin

    return {
        "split": split,
        "policy_version": POLICY_VERSION,
        "n_rows": int(len(data)),
        "n_clusters": len(clusters),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "margins": {
            "f1": f1_margin,
            "current_recall": current_recall_margin,
            "current_precision": current_precision_margin,
        },
        "point_estimates": point,
        "lower_95_bounds": {
            "delta_macro_f1": lower_f1,
            "delta_current_recall": lower_recall,
            "delta_current_precision": lower_precision,
        },
        "valid_replicates": {
            "macro_f1": len(draws_f1),
            "current_recall": len(draws_recall),
            "current_precision": len(draws_precision),
        },
        "pass": {
            "macro_f1_noninferior": bool(f1_pass),
            "current_recall_noninferior": bool(recall_pass),
            "current_precision_noninferior": bool(precision_pass),
            "overall": bool(f1_pass and recall_pass and precision_pass),
        },
        "note": "Lower bounds are the 5th percentile of paired patient-cluster bootstrap draws; invalid outputs count as errors.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--f1-margin", type=float, default=0.01)
    parser.add_argument("--current-recall-margin", type=float, default=0.02)
    parser.add_argument("--current-precision-margin", type=float, default=0.02)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    if args.split == "test" and not args.confirm_locked_test:
        parser.error("locked-test comparison requires --confirm-locked-test (selection must use dev)")
    frame = pd.read_csv(args.pred_csv, dtype=str, keep_default_na=False)
    result = compare_predictions(
        frame,
        split=args.split,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        f1_margin=args.f1_margin,
        current_recall_margin=args.current_recall_margin,
        current_precision_margin=args.current_precision_margin,
    )
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0 if result["pass"]["overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
