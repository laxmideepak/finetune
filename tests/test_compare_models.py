"""Executable checks for the paired patient-cluster noninferiority rule."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.compare_models import compare_predictions, main
from src.label_policy import LABELS, POLICY_VERSION
from src.metrics import INVALID, metrics_payload

LABELS_CYCLE = list(LABELS)


def make_frame(rows: int = 80, patients: int = 20, split: str = "dev") -> pd.DataFrame:
    records = []
    for index in range(rows):
        truth = LABELS_CYCLE[index % len(LABELS_CYCLE)]
        records.append({
            "example_id": f"e{index}",
            "patient_uid": f"c1::p{index % patients}",
            "split": split,
            "label_policy_version": POLICY_VERSION,
            "y_true": truth,
            "y_pred_candidate": truth,
            "y_pred_reference": truth,
        })
    return pd.DataFrame(records)


class TestNoninferiority(unittest.TestCase):
    def test_identical_models_are_noninferior(self):
        result = compare_predictions(make_frame(), n_bootstrap=200, seed=7)
        self.assertTrue(result["pass"]["overall"])
        self.assertEqual(result["point_estimates"]["delta_macro_f1"], 0.0)
        self.assertGreater(result["lower_95_bounds"]["delta_macro_f1"], -0.01)

    def test_clearly_worse_candidate_fails(self):
        frame = make_frame()
        frame.loc[frame["y_true"] == "CURRENT", "y_pred_candidate"] = "NEVER"
        result = compare_predictions(frame, n_bootstrap=500, seed=7)
        self.assertFalse(result["pass"]["macro_f1_noninferior"])
        self.assertFalse(result["pass"]["overall"])
        self.assertLess(result["point_estimates"]["delta_current_recall"], -0.5)

    def test_invalid_outputs_count_as_errors(self):
        frame = make_frame()
        frame.loc[frame["y_true"] == "CURRENT", "y_pred_candidate"] = INVALID
        result = compare_predictions(frame, n_bootstrap=200, seed=7)
        self.assertFalse(result["pass"]["current_recall_noninferior"])
        self.assertLess(result["point_estimates"]["delta_current_recall"], -0.5)

    def test_current_precision_requirement_catches_overprediction(self):
        frame = make_frame()
        # Candidate calls every row CURRENT: recall is perfect, precision is 25%.
        frame["y_pred_candidate"] = "CURRENT"
        result = compare_predictions(frame, n_bootstrap=500, seed=7)
        self.assertTrue(result["pass"]["current_recall_noninferior"])
        self.assertFalse(result["pass"]["current_precision_noninferior"])
        self.assertFalse(result["pass"]["overall"])

    def test_deterministic_with_seed(self):
        frame = make_frame()
        frame.loc[frame["y_true"] == "CURRENT", "y_pred_candidate"] = "FORMER"
        first = compare_predictions(frame, n_bootstrap=300, seed=11)
        second = compare_predictions(frame, n_bootstrap=300, seed=11)
        self.assertEqual(first["lower_95_bounds"], second["lower_95_bounds"])

    def test_policy_version_mismatch_rejected(self):
        frame = make_frame()
        frame["label_policy_version"] = "old-policy"
        with self.assertRaisesRegex(ValueError, "label_policy_version mismatch"):
            compare_predictions(frame)

    def test_locked_test_requires_confirmation(self):
        frame = make_frame(split="test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preds.csv"
            frame.to_csv(path, index=False)
            with self.assertRaises(SystemExit) as raised:
                main(["--pred-csv", str(path), "--split", "test"])
            self.assertEqual(raised.exception.code, 2)


class TestMetrics(unittest.TestCase):
    def test_invalid_is_separate_from_unknown(self):
        payload = metrics_payload(["CURRENT", "FORMER"], ["CURRENT", INVALID])
        self.assertEqual(payload["n_invalid"], 1)
        self.assertEqual(payload["invalid_rate"], 0.5)
        self.assertEqual(payload["coverage"], 0.5)
        # Four-class macro-F1 still scores absent classes as zero, even when
        # the valid-only view has just one class present.
        self.assertEqual(payload["macro_f1_valid_only"], 0.25)
        self.assertLess(payload["macro_f1"], 1.0)
        self.assertEqual(payload["confusion_matrix_rows_true_cols_pred"][1][-1], 1)

    def test_current_metrics(self):
        payload = metrics_payload(
            ["CURRENT", "CURRENT", "FORMER", "NEVER"],
            ["CURRENT", "FORMER", "FORMER", "NEVER"],
        )
        self.assertEqual(payload["current_recall"], 0.5)
        self.assertEqual(payload["current_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
