"""Executable checks for the shared dataset contract."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.data_contract import REQUIRED_COLUMNS, read_and_validate_csv, validate_dataframe
from src.label_policy import POLICY_VERSION

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "smoke_splits.csv"


def frame() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)


def row(**overrides: str) -> dict[str, str]:
    base = {column: "x" for column in REQUIRED_COLUMNS}
    base.update({
        "example_id": "e1",
        "corpus_id": "c1",
        "text": "Patient smokes daily",
        "label": "CURRENT",
        "patient_id": "p1",
        "note_id": "n1",
        "split": "train",
        "leakage_group": "g1",
        "label_policy_version": POLICY_VERSION,
        "label_source": "human",
        "retrieval_version": "retrieval-v2",
        "example_mode": "note-context",
        "label_scope": "note",
        "crop_id": "note",
    })
    base.update(overrides)
    return base


def one_row(**overrides: str) -> pd.DataFrame:
    return pd.DataFrame([row(**overrides)])


class TestFixture(unittest.TestCase):
    def test_fixture_is_valid_and_manifest_is_stable(self):
        clean, manifest = read_and_validate_csv(str(FIXTURE))
        self.assertEqual(len(clean), 16)
        self.assertEqual(manifest["split_counts"], {"train": 8, "dev": 4, "test": 4})
        self.assertEqual(manifest["policy_version"], POLICY_VERSION)
        self.assertEqual(manifest["example_mode_counts"]["note-context"], 16)
        again, manifest_again = read_and_validate_csv(str(FIXTURE))
        self.assertEqual(manifest["data_fingerprint"], manifest_again["data_fingerprint"])
        self.assertEqual(list(again["label"]), list(clean["label"]))

    def test_all_required_columns_present(self):
        clean, _ = read_and_validate_csv(str(FIXTURE))
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(clean.columns))


class TestRejections(unittest.TestCase):
    def test_missing_column(self):
        data = frame().drop(columns=["example_id"])
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_dataframe(data)

    def test_duplicate_example_id(self):
        data = frame()
        data.loc[1, "example_id"] = data.loc[0, "example_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate example_id"):
            validate_dataframe(data)

    def test_empty_identifier(self):
        data = frame()
        data.loc[0, "patient_id"] = "  "
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_dataframe(data)

    def test_policy_version_mismatch(self):
        data = frame()
        data["label_policy_version"] = "old-policy"
        with self.assertRaisesRegex(ValueError, "label_policy_version mismatch"):
            validate_dataframe(data)

    def test_crop_with_note_scope_rejected(self):
        with self.assertRaisesRegex(ValueError, "example_mode/label_scope mismatch"):
            validate_dataframe(
                one_row(example_mode="crop", label_scope="note", crop_id="c1"),
                require_all_splits=False, require_all_classes_per_split=False,
            )

    def test_unknown_requires_reason(self):
        data = frame()
        data.loc[data["label"] == "UNKNOWN", "unknown_reason"] = ""
        with self.assertRaisesRegex(ValueError, "UNKNOWN rows require unknown_reason"):
            validate_dataframe(data)

    def test_non_unknown_must_not_have_reason(self):
        data = frame()
        data.loc[data["label"] == "CURRENT", "unknown_reason"] = "conflict"
        with self.assertRaisesRegex(ValueError, "must not carry unknown_reason"):
            validate_dataframe(data)

    def test_invalid_label_and_split_and_source(self):
        with self.assertRaisesRegex(ValueError, "Invalid labels"):
            validate_dataframe(one_row(label="SMOKER"))
        with self.assertRaisesRegex(ValueError, "Invalid splits"):
            validate_dataframe(one_row(split="validation"))
        with self.assertRaisesRegex(ValueError, "Invalid label_source"):
            validate_dataframe(
                one_row(label_source="guessed"),
                require_all_splits=False, require_all_classes_per_split=False,
            )

    def test_cross_split_leakage_same_corpus(self):
        data = frame()
        data.loc[data["split"] == "test", "patient_id"] = data.loc[data["split"] == "train", "patient_id"].iloc[0]
        with self.assertRaisesRegex(ValueError, "Cross-split leakage in patient_id"):
            validate_dataframe(data)

    def test_same_patient_id_in_different_corpora_is_not_leakage(self):
        data = frame()
        data.loc[data["split"] == "test", "corpus_id"] = "other-corpus"
        data.loc[data["split"] == "test", "patient_id"] = data.loc[data["split"] == "train", "patient_id"].iloc[0]
        clean, _ = validate_dataframe(data)
        self.assertEqual(len(clean), 16)

    def test_note_context_key_uniqueness(self):
        data = frame()
        data.loc[1, "corpus_id"] = data.loc[0, "corpus_id"]
        data.loc[1, "note_id"] = data.loc[0, "note_id"]
        data.loc[1, "crop_id"] = data.loc[0, "crop_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate note-context example key"):
            validate_dataframe(data)

    def test_part_index_contract(self):
        data = frame()
        data["part_index"] = "0"
        data["part_count"] = "1"
        clean, _ = validate_dataframe(data)
        self.assertEqual(clean["part_index"].iloc[0], 0)
        broken = frame()
        broken["part_index"] = "2"
        broken["part_count"] = "1"
        with self.assertRaisesRegex(ValueError, "Invalid part_index/part_count"):
            validate_dataframe(broken)


if __name__ == "__main__":
    unittest.main()
