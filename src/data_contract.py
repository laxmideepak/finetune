"""Shared dataset contract for every training and evaluation entry point.

Both the encoder and generative arms must call ``validate_dataframe`` (or
``read_and_validate_csv``). The contract is intentionally stricter than the
starter scripts: a model must never be trained on a label whose scope or policy
version does not match the visible input.

Required columns
----------------
example_id            globally unique stable identifier for this row
corpus_id             dataset namespace (prevents patient-id collisions)
text                  visible model input (crop or note-context)
label                 CURRENT | FORMER | NEVER | UNKNOWN
patient_id            source patient identifier (namespaced by corpus_id)
note_id               source note identifier (namespaced by corpus_id)
split                 train | dev | test
leakage_group         connected-component id for patient/near-duplicate leakage
label_policy_version  must equal label_policy.POLICY_VERSION
label_source          human | adjudicated | synthetic-reference | teacher | mixed
retrieval_version     extractor/retrieval version that produced the visible text
example_mode          crop | note-context
label_scope           crop | note (must agree with example_mode)
crop_id               crop identifier, or "note"/"note-pN" for note-context rows

Optional columns: part_index, part_count, note_date, unknown_reason.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

try:  # running as a module: python -m src.train_encoder
    from src.label_policy import LABELS, POLICY_VERSION, UNKNOWN_REASONS, validate_label
except ImportError:  # running as a script: python src/train_encoder.py
    from label_policy import LABELS, POLICY_VERSION, UNKNOWN_REASONS, validate_label

REQUIRED_COLUMNS = (
    "example_id",
    "corpus_id",
    "text",
    "label",
    "patient_id",
    "note_id",
    "split",
    "leakage_group",
    "label_policy_version",
    "label_source",
    "retrieval_version",
    "example_mode",
    "label_scope",
    "crop_id",
)

OPTIONAL_COLUMNS = ("part_index", "part_count", "note_date", "unknown_reason")

ALLOWED_SPLITS = ("train", "dev", "test")
ALLOWED_LABEL_SOURCES = ("human", "adjudicated", "synthetic-reference", "teacher", "mixed")
ALLOWED_EXAMPLE_MODES = ("crop", "note-context")

_SCOPE_BY_MODE = {"crop": "crop", "note-context": "note"}


def _nonempty(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().ne("")


def _check_unique_rows(df: pd.DataFrame, key: list[str], label: str) -> None:
    duplicated = df.duplicated(subset=key, keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, key].head(5).to_dict("records")
        raise ValueError(f"Duplicate {label}: {sample}")


def validate_dataframe(
    df: pd.DataFrame,
    *,
    require_all_classes_per_split: bool = True,
    require_all_splits: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and normalize a dataset; return (clean_df, manifest)."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset is empty")

    clean = df.copy()
    for column in REQUIRED_COLUMNS:
        clean[column] = clean[column].astype(str).str.strip()

    empty_mask = ~clean[list(REQUIRED_COLUMNS)].apply(_nonempty).all(axis=1)
    if empty_mask.any():
        bad_ids = clean.loc[empty_mask, "example_id"].head(5).tolist()
        raise ValueError(f"Required fields must be nonempty; offending example_ids: {bad_ids}")

    bad_labels = sorted(set(clean["label"]) - set(LABELS))
    if bad_labels:
        raise ValueError(f"Invalid labels {bad_labels}; expected {list(LABELS)}")
    clean["label"] = clean["label"].map(validate_label)

    bad_splits = sorted(set(clean["split"]) - set(ALLOWED_SPLITS))
    if bad_splits:
        raise ValueError(f"Invalid splits {bad_splits}; expected {list(ALLOWED_SPLITS)}")
    if require_all_splits and set(clean["split"]) != set(ALLOWED_SPLITS):
        raise ValueError(f"All splits are required; found {sorted(set(clean['split']))}")

    bad_sources = sorted(set(clean["label_source"]) - set(ALLOWED_LABEL_SOURCES))
    if bad_sources:
        raise ValueError(f"Invalid label_source {bad_sources}; expected {list(ALLOWED_LABEL_SOURCES)}")

    bad_modes = sorted(set(clean["example_mode"]) - set(ALLOWED_EXAMPLE_MODES))
    if bad_modes:
        raise ValueError(f"Invalid example_mode {bad_modes}; expected {list(ALLOWED_EXAMPLE_MODES)}")

    policy_mismatch = clean["label_policy_version"] != POLICY_VERSION
    if policy_mismatch.any():
        found = sorted(set(clean.loc[policy_mismatch, "label_policy_version"]))
        raise ValueError(
            f"label_policy_version mismatch: found {found}, expected {POLICY_VERSION!r}; "
            "relabel or migrate before training"
        )

    scope_mismatch = clean.apply(
        lambda row: _SCOPE_BY_MODE[row["example_mode"]] != row["label_scope"], axis=1
    )
    if scope_mismatch.any():
        sample = clean.loc[scope_mismatch, ["example_id", "example_mode", "label_scope"]].head(5).to_dict("records")
        raise ValueError(f"example_mode/label_scope mismatch: {sample}")

    crop_mode = clean["example_mode"] == "crop"
    note_label_crop = crop_mode & (clean["label_scope"] == "note")
    if note_label_crop.any():
        raise ValueError("crop examples cannot carry note-scope labels")

    # A note-context row is the only example for its note part. Multiple parts
    # are allowed only when part_index is explicit and unique.
    note_context = clean[clean["example_mode"] == "note-context"]
    if not note_context.empty:
        key = ["corpus_id", "note_id", "crop_id"]
        _check_unique_rows(note_context, key, "note-context example key")

    _check_unique_rows(clean, ["example_id"], "example_id")

    # namespace identifiers: "123" in two corpora is not the same patient
    for column in ("patient_id", "note_id", "leakage_group"):
        namespaced = clean["corpus_id"] + "::" + clean[column]
        probe = clean.assign(_namespaced=namespaced)
        if probe.groupby("_namespaced")["split"].nunique().max() > 1:
            offenders = (
                probe.groupby("_namespaced")["split"].nunique()
                .loc[lambda counts: counts > 1].head(5).index.tolist()
            )
            raise ValueError(f"Cross-split leakage in {column}: {offenders}")

    if "unknown_reason" in clean.columns:
        clean["unknown_reason"] = clean["unknown_reason"].fillna("").astype(str).str.strip()
        bad_reasons = sorted(set(clean.loc[clean["unknown_reason"] != "", "unknown_reason"]) - set(UNKNOWN_REASONS))
        if bad_reasons:
            raise ValueError(f"Invalid unknown_reason values {bad_reasons}; expected {list(UNKNOWN_REASONS)}")
        missing_reason = clean["label"].eq("UNKNOWN") & clean["unknown_reason"].eq("")
        if missing_reason.any():
            sample = clean.loc[missing_reason, "example_id"].head(5).tolist()
            raise ValueError(f"UNKNOWN rows require unknown_reason; offending example_ids: {sample}")
        labeled_non_unknown = ~clean["label"].eq("UNKNOWN") & clean["unknown_reason"].ne("")
        if labeled_non_unknown.any():
            sample = clean.loc[labeled_non_unknown, "example_id"].head(5).tolist()
            raise ValueError(f"Non-UNKNOWN rows must not carry unknown_reason; offending example_ids: {sample}")

    if "part_index" in clean.columns or "part_count" in clean.columns:
        if not {"part_index", "part_count"}.issubset(clean.columns):
            raise ValueError("part_index and part_count must be provided together")
        part_index = pd.to_numeric(clean["part_index"], errors="coerce")
        part_count = pd.to_numeric(clean["part_count"], errors="coerce")
        invalid = part_index.isna() | part_count.isna() | (part_count < 1) | (part_index < 0) | (part_index >= part_count)
        if invalid.any():
            sample = clean.loc[invalid, "example_id"].head(5).tolist()
            raise ValueError(f"Invalid part_index/part_count; offending example_ids: {sample}")
        clean["part_index"] = part_index.astype(int)
        clean["part_count"] = part_count.astype(int)

    if require_all_classes_per_split:
        for split in ALLOWED_SPLITS:
            present = set(clean.loc[clean["split"] == split, "label"])
            if present != set(LABELS):
                raise ValueError(
                    f"All four classes must be represented in {split}; missing {sorted(set(LABELS) - present)}"
                )

    fingerprint_source = "\n".join(
        f"{row.example_id}|{row.label}|{row.split}"
        for row in clean.sort_values("example_id").itertuples()
    )
    manifest = {
        "policy_version": POLICY_VERSION,
        "n_rows": int(len(clean)),
        "n_unique_examples": int(clean["example_id"].nunique()),
        "n_patients": int((clean["corpus_id"] + "::" + clean["patient_id"]).nunique()),
        "n_notes": int((clean["corpus_id"] + "::" + clean["note_id"]).nunique()),
        "split_counts": {split: int((clean["split"] == split).sum()) for split in ALLOWED_SPLITS},
        "class_counts_by_split": {
            split: {label: int(((clean["split"] == split) & (clean["label"] == label)).sum()) for label in LABELS}
            for split in ALLOWED_SPLITS
        },
        "example_mode_counts": {mode: int((clean["example_mode"] == mode).sum()) for mode in ALLOWED_EXAMPLE_MODES},
        "label_source_counts": {
            source: int((clean["label_source"] == source).sum()) for source in ALLOWED_LABEL_SOURCES
        },
        "data_fingerprint": hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest(),
    }
    return clean.reset_index(drop=True), manifest


def read_and_validate_csv(
    path: str,
    *,
    require_all_classes_per_split: bool = True,
    require_all_splits: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return validate_dataframe(
        frame,
        require_all_classes_per_split=require_all_classes_per_split,
        require_all_splits=require_all_splits,
    )
