#!/usr/bin/env python3
"""E0 + E3 zero-DUA smoke on the Wright synthetic benchmark (pipeline-design.md §14).

E0  retrieval sanity: run the Stage-3 extractor over all 3,000 notes and report
    crops/note, crop-token distribution, merge/truncation/overflow counts, and the
    retrieval false-negative rate (notes with a known status but no crop).
E3  arm-A floor: TF-IDF word+char n-grams -> logistic regression / linear SVM,
    trained on crop text (trigger-anchored) and on the full note (tail-free
    baseline), under two label views:
      * native   - Wright's own labels (any documented quit => FORMER)
      * one-year - our ratified policy (smoking-status-v1) derived from
                   quit_date/quit_year_{min,max} vs note_date via
                   label_policy.status_from_cessation; unresolvable timing
                   => UNKNOWN(timing-unresolved)
    The native view is pipeline-comparability only; the one-year view is the
    label-policy sensitivity slice. Neither is gold (LLM-adjudicated references).

Only dev metrics are reported; the test split is written but never scored here.

    python scripts/run_wright_smoke.py --wright-dir /path/to/smokingbenchmark \
        --out experiments/E0_E3_wright

The tokenizer is the offline whitespace tokenizer (Hugging Face Hub is not
reachable from every environment); crop token counts are therefore word counts,
not ModernBERT word pieces. Re-run with --tokenizer once the Hub is reachable.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extract_windows import RETRIEVAL_VERSION, WhitespaceOffsetTokenizer, extract  # noqa: E402
from src.label_policy import POLICY_VERSION, status_from_cessation  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402

CONDITIONS = ("basic", "messy", "complex")
NATIVE_MAP = {"current": "CURRENT", "former": "FORMER", "never": "NEVER", "unknown": "UNKNOWN"}
SPLIT_SEED = 20261029


def _bounds(row: pd.Series) -> tuple[date, date] | None:
    """Earliest/latest possible cessation date supported by the answer key."""
    quit_date = str(row.get("quit_date") or "").strip()
    precision = str(row.get("quit_date_precision") or "").strip()
    if quit_date in ("", "nan"):
        return None
    parts = quit_date.split("-")
    if precision == "day" and len(parts) == 3:
        day = date(int(parts[0]), int(parts[1]), int(parts[2]))
        return day, day
    if precision == "month" and len(parts) >= 2:
        year, month = int(parts[0]), int(parts[1])
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    lo = row.get("quit_year_min")
    hi = row.get("quit_year_max")
    lo = int(lo) if pd.notna(lo) else int(parts[0])
    hi = int(hi) if pd.notna(hi) else int(parts[0])
    return date(lo, 1, 1), date(hi, 12, 31)


def one_year_label(row: pd.Series) -> tuple[str, str | None]:
    native = NATIVE_MAP[row["smoking_status"]]
    if native != "FORMER":
        return native, ("no-mention" if native == "UNKNOWN" else None)
    bounds = _bounds(row)
    if bounds is None:
        return status_from_cessation(None)
    note_day = date.fromisoformat(row["note_date"])
    earliest, latest = bounds
    longest = (note_day - earliest).days
    shortest = max((note_day - latest).days, 0)
    shortest_label, _ = status_from_cessation(shortest)
    longest_label, _ = status_from_cessation(longest)
    if shortest_label == longest_label:
        return shortest_label, None
    return status_from_cessation(None)


def load(wright_dir: Path) -> pd.DataFrame:
    frames = []
    for condition in CONDITIONS:
        answers = pd.read_csv(wright_dir / condition / "answers.csv", dtype={"quit_date": str})
        answers["condition"] = condition
        answers["text"] = [
            (wright_dir / condition / "notes" / f"{pid}.txt").read_text(encoding="utf-8")
            for pid in answers["patient_id"]
        ]
        frames.append(answers)
    df = pd.concat(frames, ignore_index=True)
    df["label_native"] = df["smoking_status"].map(NATIVE_MAP)
    mapped = df.apply(one_year_label, axis=1)
    df["label_one_year"] = [m[0] for m in mapped]
    df["unknown_reason_one_year"] = [m[1] for m in mapped]
    return df


def assign_splits(df: pd.DataFrame) -> pd.Series:
    """Patient-disjoint 60/20/20, stratified by condition x native label."""
    rng = np.random.default_rng(SPLIT_SEED)
    split = pd.Series("train", index=df.index)
    for _, group in df.groupby(["condition", "label_native"]):
        idx = rng.permutation(group.index.to_numpy())
        n_dev = int(round(0.2 * len(idx)))
        n_test = int(round(0.2 * len(idx)))
        split[idx[:n_dev]] = "dev"
        split[idx[n_dev : n_dev + n_test]] = "test"
    return split


def run_e0(df: pd.DataFrame, tokenizer) -> tuple[pd.DataFrame, dict]:
    rows = []
    started = time.perf_counter()
    for record in df.itertuples():
        result = extract(record.text, tokenizer)
        stats = result["stats"]
        rows.append(
            {
                "patient_id": record.patient_id,
                "crop_text": "\n\n".join(crop["text"] for crop in result["crops"]),
                "n_crops": len(result["crops"]),
                "n_wider": len(result["wider_context"]),
                "n_triggers": stats["n_triggers"],
                "n_triggers_used": stats["n_triggers_used"],
                "n_clusters": stats["n_clusters"],
                "note_tokens": stats["note_tokens"],
                "crop_tokens": [crop["n_tokens"] for crop in result["crops"]],
                "truncated_crops": stats["truncated_crops"],
                "boilerplate_sentences": stats["n_boilerplate_sentences"],
                "within_section_duplicates": stats["n_within_section_duplicates"],
                "unpopulated_templates": stats["n_unpopulated_templates"],
            }
        )
    elapsed = time.perf_counter() - started
    crops = pd.DataFrame(rows)
    merged = df[["patient_id", "condition", "label_native", "label_one_year"]].merge(crops, on="patient_id")
    all_tokens = [t for tokens in merged["crop_tokens"] for t in tokens]

    def per(group: pd.DataFrame) -> dict:
        tokens = [t for ts in group["crop_tokens"] for t in ts]
        known = group[group["label_native"] != "UNKNOWN"]
        return {
            "notes": int(len(group)),
            "crops_per_note_mean": float(group["n_crops"].mean()),
            "notes_without_crop": int((group["n_crops"] == 0).sum()),
            "retrieval_fn_rate_known_status": float((known["n_crops"] == 0).mean()) if len(known) else None,
            "triggers_per_note_mean": float(group["n_triggers"].mean()),
            "merge_ratio_triggers_used_per_crop": float(group["n_triggers_used"].sum() / max(group["n_crops"].sum(), 1)),
            "crop_tokens_p50": float(np.percentile(tokens, 50)) if tokens else None,
            "crop_tokens_p95": float(np.percentile(tokens, 95)) if tokens else None,
            "crop_tokens_max": int(max(tokens)) if tokens else 0,
            "crops_over_256": int(sum(t > 256 for t in tokens)),
            "crops_over_512": int(sum(t > 512 for t in tokens)),
            "truncated_crops": int(group["truncated_crops"].sum()),
            "notes_with_wider_context": int((group["n_wider"] > 0).sum()),
            "note_tokens_p50": float(group["note_tokens"].median()),
            "crop_to_note_token_ratio": float(sum(tokens) / max(group["note_tokens"].sum(), 1)),
            "boilerplate_sentences_dropped": int(group["boilerplate_sentences"].sum()),
            "within_section_duplicates_dropped": int(group["within_section_duplicates"].sum()),
            "unpopulated_templates": int(group["unpopulated_templates"].sum()),
        }

    report = {
        "retrieval_version": RETRIEVAL_VERSION,
        "tokenizer": getattr(tokenizer, "name_or_path", str(tokenizer)),
        "extract_seconds_total": round(elapsed, 2),
        "extract_notes_per_second": round(len(df) / elapsed, 1),
        "overall": per(merged),
        "by_condition": {c: per(g) for c, g in merged.groupby("condition")},
        "retrieval_fn_examples": merged.loc[
            (merged["n_crops"] == 0) & (merged["label_native"] != "UNKNOWN"), ["patient_id", "label_native"]
        ].head(25).to_dict("records"),
        "crop_tokens_all_p50": float(np.percentile(all_tokens, 50)) if all_tokens else None,
    }
    return crops, report


def _model(kind: str):
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    if kind == "lr":
        clf = LogisticRegression(C=10.0, max_iter=4000, class_weight="balanced")
    else:
        clf = LinearSVC(C=1.0, class_weight="balanced")
    return make_pipeline(features, clf)


def run_e3(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    train = df[df["split"] == "train"]
    dev = df[df["split"] == "dev"]
    results: dict = {}
    predictions = dev[["patient_id", "condition", "label_native", "label_one_year", "n_crops"]].copy()
    for label_view in ("label_native", "label_one_year"):
        for input_view in ("crop_text", "text"):
            for kind in ("lr", "svm"):
                name = f"{label_view.replace('label_', '')}|{'crops' if input_view == 'crop_text' else 'full-note'}|{kind}"
                started = time.perf_counter()
                model = _model(kind)
                train_text = train[input_view].where(train[input_view].str.len() > 0, "__no_crop__")
                dev_text = dev[input_view].where(dev[input_view].str.len() > 0, "__no_crop__")
                model.fit(train_text, train[label_view])
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                pred = np.asarray(model.predict(dev_text), dtype=object)
                predict_seconds = time.perf_counter() - started
                if input_view == "crop_text":
                    # Stage 6 rule: a retrieval-negative note is UNKNOWN(retrieval-negative).
                    pred[(dev["n_crops"] == 0).to_numpy()] = "UNKNOWN"
                predictions[name] = pred
                payload = metrics_payload(dev[label_view].to_numpy(), pred)
                payload["by_condition_macro_f1"] = {
                    c: metrics_payload(dev.loc[dev["condition"] == c, label_view].to_numpy(), pred[(dev["condition"] == c).to_numpy()])["macro_f1"]
                    for c in CONDITIONS
                }
                payload["fit_seconds"] = round(fit_seconds, 2)
                payload["dev_predict_seconds"] = round(predict_seconds, 3)
                payload["dev_notes_per_second"] = round(len(dev) / max(predict_seconds, 1e-9), 1)
                results[name] = payload
    return results, predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wright-dir", required=True, help="clone of github.com/vclic/smokingbenchmark")
    parser.add_argument("--out", default="experiments/E0_E3_wright")
    parser.add_argument("--tokenizer", default=None, help="optional pinned HF fast tokenizer (needs Hub access)")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    else:
        tokenizer = WhitespaceOffsetTokenizer()

    df = load(Path(args.wright_dir))
    df["split"] = assign_splits(df)
    label_report = {
        "policy_version": POLICY_VERSION,
        "native_counts": df["label_native"].value_counts().to_dict(),
        "one_year_counts": df["label_one_year"].value_counts().to_dict(),
        "native_to_one_year": pd.crosstab(df["label_native"], df["label_one_year"]).to_dict(),
        "one_year_unknown_reasons": df["unknown_reason_one_year"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
    }

    crops, e0 = run_e0(df, tokenizer)
    df = df.merge(crops[["patient_id", "crop_text", "n_crops"]], on="patient_id")
    e3, predictions = run_e3(df)

    df[["patient_id", "condition", "split", "label_native", "label_one_year", "unknown_reason_one_year"]].to_csv(
        out / "labels_and_splits.csv", index=False
    )
    predictions.to_csv(out / "dev_predictions.csv", index=False)
    (out / "labels.json").write_text(json.dumps(label_report, indent=2, default=str))
    (out / "e0_retrieval.json").write_text(json.dumps(e0, indent=2))
    (out / "e3_tfidf_dev.json").write_text(json.dumps(e3, indent=2))

    print(json.dumps(label_report, indent=2, default=str))
    print(json.dumps({k: v for k, v in e0.items() if k != "retrieval_fn_examples"}, indent=2))
    for name, payload in e3.items():
        print(
            f"{name:28s} macroF1={payload['macro_f1']:.3f} acc={payload['accuracy_all']:.3f} "
            f"CUR_R={payload['current_recall']} CUR_P={payload['current_precision']} "
            f"by_cond={ {k: round(v, 3) for k, v in payload['by_condition_macro_f1'].items()} }"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
