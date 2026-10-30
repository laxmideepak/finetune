#!/usr/bin/env python3
"""H6-EVIDENCE: native-status classifier on evidence sentences vs merged crops
(protocol: experiments/h6-evidence-sentences/protocol.md).

    python scripts/run_h6_evidence.py --wright-dir ../smokingbenchmark --mode cv
    python scripts/run_h6_evidence.py --wright-dir ../smokingbenchmark --mode dev \
        --out experiments/h6-evidence-sentences/results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_h5_date_resolver import _score, one_year_label_v2  # noqa: E402
from run_wright_smoke import CONDITIONS, _model, assign_splits, load  # noqa: E402
from src.evidence import evidence_text  # noqa: E402
from src.extract_windows import WhitespaceOffsetTokenizer, extract  # noqa: E402
from src.quit_resolver import RESOLVER_VERSION, apply_one_year_rule, resolve_quit  # noqa: E402

VIEWS = {"S-crop": "crop_text", "S-evid": "evid0", "S-evid+1": "evid1"}


def build_views(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = WhitespaceOffsetTokenizer()
    crop, e0, e1 = [], [], []
    for note in df["text"]:
        result = extract(note, tokenizer)
        crop.append("\n\n".join(c["text"] for c in result["crops"]))
        e0.append(evidence_text(note, result, context=0))
        e1.append(evidence_text(note, result, context=1))
    df = df.copy()
    df["crop_text"], df["evid0"], df["evid1"] = crop, e0, e1
    return df


def _fill(s: pd.Series) -> pd.Series:
    return s.where(s.str.len() > 0, "__no_evidence__")


def _resolve(df: pd.DataFrame, status) -> list[str]:
    out = []
    for (_, row), st in zip(df.iterrows(), status):
        note_day = date.fromisoformat(row["note_date"])
        res = resolve_quit(row["crop_text"], note_date=note_day, age=int(row["age"]))
        out.append(apply_one_year_rule(st, res, note_date=note_day)[0])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--mode", choices=("cv", "dev"), required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    df = load(Path(args.wright_dir))
    df["split"] = assign_splits(df)
    df["label_one_year_v2"] = df.apply(one_year_label_v2, axis=1)
    df = build_views(df)
    train = df[df["split"] == "train"]
    results: dict = {"mode": args.mode, "resolver_version": RESOLVER_VERSION, "arms": {}}

    for arm, column in VIEWS.items():
        empty_train = int((train[column].str.len() == 0).sum())
        if args.mode == "cv":
            cv = StratifiedKFold(5, shuffle=True, random_state=0)
            pred = cross_val_predict(_model("svm"), _fill(train[column]), train["label_native"], cv=cv).astype(object)
            pred[(train[column].str.len() == 0).to_numpy()] = "UNKNOWN"
            acc = {c: float((pred[(train["condition"] == c).to_numpy()] == train.loc[train["condition"] == c, "label_native"]).mean()) for c in CONDITIONS}
            results["arms"][arm] = {"native_accuracy_by_condition": acc, "native_accuracy": float((pred == train["label_native"]).mean()), "empty_inputs": empty_train}
            print(f"{arm:10s} native acc={results['arms'][arm]['native_accuracy']:.4f} by_cond={ {k: round(v, 4) for k, v in acc.items()} } empty={empty_train}")
        else:
            dev = df[df["split"] == "dev"]
            model = _model("svm").fit(_fill(train[column]), train["label_native"])
            native = model.predict(_fill(dev[column])).astype(object)
            native[(dev[column].str.len() == 0).to_numpy()] = "UNKNOWN"
            hybrid = _resolve(dev, native)
            for version, label in (("v1", "label_one_year"), ("v2", "label_one_year_v2")):
                payload = _score(dev, hybrid, label)
                payload["native_accuracy"] = float((native == dev["label_native"]).mean())
                results["arms"][f"{arm}+R|{version}"] = payload
                print(
                    f"{arm + '+R|' + version:16s} macroF1={payload['macro_f1']:.3f} nativeAcc={payload['native_accuracy']:.3f} "
                    f"CUR_R={payload['current_recall']:.3f} CUR_P={payload['current_precision']:.3f} "
                    f"by_cond={ {k: round(v, 3) for k, v in payload['by_condition_macro_f1'].items()} }"
                )
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"h6_{args.mode}.json").write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
