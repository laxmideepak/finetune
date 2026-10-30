#!/usr/bin/env python3
"""H5-DATE: native-status classifier + rule-based quit-time resolver (protocol:
experiments/h5-date-resolver/protocol.md).

Arms (scored on one split per call):
  A0                        TF-IDF crops -> LinearSVM trained on one-year labels
  A-native+R                TF-IDF crops -> LinearSVM on native labels, then resolver
  Oracle-status+R           gold native status, then resolver
  Oracle-status+Oracle-quit gold native status + answer-key quit fields (sanity)

Resolver development uses --split train (Oracle-status+R and resolver diagnostics
only, no classifier scoring). Dev is scored once the resolver is frozen.

    python scripts/run_h5_date_resolver.py --wright-dir ../smokingbenchmark --split train
    python scripts/run_h5_date_resolver.py --wright-dir ../smokingbenchmark --split dev \
        --out experiments/h5-date-resolver/results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_wright_smoke import (  # noqa: E402
    CONDITIONS,
    _bounds,
    _model,
    assign_splits,
    load,
)
from src.extract_windows import WhitespaceOffsetTokenizer, extract  # noqa: E402
from src.label_policy import status_from_cessation  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402
from src.quit_resolver import RESOLVER_VERSION, apply_one_year_rule, resolve_quit  # noqa: E402


def _crops(df: pd.DataFrame) -> pd.Series:
    tokenizer = WhitespaceOffsetTokenizer()
    return pd.Series(
        ["\n\n".join(c["text"] for c in extract(t, tokenizer)["crops"]) for t in df["text"]], index=df.index
    )


def _oracle_quit(row: pd.Series) -> str:
    if row["label_native"] != "FORMER":
        return row["label_native"]
    bounds = _bounds(row)
    if bounds is None:
        return "UNKNOWN"
    note_day = date.fromisoformat(row["note_date"])
    lo = status_from_cessation(max((note_day - bounds[1]).days, 0))[0]
    hi = status_from_cessation((note_day - bounds[0]).days)[0]
    return lo if lo == hi else "UNKNOWN"


def _resolve_all(df: pd.DataFrame, status: pd.Series, source: str) -> tuple[list[str], list[dict]]:
    labels, traces = [], []
    for idx, row in df.iterrows():
        resolution = resolve_quit(row[source], note_date=date.fromisoformat(row["note_date"]), age=int(row["age"]))
        label, reason = apply_one_year_rule(status[idx], resolution, note_date=date.fromisoformat(row["note_date"]))
        labels.append(label)
        traces.append({"patient_id": row["patient_id"], "status_in": status[idx], "label": label, "reason": reason, **resolution.as_dict()})
    return labels, traces


def _score(df: pd.DataFrame, pred) -> dict:
    pred = np.asarray(pred, dtype=object)
    payload = metrics_payload(df["label_one_year"].to_numpy(), pred)
    payload["by_condition_macro_f1"] = {
        c: metrics_payload(df.loc[df["condition"] == c, "label_one_year"].to_numpy(), pred[(df["condition"] == c).to_numpy()])["macro_f1"]
        for c in CONDITIONS
    }
    timing = (df["unknown_reason_one_year"] == "timing-unresolved").to_numpy()
    recent = ((df["label_native"] == "FORMER") & (df["label_one_year"] == "CURRENT")).to_numpy()
    payload["timing_unresolved_recall"] = float((pred[timing] == "UNKNOWN").mean()) if timing.any() else None
    payload["recent_quitter_recall"] = f"{int((pred[recent] == 'CURRENT').sum())}/{int(recent.sum())}"
    return payload


def _resolver_diagnostics(df: pd.DataFrame, traces: list[dict]) -> dict:
    """Resolver accuracy against the answer key on gold-FORMER notes."""
    t = pd.DataFrame(traces).set_index("patient_id")
    former = df[df["label_native"] == "FORMER"].set_index("patient_id")
    rows = []
    for pid, row in former.iterrows():
        tr = t.loc[pid]
        has_key = pd.notna(row["quit_year_min"])
        found = tr["earliest"] is not None
        ok = None
        if has_key and found:
            ok = int(row["quit_year_min"]) <= tr["earliest"].year and tr["latest"].year <= int(row["quit_year_max"])
        rows.append({"patient_id": pid, "condition": row["condition"], "source": row["quit_date_source"], "has_key": has_key, "found": found, "year_ok": ok})
    d = pd.DataFrame(rows)
    by_source = (
        d.groupby("source")
        .agg(n=("found", "size"), found=("found", "sum"), year_ok=("year_ok", lambda s: int(sum(bool(x) for x in s if x is not None))))
        .sort_values("n", ascending=False)
    )
    keyed = d[d["has_key"]]
    unkeyed = d[~d["has_key"]]
    return {
        "former_notes": int(len(d)),
        "with_answer_key_quit": int(len(keyed)),
        "coverage_on_keyed": float(keyed["found"].mean()) if len(keyed) else None,
        "year_accuracy_on_keyed_found": float(keyed.loc[keyed["found"], "year_ok"].astype(bool).mean()) if keyed["found"].any() else None,
        "false_finds_on_unkeyed": int(unkeyed["found"].sum()),
        "by_source": by_source.reset_index().to_dict("records"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--source", choices=("crop_text", "text"), default="crop_text", help="text the resolver reads")
    parser.add_argument("--out", default=None)
    parser.add_argument("--show-errors", type=int, default=0, help="print N resolver errors (train only)")
    args = parser.parse_args(argv)
    if args.split != "train" and args.show_errors:
        parser.error("--show-errors is only allowed on the train split")

    df = load(Path(args.wright_dir))
    df["split"] = assign_splits(df)
    df["crop_text"] = _crops(df)
    part = df[df["split"] == args.split].copy()

    results: dict = {"resolver_version": RESOLVER_VERSION, "split": args.split, "resolver_source": args.source, "arms": {}}
    sanity = part.apply(_oracle_quit, axis=1)
    results["arms"]["Oracle-status+Oracle-quit"] = _score(part, sanity)
    oracle_pred, traces = _resolve_all(part, part["label_native"], args.source)
    results["arms"]["Oracle-status+R"] = _score(part, oracle_pred)
    results["resolver_diagnostics"] = _resolver_diagnostics(part, traces)

    if args.split == "dev":
        train = df[df["split"] == "train"]
        fill = lambda s: s.where(s.str.len() > 0, "__no_crop__")  # noqa: E731
        no_crop = (part["crop_text"].str.len() == 0).to_numpy()
        a0 = _model("svm").fit(fill(train["crop_text"]), train["label_one_year"]).predict(fill(part["crop_text"])).astype(object)
        a0[no_crop] = "UNKNOWN"
        results["arms"]["A0"] = _score(part, a0)
        native = _model("svm").fit(fill(train["crop_text"]), train["label_native"]).predict(fill(part["crop_text"])).astype(object)
        native[no_crop] = "UNKNOWN"
        hybrid, _ = _resolve_all(part, pd.Series(native, index=part.index), args.source)
        results["arms"]["A-native+R"] = _score(part, hybrid)

    for name, payload in results["arms"].items():
        print(
            f"{name:28s} macroF1={payload['macro_f1']:.3f} CUR_R={payload['current_recall']:.3f} "
            f"timingUNK_R={payload['timing_unresolved_recall']} recent={payload['recent_quitter_recall']} "
            f"by_cond={ {k: round(v, 3) for k, v in payload['by_condition_macro_f1'].items()} }"
        )
    diag = results["resolver_diagnostics"]
    print(json.dumps({k: v for k, v in diag.items() if k != "by_source"}, indent=1))
    for r in diag["by_source"]:
        print(f"  {r['source']:38s} n={r['n']:4d} found={r['found']:4d} year_ok={r['year_ok']:4d}")

    if args.show_errors:
        t = pd.DataFrame(traces).set_index("patient_id")
        bad = part[(np.asarray(oracle_pred, dtype=object) != part["label_one_year"].to_numpy())]
        for _, row in bad.head(args.show_errors).iterrows():
            tr = t.loc[row["patient_id"]]
            print(f"--- {row['patient_id']} truth={row['label_one_year']} pred={tr['label']} key={row['quit_date']}({row['quit_date_source']}) "
                  f"note={row['note_date']} found={tr['matched_text']!r} [{tr['earliest']}..{tr['latest']}] rule={tr['rule']}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"h5_{args.split}_{args.source}.json").write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
