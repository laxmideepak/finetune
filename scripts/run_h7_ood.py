#!/usr/bin/env python3
"""H7-OOD: transfer of the evidence-sentence status classifier to SDOH-NLI
(protocol: experiments/h7-ood-sdoh-nli/protocol.md).

    python scripts/run_h7_ood.py --wright-dir ../smokingbenchmark --sdoh-dir ../SDOH-NLI \
        --eval validation --out experiments/h7-ood-sdoh-nli/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_wright_smoke import _model, assign_splits, load  # noqa: E402
from src.evidence import evidence_text  # noqa: E402
from src.extract_windows import WhitespaceOffsetTokenizer, extract  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402
from src.status_rules import rule_status  # noqa: E402

H_CUR = "The person is currently a smoker."
H_NOTCUR = "The person is currently not a smoker."
H_PAST = "The person was a smoker in the past."
H_NOTPAST = "The person wasn't a smoker in the past."


def sdoh_label(row: pd.Series) -> tuple[str, str | None]:
    cur, notcur, past, notpast = (bool(row[h]) for h in (H_CUR, H_NOTCUR, H_PAST, H_NOTPAST))
    if cur:
        return "CURRENT", None
    if notcur and past and not notpast:
        return "FORMER", None
    if notcur and notpast and not past:
        return "NEVER", None
    if notcur and not past and not notpast:
        return "UNKNOWN", "denial-only"
    if past and not notcur:
        return "UNKNOWN", "past-use-current-unstated"
    if not (cur or notcur or past or notpast):
        return "UNKNOWN", "no-mention"
    return "UNKNOWN", "conflict"


def load_sdoh(sdoh_dir: Path, split: str) -> pd.DataFrame:
    raw = pd.read_csv(sdoh_dir / f"sdoh_nli_{split}.csv")
    raw = raw[raw["hypothesis"].isin([H_CUR, H_NOTCUR, H_PAST, H_NOTPAST])]
    raw["label"] = raw["label"].astype(str).str.lower().eq("true")
    wide = raw.pivot_table(index="premise", columns="hypothesis", values="label", aggfunc="first").fillna(False).reset_index()
    mapped = wide.apply(sdoh_label, axis=1)
    wide["label_native"] = [m[0] for m in mapped]
    wide["reason"] = [m[1] for m in mapped]
    wide = wide.rename(columns={"premise": "text"})
    wide["condition"] = "sdoh"
    return wide


def add_evidence(df: pd.DataFrame) -> pd.DataFrame:
    tokenizer = WhitespaceOffsetTokenizer()
    df = df.copy()
    df["evid0"] = [evidence_text(t, extract(t, tokenizer), context=0) for t in df["text"]]
    return df


def _fill(s: pd.Series) -> pd.Series:
    return s.where(s.str.len() > 0, "__no_evidence__")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = _model("svm").fit(_fill(train["evid0"]), train["label_native"])
    pred = model.predict(_fill(test["evid0"])).astype(object)
    pred[(test["evid0"].str.len() == 0).to_numpy()] = "UNKNOWN"
    return pred


def paired_bootstrap(y, a, b, n=2000, seed=20261030) -> dict:
    from src.metrics import confusion, macro_f1

    rng = np.random.default_rng(seed)
    y, a, b = (np.asarray(v, dtype=object) for v in (y, a, b))
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        deltas.append(macro_f1(confusion(y[idx], a[idx])) - macro_f1(confusion(y[idx], b[idx])))
    deltas = np.asarray(deltas)
    return {"mean_delta": float(deltas.mean()), "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))], "p_le_0": float((deltas <= 0).mean())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--eval", choices=("validation", "test"), default="validation")
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if args.eval == "test" and not args.confirm_locked_test:
        parser.error("the SDOH-NLI test split is locked; pass --confirm-locked-test for the final run only")

    wright = load(Path(args.wright_dir))
    wright["split"] = assign_splits(wright)
    wright = add_evidence(wright)
    w_train, w_dev = wright[wright["split"] == "train"], wright[wright["split"] == "dev"]
    s_train = add_evidence(load_sdoh(Path(args.sdoh_dir), "train"))
    s_eval = add_evidence(load_sdoh(Path(args.sdoh_dir), args.eval))
    both = pd.concat([w_train[["evid0", "label_native"]], s_train[["evid0", "label_native"]]], ignore_index=True)

    smoking_bearing = s_eval["label_native"].ne("UNKNOWN") | s_eval["reason"].ne("no-mention")
    results: dict = {
        "eval": args.eval,
        "sdoh_label_counts": {"train": s_train["label_native"].value_counts().to_dict(), args.eval: s_eval["label_native"].value_counts().to_dict()},
        "sdoh_unknown_reasons": s_eval["reason"].value_counts().to_dict(),
        "retrieval_negative_on_smoking_bearing": int(((s_eval["evid0"].str.len() == 0) & smoking_bearing).sum()),
        "smoking_bearing_eval_premises": int(smoking_bearing.sum()),
        "arms": {},
    }
    preds = {
        "W->S": fit_predict(w_train, s_eval),
        "S->S": fit_predict(s_train, s_eval),
        "W+S->S": fit_predict(both, s_eval),
        "Rules": np.asarray([rule_status(e) for e in s_eval["evid0"]], dtype=object),
    }
    y = s_eval["label_native"].to_numpy()
    for name, pred in preds.items():
        payload = metrics_payload(y, pred)
        payload["macro_f1_smoking_bearing"] = metrics_payload(y[smoking_bearing.to_numpy()], pred[smoking_bearing.to_numpy()])["macro_f1"]
        results["arms"][name] = payload
    reg = fit_predict(both, w_dev)
    results["arms"]["W+S->W(dev)"] = metrics_payload(w_dev["label_native"].to_numpy(), reg)
    results["paired_bootstrap_W+S_minus_S"] = paired_bootstrap(y, preds["W+S->S"], preds["S->S"])
    results["paired_bootstrap_W+S_minus_W"] = paired_bootstrap(y, preds["W+S->S"], preds["W->S"])

    print(json.dumps({k: v for k, v in results.items() if k != "arms"}, indent=1))
    for name, p in results["arms"].items():
        f1s = {k: round(v["f1"], 3) for k, v in p["per_class"].items()}
        extra = f" smokingBearingF1={p['macro_f1_smoking_bearing']:.3f}" if "macro_f1_smoking_bearing" in p else ""
        print(f"{name:12s} n={p['n']} macroF1={p['macro_f1']:.3f}{extra} CUR_R={p['current_recall']} CUR_P={p['current_precision']} per-class={f1s}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"h7_{args.eval}.json").write_text(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
