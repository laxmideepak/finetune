#!/usr/bin/env python3
"""H13-CASCADE: TF-IDF first, Flan-T5-base only on low-margin sentences
(protocol: experiments/h13-cascade/protocol.md). Uses H9 out-of-fold Flan predictions.

    python scripts/run_h13_cascade.py --sdoh-dir ../SDOH-NLI --out experiments/h13-cascade/results.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_h7_ood import add_evidence, load_sdoh  # noqa: E402
from run_wright_smoke import _model  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402

FLAN_USD_PER_M = 5.08  # H12 measurement, 4 threads, assumed $0.17/h
TFIDF_USD_PER_M = 0.04  # pipeline_cli bench (per note; used as the per-sentence floor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from sklearn.model_selection import StratifiedKFold

    s = add_evidence(load_sdoh(Path(args.sdoh_dir), "train")).reset_index(drop=True)
    y = s["label_native"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(s, y))
    flan = np.full(len(y), None, dtype=object)
    for f in glob.glob("experiments/h9-flan-t5/results/base_ft-S_fold*.json"):
        d = json.loads(Path(f).read_text())
        flan[folds[d["fold"]][1]] = d["pred"]
    tfidf = np.full(len(y), "UNKNOWN", dtype=object)
    margin = np.full(len(y), np.inf)
    has = (s["evid0"].str.len() > 0).to_numpy()
    for tr, te in folds:
        a = s.iloc[tr]
        a = a[a["evid0"].str.len() > 0]
        model = _model("svm").fit(a["evid0"], a["label_native"])
        te_has = te[has[te]]
        scores = model.decision_function(s["evid0"].iloc[te_has])
        top2 = np.sort(scores, axis=1)[:, -2:]
        tfidf[te_has] = model.classes_[scores.argmax(1)]
        margin[te_has] = top2[:, 1] - top2[:, 0]

    base = metrics_payload(y, tfidf)["macro_f1"]
    full = metrics_payload(y, flan)["macro_f1"]
    rows = []
    for budget in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0):
        pred = tfidf.copy()
        for _, te in folds:
            te_has = te[has[te]]
            k = int(round(budget * len(te_has)))
            route = te_has[np.argsort(margin[te_has])[:k]]
            pred[route] = flan[route]
        m = metrics_payload(y, pred)
        gain = (m["macro_f1"] - base) / (full - base) if full != base else float("nan")
        rows.append({"flan_call_share": budget, "pooled_macro_f1": round(m["macro_f1"], 4), "NEVER_f1": round(m["per_class"]["NEVER"]["f1"], 3),
                     "share_of_flan_gain": round(gain, 3), "usd_per_million_sentences": round(TFIDF_USD_PER_M + budget * FLAN_USD_PER_M, 2)})
        print(f"Flan on {int(budget * 100):3d}% -> macroF1 {m['macro_f1']:.3f}  NEVER {m['per_class']['NEVER']['f1']:.2f}  gain kept {gain:5.0%}  ~${rows[-1]['usd_per_million_sentences']}/M")
    if args.out:
        Path(args.out).write_text(json.dumps({"tfidf": base, "flan": full, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
