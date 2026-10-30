#!/usr/bin/env python3
"""H16: compare Flan-T5-base fine-tuned with like-for-like TF-IDF on repeated 5-fold splits
(protocol: experiments/h16-repeated-cv/protocol.md). Reads Flan fold outputs written by
run_h9_flan_cv.py with --tag base-cv{r} --cv-seed r.

    python scripts/run_h16_compare.py --sdoh-dir ../SDOH-NLI --out experiments/h16-repeated-cv/results/h16.json
"""

from __future__ import annotations

import argparse
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

LABELS4 = ["CURRENT", "FORMER", "NEVER", "UNKNOWN"]


def macro(y, p, labels):
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def three(v):
    return np.array(["OTHER" if x in ("NEVER", "UNKNOWN") else x for x in v], dtype=object)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--results", default="experiments/h16-repeated-cv/results")
    parser.add_argument("--repeats", default="1,2,3")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from sklearn.model_selection import StratifiedKFold

    s = add_evidence(load_sdoh(Path(args.sdoh_dir), "train")).reset_index(drop=True)
    y = s["label_native"].to_numpy()
    has = (s["evid0"].str.len() > 0).to_numpy()
    rng = np.random.default_rng(20261030)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(5000)]
    report = {"repeats": {}}
    for r in [int(x) for x in args.repeats.split(",")]:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=r).split(s, y))
        flan = np.full(len(y), None, dtype=object)
        tfidf = np.full(len(y), "UNKNOWN", dtype=object)
        fold_rows = []
        for k, (tr, te) in enumerate(folds):
            path = Path(args.results) / f"base-cv{r}_ft-S_fold{k}.json"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            flan[te] = json.loads(path.read_text())["pred"]
            a = s.iloc[tr]
            a = a[a["evid0"].str.len() > 0]
            t = te[has[te]]
            tfidf[t] = _model("svm").fit(a["evid0"], a["label_native"]).predict(s["evid0"].iloc[t])
            fold_rows.append({"fold": k, "flan": round(macro(y[te], flan[te], LABELS4), 4), "tfidf": round(macro(y[te], tfidf[te], LABELS4), 4)})
        d4 = np.array([macro(y[i], flan[i], LABELS4) - macro(y[i], tfidf[i], LABELS4) for i in boots])
        y3, f3, t3 = three(y), three(flan), three(tfidf)
        d3 = np.array([macro(y3[i], f3[i], ["CURRENT", "FORMER", "OTHER"]) - macro(y3[i], t3[i], ["CURRENT", "FORMER", "OTHER"]) for i in boots])
        rep = {
            "flan_pooled": round(macro(y, flan, LABELS4), 4),
            "tfidf_pooled": round(macro(y, tfidf, LABELS4), 4),
            "delta_4class": round(macro(y, flan, LABELS4) - macro(y, tfidf, LABELS4), 4),
            "ci95_4class": [round(float(np.percentile(d4, 2.5)), 4), round(float(np.percentile(d4, 97.5)), 4)],
            "delta_3class": round(macro(y3, f3, ["CURRENT", "FORMER", "OTHER"]) - macro(y3, t3, ["CURRENT", "FORMER", "OTHER"]), 4),
            "ci95_3class": [round(float(np.percentile(d3, 2.5)), 4), round(float(np.percentile(d3, 97.5)), 4)],
            "folds": fold_rows,
            "folds_won": int(sum(f["flan"] > f["tfidf"] for f in fold_rows)),
        }
        report["repeats"][r] = rep
        print(f"repeat {r}: Flan {rep['flan_pooled']:.3f} vs TF-IDF {rep['tfidf_pooled']:.3f}  Δ4={rep['delta_4class']:+.3f} {rep['ci95_4class']}  Δ3={rep['delta_3class']:+.3f} {rep['ci95_3class']}  folds won {rep['folds_won']}/5")
    reps = list(report["repeats"].values())
    report["summary"] = {
        "mean_delta_4class": round(float(np.mean([r["delta_4class"] for r in reps])), 4),
        "all_repeats_positive_4class": all(r["delta_4class"] > 0 for r in reps),
        "all_repeats_positive_3class": all(r["delta_3class"] > 0 for r in reps),
        "folds_won_total": int(sum(r["folds_won"] for r in reps)),
        "folds_total": 5 * len(reps),
    }
    print(json.dumps(report["summary"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
