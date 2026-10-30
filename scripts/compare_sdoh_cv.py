#!/usr/bin/env python3
"""Pool out-of-fold predictions from the SDOH-NLI 5-fold experiments (H8, H8b, H9, H10)
and compare every arm with TF-IDF on the same premises (paired bootstrap over premises).

    python scripts/compare_sdoh_cv.py --sdoh-dir ../SDOH-NLI --out experiments/sdoh-cv-leaderboard.json
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
from run_h10_guard import guard  # noqa: E402
from src.metrics import confusion, macro_f1, metrics_payload  # noqa: E402


def collect(n: int) -> dict[str, np.ndarray]:
    arms: dict[str, np.ndarray] = {}

    def put(name: str, ids, pred) -> None:
        arms.setdefault(name, np.full(n, None, dtype=object))[np.asarray(ids)] = pred

    folds = {}
    for f in sorted(glob.glob("experiments/h8-pretrained-encoder-cpu/results/fold*.json")):
        d = json.loads(Path(f).read_text())
        folds[d["fold"]] = d["test_ids"]
        for arm, v in d["arms"].items():
            put({"tfidf-S": "TF-IDF (S)", "tfidf-WS": "TF-IDF (W+S)", "enc-S": "SciBERT generic (S)", "enc-WS": "SciBERT generic (W+S)"}[arm], d["test_ids"], v["pred"])
    for seed_dir in sorted(glob.glob("experiments/h8b-encoder-small-data-recipe/results/seed*")):
        for f in sorted(glob.glob(f"{seed_dir}/fold*.json")):
            d = json.loads(Path(f).read_text())
            if "enc-S" in d["arms"]:
                put(f"SciBERT 20ep ({Path(seed_dir).name})", d["test_ids"], d["arms"]["enc-S"]["pred"])
    for f in sorted(glob.glob("experiments/h9-flan-t5/results/*_ft-S_fold*.json")):
        d = json.loads(Path(f).read_text())
        put(f"Flan-T5-{d['model']} fine-tuned", folds[d["fold"]], d["pred"])
    return arms, folds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args(argv)

    sdoh = add_evidence(load_sdoh(Path(args.sdoh_dir), "train")).reset_index(drop=True)
    y = sdoh["label_native"].to_numpy()
    arms, folds = collect(len(y))
    for size in ("base", "large"):
        path = Path(f"experiments/h9-flan-t5/results/{size}_zeroshot.json")
        if path.exists():
            zs = np.asarray(json.loads(path.read_text())["pred"], dtype=object)
            arms[f"Flan-T5-{size} zero-shot"] = zs
            arms[f"Flan-T5-{size} zero-shot + guard"] = guard(zs, sdoh["evid0"].tolist())
    ref = arms["TF-IDF (S)"]
    rng = np.random.default_rng(20261030)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(args.n_bootstrap)]
    rows = []
    for name, pred in arms.items():
        complete = all(p is not None for p in pred)
        row = {"arm": name, "complete": complete}
        if complete:
            m = metrics_payload(y, pred)
            row.update({"pooled_macro_f1": round(m["macro_f1"], 4), "per_class_f1": {k: round(v["f1"], 3) for k, v in m["per_class"].items()},
                        "fold_macro_f1": [round(metrics_payload(y[np.asarray(te)], pred[np.asarray(te)])["macro_f1"], 3) for _, te in sorted(folds.items())]})
            if name != "TF-IDF (S)":
                d = np.array([macro_f1(confusion(y[i], pred[i])) - macro_f1(confusion(y[i], ref[i])) for i in boots])
                row["delta_vs_tfidf"] = {"mean": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], "p_le_0": round(float((d <= 0).mean()), 4)}
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("pooled_macro_f1") or 0))
    for r in rows:
        if r["complete"]:
            delta = r.get("delta_vs_tfidf")
            dtxt = f"Δ={delta['mean']:+.3f} CI[{delta['ci95'][0]:+.3f},{delta['ci95'][1]:+.3f}] p≤0={delta['p_le_0']:.3f}" if delta else "(reference)"
            print(f"{r['arm']:34s} pooled={r['pooled_macro_f1']:.3f} folds={r['fold_macro_f1']} NEVER={r['per_class_f1']['NEVER']:.2f} {dtxt}")
        else:
            print(f"{r['arm']:34s} (incomplete)")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
