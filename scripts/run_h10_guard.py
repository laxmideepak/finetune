#!/usr/bin/env python3
"""H10-GUARD: zero-shot Flan-T5 + policy guard (protocol: experiments/h10-llm-policy-guard/protocol.md).

Reuses H9 zero-shot predictions on SDOH-NLI train (CV, exploratory) and runs the zero-shot
models once on SDOH-NLI validation (confirmatory) and Wright dev.

    python scripts/run_h10_guard.py --wright-dir ../smokingbenchmark --sdoh-dir ../SDOH-NLI \
        --models-dir ../models --out experiments/h10-llm-policy-guard/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_h7_ood import add_evidence, fit_predict, load_sdoh  # noqa: E402
from run_wright_smoke import assign_splits, load  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402
from src.status_rules import NEVER, rule_status  # noqa: E402


def guard(preds, evidence) -> np.ndarray:
    out = np.asarray(preds, dtype=object).copy()
    for i, (p, ev) in enumerate(zip(out, evidence)):
        if p == "NEVER" and not NEVER.search(ev or ""):
            out[i] = "UNKNOWN"
    return out


def zero_shot(model_dir: str, texts) -> np.ndarray:
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    from run_h9_flan_cv import score_options

    tok = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).eval()
    pred = np.full(len(texts), "UNKNOWN", dtype=object)
    idx = [i for i, t in enumerate(texts) if t.strip()]
    if idx:
        pred[idx] = score_options(tok, model, [texts[i] for i in idx], calibrate=True)
    return pred


def summarize(y, pred, bearing) -> dict:
    m = metrics_payload(y, pred)
    return {
        "macro_f1": m["macro_f1"],
        "macro_f1_smoking_bearing": metrics_payload(y[bearing], pred[bearing])["macro_f1"],
        "per_class_f1": {k: round(v["f1"], 3) for k, v in m["per_class"].items()},
        "current_recall": m["current_recall"],
        "current_precision": m["current_precision"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--h9-results", default="experiments/h9-flan-t5/results")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    import torch
    from sklearn.model_selection import StratifiedKFold

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    # Confirmatory: SDOH-NLI validation.
    s_train = add_evidence(load_sdoh(Path(args.sdoh_dir), "train")).reset_index(drop=True)
    s_val = add_evidence(load_sdoh(Path(args.sdoh_dir), "validation")).reset_index(drop=True)
    yv = s_val["label_native"].to_numpy()
    bv = (s_val["label_native"].ne("UNKNOWN") | s_val["reason"].ne("no-mention")).to_numpy()
    ev = s_val["evid0"].tolist()
    val = {}
    for size in ("base", "large"):
        zs = zero_shot(str(Path(args.models_dir) / f"flan-t5-{size}"), ev)
        val[f"ZS-{size}"] = summarize(yv, zs, bv)
        val[f"ZS-{size}+guard"] = summarize(yv, guard(zs, ev), bv)
    val["TF-IDF S"] = summarize(yv, fit_predict(s_train, s_val), bv)
    val["Rules"] = summarize(yv, np.asarray([rule_status(e) for e in ev], dtype=object), bv)
    results["sdoh_validation"] = val

    # Exploratory: train CV folds (zero-shot predictions reused from H9).
    y = s_train["label_native"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(s_train, y))
    cv = {}
    for size in ("base", "large"):
        zs = np.asarray(json.loads((Path(args.h9_results) / f"{size}_zeroshot.json").read_text())["pred"], dtype=object)
        g = guard(zs, s_train["evid0"].tolist())
        cv[f"ZS-{size}"] = float(np.mean([metrics_payload(y[te], zs[te])["macro_f1"] for _, te in folds]))
        cv[f"ZS-{size}+guard"] = float(np.mean([metrics_payload(y[te], g[te])["macro_f1"] for _, te in folds]))
        cv[f"ZS-{size}+guard_per_class"] = {k: round(v["f1"], 3) for k, v in metrics_payload(y, g)["per_class"].items()}
    cv["TF-IDF S"] = float(np.mean([metrics_payload(y[te], fit_predict(s_train.iloc[tr], s_train.iloc[te]))["macro_f1"] for tr, te in folds]))
    results["sdoh_train_cv_exploratory"] = cv

    # Exploratory: Wright dev (native) guard effect.
    wright = load(Path(args.wright_dir))
    wright["split"] = assign_splits(wright)
    dev = add_evidence(wright[wright["split"] == "dev"]).reset_index(drop=True)
    zs = zero_shot(str(Path(args.models_dir) / "flan-t5-large"), dev["evid0"].tolist())
    g = guard(zs, dev["evid0"].tolist())
    yd = dev["label_native"].to_numpy()
    results["wright_dev_native"] = {
        "ZS-large": metrics_payload(yd, zs)["macro_f1"],
        "ZS-large+guard": metrics_payload(yd, g)["macro_f1"],
        "changed": int((zs != g).sum()),
        "changed_to_wrong": int(((zs != g) & (g != yd)).sum()),
    }
    (out / "h10.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
