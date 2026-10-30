#!/usr/bin/env python3
"""H12-FINAL: confirm fine-tuned Flan-T5-base on locked SDOH-NLI test, SDOH validation,
the Wright end-to-end pipeline, and CPU cost (protocol: experiments/h12-final-confirmation/protocol.md).

    python scripts/run_h12_final.py --flan ../models/flan-t5-base --wright-dir ../smokingbenchmark \
        --sdoh-dir ../SDOH-NLI --out experiments/h12-final-confirmation/results --confirm-locked-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_h5_date_resolver import one_year_label_v2  # noqa: E402
from run_h7_ood import add_evidence, fit_predict, load_sdoh  # noqa: E402
from run_h9_flan_cv import finetune, score_options  # noqa: E402
from run_wright_smoke import assign_splits, load  # noqa: E402
from src.metrics import confusion, macro_f1, metrics_payload  # noqa: E402
from src.pipeline import SmokingStatusPipeline, build_tfidf_classifier  # noqa: E402


class FlanClassifier:
    """Adapter so a fine-tuned Flan-T5 plugs into SmokingStatusPipeline."""

    def __init__(self, tok, model):
        self.tok, self.model = tok, model

    def predict(self, texts):
        return list(score_options(self.tok, self.model, list(texts)))


def summary(y, pred, bearing=None) -> dict:
    m = metrics_payload(y, pred)
    out = {"n": int(len(y)), "macro_f1": round(m["macro_f1"], 4), "per_class_f1": {k: round(v["f1"], 3) for k, v in m["per_class"].items()},
           "current_recall": m["current_recall"], "current_precision": m["current_precision"]}
    if bearing is not None:
        out["macro_f1_smoking_bearing"] = round(metrics_payload(y[bearing], pred[bearing])["macro_f1"], 4)
    return out


def paired(y, a, b, n=5000, seed=20261030) -> dict:
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(macro_f1(confusion(y[i], a[i])) - macro_f1(confusion(y[i], b[i])))
    d = np.asarray(d)
    return {"delta": round(float(macro_f1(confusion(y, a)) - macro_f1(confusion(y, b))), 4), "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)], "p_le_0": round(float((d <= 0).mean()), 4)}


def with_evidence(model_predict, texts: pd.Series) -> np.ndarray:
    pred = np.full(len(texts), "UNKNOWN", dtype=object)
    has = (texts.str.len() > 0).to_numpy()
    if has.any():
        pred[has] = model_predict(texts[has].tolist())
    return pred


def run_wright(args, s_train: pd.DataFrame, out: Path) -> int:
    """Wright end-to-end pipeline stage (separate so it can run in a fresh process)."""
    wright = load(Path(args.wright_dir))
    wright["split"] = assign_splits(wright)
    wright["label_one_year_v2"] = wright.apply(one_year_label_v2, axis=1)
    w_train = add_evidence(wright[wright["split"] == "train"])
    both = pd.concat([w_train[["evid0", "label_native"]], s_train[["evid0", "label_native"]]], ignore_index=True)
    both = both[both["evid0"].str.len() > 0]
    tok_w, model_w = finetune(args.flan, both["evid0"].tolist(), both["label_native"].tolist(), epochs=3, seed=42, lr=3e-4)
    dev = wright[wright["split"] == "dev"]
    notes, dates, ages = dev["text"].tolist(), [date.fromisoformat(d) for d in dev["note_date"]], dev["age"].astype(int).tolist()
    flan_pipe = SmokingStatusPipeline(FlanClassifier(tok_w, model_w), name="flan-t5-base-ft[wright+sdoh]")
    tfidf_pipe = SmokingStatusPipeline(build_tfidf_classifier().fit(both["evid0"], both["label_native"]), name="tfidf[wright+sdoh]")
    wres = {}
    for name, pipe in (("Flan-T5-base FT pipeline", flan_pipe), ("TF-IDF pipeline", tfidf_pipe)):
        preds = pipe.predict(notes, dates, ages)
        labels = np.asarray([p.label for p in preds], dtype=object)
        native = np.asarray([p.native_status for p in preds], dtype=object)
        wres[name] = {"one_year_v2": summary(dev["label_one_year_v2"].to_numpy(), labels), "native_accuracy": round(float((native == dev["label_native"].to_numpy()).mean()), 4)}
    results = {"wright_dev_pipeline": wres}
    print("wright", json.dumps(wres, default=str), flush=True)

    (out / "h12_wright.json").write_text(json.dumps(results, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flan", required=True)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hourly-price", type=float, default=0.17)
    parser.add_argument("--skip-wright", action="store_true")
    parser.add_argument("--wright-only", action="store_true", help="run only the Wright pipeline stage (does not touch SDOH test)")
    args = parser.parse_args(argv)
    if not args.confirm_locked_test and not args.wright_only:
        parser.error("this run opens the locked SDOH-NLI test split; pass --confirm-locked-test")

    import torch

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    s_train = add_evidence(load_sdoh(Path(args.sdoh_dir), "train"))
    if args.wright_only:
        return run_wright(args, s_train, out)
    train_rows = s_train[s_train["evid0"].str.len() > 0]
    started = time.perf_counter()
    tok, model = finetune(args.flan, train_rows["evid0"].tolist(), train_rows["label_native"].tolist(), epochs=3, seed=42, lr=3e-4)
    results["sdoh_train_seconds"] = round(time.perf_counter() - started, 1)
    model.save_pretrained(str(out / "flan-t5-base-sdoh"))
    tok.save_pretrained(str(out / "flan-t5-base-sdoh"))

    for split in ("validation", "test"):
        ev = add_evidence(load_sdoh(Path(args.sdoh_dir), split)).reset_index(drop=True)
        y = ev["label_native"].to_numpy()
        bearing = (ev["label_native"].ne("UNKNOWN") | ev["reason"].ne("no-mention")).to_numpy()
        flan = with_evidence(lambda t: score_options(tok, model, t), ev["evid0"])
        tfidf = fit_predict(s_train, ev)
        results[f"sdoh_{split}"] = {"Flan-T5-base FT": summary(y, flan, bearing), "TF-IDF S": summary(y, tfidf, bearing), "paired_flan_minus_tfidf": paired(y, flan, tfidf),
                                    "label_counts": ev["label_native"].value_counts().to_dict()}
        print(split, json.dumps(results[f"sdoh_{split}"], default=str), flush=True)

    # Efficiency: warm CPU inference on SDOH test evidence sentences, repeated to ~200 sentences.
    sentences = [t for t in add_evidence(load_sdoh(Path(args.sdoh_dir), "test"))["evid0"] if t.strip()]
    sentences = (sentences * (200 // max(len(sentences), 1) + 1))[:200]
    score_options(tok, model, sentences[:16])
    t0 = time.perf_counter()
    score_options(tok, model, sentences)
    rate = len(sentences) / (time.perf_counter() - t0)
    results["efficiency"] = {"flan_sentences_per_s_4threads": round(rate, 2), "hourly_price_assumed": args.hourly_price,
                             "flan_usd_per_million_sentences": round(1e6 * args.hourly_price / (3600 * rate), 2), "params": int(sum(p.numel() for p in model.parameters()))}
    print("efficiency", results["efficiency"], flush=True)

    if not args.skip_wright:
        del model
        import gc

        gc.collect()
        run_wright(args, s_train, out)
    (out / "h12.json").write_text(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
