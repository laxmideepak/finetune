#!/usr/bin/env python3
"""H15: score the two final pipelines once on the locked Wright test split
(protocol: experiments/h15-wright-test/protocol.md).

    python scripts/run_h15_wright_test.py --flan ../models/flan-t5-base --wright-dir ../smokingbenchmark \
        --sdoh-dir ../SDOH-NLI --out experiments/h15-wright-test/results --confirm-locked-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from run_h7_ood import add_evidence, load_sdoh  # noqa: E402
from run_h9_flan_cv import finetune  # noqa: E402
from run_h12_final import FlanClassifier, summary  # noqa: E402
from run_wright_smoke import CONDITIONS, assign_splits, load  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402
from src.pipeline import SmokingStatusPipeline, build_tfidf_classifier  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flan", required=True)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.confirm_locked_test:
        parser.error("this run opens the locked Wright test split; pass --confirm-locked-test")

    import torch

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wright = load(Path(args.wright_dir))
    wright["split"] = assign_splits(wright)
    wright["label_one_year_v2"] = wright.apply(one_year_label_v2, axis=1)
    w_train = add_evidence(wright[wright["split"] == "train"])
    s_train = add_evidence(load_sdoh(Path(args.sdoh_dir), "train"))
    both = pd.concat([w_train[["evid0", "label_native"]], s_train[["evid0", "label_native"]]], ignore_index=True)
    both = both[both["evid0"].str.len() > 0]

    tok, model = finetune(args.flan, both["evid0"].tolist(), both["label_native"].tolist(), epochs=3, seed=42, lr=3e-4)
    model.save_pretrained(str(out / "flan-t5-base-wright-sdoh"))
    tok.save_pretrained(str(out / "flan-t5-base-wright-sdoh"))

    test = wright[wright["split"] == "test"].reset_index(drop=True)
    notes, dates, ages = test["text"].tolist(), [date.fromisoformat(d) for d in test["note_date"]], test["age"].astype(int).tolist()
    pipelines = {
        "Flan-T5-base pipeline": SmokingStatusPipeline(FlanClassifier(tok, model), name="flan-t5-base-ft[wright+sdoh]"),
        "TF-IDF pipeline": SmokingStatusPipeline(build_tfidf_classifier().fit(both["evid0"], both["label_native"]), name="tfidf[wright+sdoh]"),
    }
    results, rows = {}, {"patient_id": test["patient_id"], "condition": test["condition"], "gold_v1": test["label_one_year"], "gold_v2": test["label_one_year_v2"], "gold_native": test["label_native"]}
    recent = ((test["label_native"] == "FORMER") & (test["label_one_year_v2"] == "CURRENT")).to_numpy()
    for name, pipe in pipelines.items():
        preds = pipe.predict(notes, dates, ages)
        labels = np.asarray([p.label for p in preds], dtype=object)
        native = np.asarray([p.native_status for p in preds], dtype=object)
        rows[f"{name}|label"] = labels
        res = {
            "one_year_v2": summary(test["label_one_year_v2"].to_numpy(), labels),
            "one_year_v1": summary(test["label_one_year"].to_numpy(), labels),
            "native_accuracy": round(float((native == test["label_native"].to_numpy()).mean()), 4),
            "by_condition_v2": {c: round(metrics_payload(test.loc[test["condition"] == c, "label_one_year_v2"].to_numpy(), labels[(test["condition"] == c).to_numpy()])["macro_f1"], 4) for c in CONDITIONS},
            "recent_quitter_recall": f"{int((labels[recent] == 'CURRENT').sum())}/{int(recent.sum())}",
        }
        results[name] = res
        print(name, json.dumps(res, default=str), flush=True)
    (out / "h15.json").write_text(json.dumps(results, indent=1, default=str))
    pd.DataFrame(rows).to_json(out / "h15_per_note.json", orient="records", indent=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
