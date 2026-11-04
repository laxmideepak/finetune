#!/usr/bin/env python3
"""H19 one-time evaluation (protocol: experiments/h19-iterative-wsp/protocol.md).

Evaluation premises = SDOH-NLI train fold 0 (H9 folds) + SDOH-NLI validation; never seen by the editor.
"""

from __future__ import annotations

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

import pandas as pd  # noqa: E402

from run_h7_ood import add_evidence, fit_predict, load_sdoh  # noqa: E402
from run_h10_guard import guard  # noqa: E402
from src.flan_status import finetune, load_tokenizer, score_options  # noqa: E402
from src.metrics import confusion, macro_f1, metrics_payload  # noqa: E402

LARGE = "/home/user/ext/models/flan-t5-large"
BASE = "/home/user/ext/models/flan-t5-base"
SDOH = Path("/home/user/ext/datasets/SDOH-NLI")
OUT = REPO_ROOT / "experiments/h19-iterative-wsp/results"


def main() -> int:
    import torch
    from sklearn.model_selection import StratifiedKFold
    from transformers import T5ForConditionalGeneration

    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    s = add_evidence(load_sdoh(SDOH, "train")).reset_index(drop=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(s, s["label_native"].to_numpy()))
    dev = s.iloc[np.sort(np.concatenate([folds[k][1] for k in (1, 2, 3, 4)]))]
    val = add_evidence(load_sdoh(SDOH, "validation"))
    ev = pd.concat([s.iloc[folds[0][1]].assign(part="train-fold0"), val.assign(part="validation")], ignore_index=True)
    y = ev["label_native"].to_numpy()
    has = (ev["evid0"].str.len() > 0).to_numpy()
    texts = ev.loc[has, "evid0"].tolist()

    def run(predict_fn) -> np.ndarray:
        p = np.full(len(ev), "UNKNOWN", dtype=object)
        p[has] = predict_fn(texts)
        return p

    preds = {}
    tok = load_tokenizer(LARGE)
    model = T5ForConditionalGeneration.from_pretrained(LARGE).eval()
    preds["Flan-T5-large zero-shot"] = run(lambda t: score_options(tok, model, t, calibrate=True))
    preds["zero-shot + rule guard"] = guard(preds["Flan-T5-large zero-shot"], ev["evid0"].tolist())
    edit_dir = REPO_ROOT / "experiments/h19-iterative-wsp/edit"
    sd = model.state_dict()
    for name in json.loads((edit_dir / "manifest.json").read_text()):
        sd[name].copy_(torch.from_numpy(np.load(edit_dir / f"{name}.npy")))
    preds["Flan-T5-large + iterative weight edit"] = run(lambda t: score_options(tok, model, t, calibrate=True))
    preds["weight edit + rule guard (exploratory)"] = guard(preds["Flan-T5-large + iterative weight edit"], ev["evid0"].tolist())
    del model
    d = dev[dev["evid0"].str.len() > 0]
    tok_b, model_b = finetune(BASE, d["evid0"].tolist(), d["label_native"].tolist(), epochs=3, seed=42, lr=3e-4)
    preds["Flan-T5-base fine-tuned (same dev data)"] = run(lambda t: score_options(tok_b, model_b, t))
    preds["TF-IDF (same dev data)"] = fit_predict(dev, ev)

    rng = np.random.default_rng(20261104)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(5000)]
    ref = preds["Flan-T5-large zero-shot"]
    res = {"n": int(len(y)), "label_counts": ev["label_native"].value_counts().to_dict(), "arms": {}}
    for name, p in preds.items():
        m = metrics_payload(y, p)
        row = {"macro_f1": round(m["macro_f1"], 4), "per_class_f1": {k: round(v["f1"], 3) for k, v in m["per_class"].items()},
               "by_part": {part: round(metrics_payload(y[(ev["part"] == part).to_numpy()], p[(ev["part"] == part).to_numpy()])["macro_f1"], 4) for part in ("train-fold0", "validation")}}
        if name != "Flan-T5-large zero-shot":
            dl = np.array([macro_f1(confusion(y[i], p[i])) - macro_f1(confusion(y[i], ref[i])) for i in boots])
            row["delta_vs_zero_shot"] = {"mean": round(float(macro_f1(confusion(y, p)) - macro_f1(confusion(y, ref))), 4), "ci95": [round(float(np.percentile(dl, 2.5)), 4), round(float(np.percentile(dl, 97.5)), 4)]}
        res["arms"][name] = row
        print(f"{name:42s} macroF1={row['macro_f1']:.3f} NEVER={row['per_class_f1']['NEVER']:.2f} by_part={row['by_part']} {row.get('delta_vs_zero_shot', '')}", flush=True)
    ft, ed = preds["Flan-T5-base fine-tuned (same dev data)"], preds["Flan-T5-large + iterative weight edit"]
    dl = np.array([macro_f1(confusion(y[i], ed[i])) - macro_f1(confusion(y[i], ft[i])) for i in boots])
    res["edit_minus_finetuned"] = {"mean": round(float(macro_f1(confusion(y, ed)) - macro_f1(confusion(y, ft))), 4), "ci95": [round(float(np.percentile(dl, 2.5)), 4), round(float(np.percentile(dl, 97.5)), 4)]}
    print("edit minus fine-tuned:", res["edit_minus_finetuned"])
    (OUT / "h19_eval.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame({"text": ev["evid0"], "part": ev["part"], "gold": y, **{k: v for k, v in preds.items()}}).to_json(OUT / "h19_eval_per_item.json", orient="records", indent=0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
