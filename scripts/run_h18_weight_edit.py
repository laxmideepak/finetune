#!/usr/bin/env python3
"""H18: gradient-free rank-1 weight edit of Flan-T5-large to separate denial-only from never
(protocol: experiments/h18-weight-space-edit/protocol.md).

    python scripts/run_h18_weight_edit.py --model ../models/flan-t5-large --sdoh-dir ../SDOH-NLI \
        --out experiments/h18-weight-space-edit/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_h7_ood import add_evidence, load_sdoh  # noqa: E402
from src.flan_status import PROMPT, load_tokenizer, score_options  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402

ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0)


def activations(tok, model, texts, layer: int):
    """Token-mean MLP hidden activation (input to wo) and block output for each text. Forward only."""
    import torch

    block = model.encoder.block[layer]
    mlp = block.layer[1].DenseReluDense
    keys, vals = [], []
    store = {}
    h1 = mlp.wo.register_forward_hook(lambda m, inp, out: store.__setitem__("k", inp[0].detach()))
    h2 = block.register_forward_hook(lambda m, inp, out: store.__setitem__("v", out[0].detach()))
    try:
        with torch.no_grad():
            for t in texts:
                enc = tok(PROMPT.format(text=t), return_tensors="pt", truncation=True, max_length=256)
                model.encoder(**enc)
                keys.append(store["k"][0].mean(0).double().numpy())
                vals.append(store["v"][0].mean(0).double().numpy())
    finally:
        h1.remove()
        h2.remove()
    return np.stack(keys), np.stack(vals)


def macro(y, p):
    return metrics_payload(np.asarray(y, dtype=object), np.asarray(p, dtype=object))["macro_f1"]


def predict(tok, model, texts):
    out = np.full(len(texts), "UNKNOWN", dtype=object)
    idx = [i for i, t in enumerate(texts) if t.strip()]
    if idx:
        out[idx] = score_options(tok, model, [texts[i] for i in idx], calibrate=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args(argv)

    import torch
    from sklearn.model_selection import StratifiedKFold
    from transformers import T5ForConditionalGeneration

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = add_evidence(load_sdoh(Path(args.sdoh_dir), "train")).reset_index(drop=True)
    y = s["label_native"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(s, y))
    tok = load_tokenizer(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).eval()
    wo = model.encoder.block[args.layer].layer[1].DenseReluDense.wo
    w0 = wo.weight.data.clone()
    total = sum(p.numel() for p in model.parameters())

    for k in [int(f) for f in args.folds.split(",")]:
        path = out / f"fold{k}.json"
        if path.exists():
            continue
        started = time.perf_counter()
        tr, te = folds[k]
        a, b = s.iloc[tr], s.iloc[te]
        a_ev = a[a["evid0"].str.len() > 0]
        d_texts = a_ev.loc[(a_ev["label_native"] == "UNKNOWN") & (a_ev["reason"] == "denial-only"), "evid0"].tolist()
        n_texts = a_ev.loc[a_ev["label_native"] == "NEVER", "evid0"].tolist()
        kd, vd = activations(tok, model, d_texts, args.layer)
        kn, vn = activations(tok, model, n_texts, args.layer)
        key = kd.mean(0) - kn.mean(0)
        val = vd.mean(0) - vn.mean(0)
        delta = torch.tensor(np.outer(val, key) / float(key @ key), dtype=wo.weight.dtype)
        train_scores = {}
        for alpha in ALPHAS:
            wo.weight.data = w0 + alpha * delta
            train_scores[alpha] = macro(a["label_native"].to_numpy(), predict(tok, model, a["evid0"].tolist()))
        best = max(ALPHAS, key=lambda x: (train_scores[x], -x))
        wo.weight.data = w0 + best * delta
        pred = predict(tok, model, b["evid0"].tolist())
        wo.weight.data = w0.clone()
        m = metrics_payload(b["label_native"].to_numpy(), pred)
        rec = {"fold": k, "alpha": best, "train_macro_f1_by_alpha": train_scores, "n_denial": len(d_texts), "n_never": len(n_texts),
               "macro_f1": m["macro_f1"], "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()}, "pred": pred.tolist(),
               "edited_matrix": f"encoder.block.{args.layer}.layer.1.DenseReluDense.wo", "matrix_share_of_weights": wo.weight.numel() / total,
               "edit_rank": 1, "seconds": round(time.perf_counter() - started, 1)}
        path.write_text(json.dumps(rec, indent=1, default=str))
        print(f"fold {k}: alpha={best} train={ {a_: round(v_, 3) for a_, v_ in train_scores.items()} } test macroF1={m['macro_f1']:.3f} NEVER={m['per_class']['NEVER']['f1']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
