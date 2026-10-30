#!/usr/bin/env python3
"""H9-FLAN: Flan-T5 (instruction-tuned seq2seq LM) on evidence sentences, zero-shot and
fine-tuned, on the same SDOH-NLI folds as H8 (protocol: experiments/h9-flan-t5/protocol.md).

Zero-shot uses rank classification: the model scores each answer string and the most
likely one wins, so the output is always one of the four labels (no free-text parsing).

    python scripts/run_h9_flan_cv.py --model ../models/flan-t5-base --tag base \
        --wright-dir ../smokingbenchmark --sdoh-dir ../SDOH-NLI --out experiments/h9-flan-t5/results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_h7_ood import add_evidence, load_sdoh  # noqa: E402
from run_wright_smoke import assign_splits, load  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402

ANSWERS = {"CURRENT": "current smoker", "FORMER": "former smoker", "NEVER": "never smoked", "UNKNOWN": "unknown"}
# Flan's own multiple-choice format (fixed on 8 hand-written policy cases, not on any eval data).
PROMPT = (
    "{text}\n\nBased on the text above, what is the patient's own cigarette smoking status?\n"
    "OPTIONS:\n- current smoker\n- former smoker\n- never smoked\n- unknown"
)


def score_options(tok, model, texts: list[str], batch: int = 16, calibrate: bool = False) -> np.ndarray:
    """Rank classification: sum of target-token log-probs for each answer string.

    With ``calibrate`` the score of each answer given a content-free input ("N/A") is
    subtracted (contextual calibration, Zhao et al. 2021) to remove answer-prior bias.
    """
    import torch

    labels = list(ANSWERS)
    targets = tok([ANSWERS[label] for label in labels], return_tensors="pt", padding=True)
    preds = np.empty(len(texts), dtype=object)
    prior = None
    if calibrate:
        calibrate = False
        prior = _raw_scores(tok, model, ["N/A"], targets, labels)[0]
    with torch.no_grad():
        for start in range(0, len(texts), batch):
            chunk = texts[start : start + batch]
            enc = tok([PROMPT.format(text=t) for t in chunk], return_tensors="pt", padding=True, truncation=True, max_length=256)
            encoder_out = model.get_encoder()(**enc)
            scores = torch.zeros(len(chunk), len(labels))
            for j in range(len(labels)):
                tgt = targets.input_ids[j : j + 1].repeat(len(chunk), 1)
                mask = targets.attention_mask[j : j + 1].repeat(len(chunk), 1)
                tgt_labels = tgt.masked_fill(mask == 0, -100)
                out = model(encoder_outputs=encoder_out, attention_mask=enc.attention_mask, labels=tgt_labels)
                logp = torch.log_softmax(out.logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                scores[:, j] = (logp * mask).sum(-1)
            if prior is not None:
                scores = scores - prior
            for k, row in enumerate(scores.argmax(-1).tolist()):
                preds[start + k] = labels[row]
    return preds


def _raw_scores(tok, model, texts, targets, labels):
    import torch

    with torch.no_grad():
        enc = tok([PROMPT.format(text=t) for t in texts], return_tensors="pt", padding=True)
        encoder_out = model.get_encoder()(**enc)
        scores = torch.zeros(len(texts), len(labels))
        for j in range(len(labels)):
            tgt = targets.input_ids[j : j + 1].repeat(len(texts), 1)
            mask = targets.attention_mask[j : j + 1].repeat(len(texts), 1)
            out = model(encoder_outputs=encoder_out, attention_mask=enc.attention_mask, labels=tgt.masked_fill(mask == 0, -100))
            logp = torch.log_softmax(out.logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            scores[:, j] = (logp * mask).sum(-1)
    return scores


def finetune(model_dir: str, texts: list[str], labels: list[str], *, epochs: int, seed: int, lr: float, batch: int = 16, optimizer: str = "adamw"):
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    tok = T5Tokenizer.from_pretrained(model_dir, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.train()
    # Oversample minority classes to the size of the largest one (labels are strings here).
    df = pd.DataFrame({"t": texts, "y": labels})
    top = df["y"].value_counts().max()
    df = pd.concat([g.sample(top, replace=True, random_state=seed) for _, g in df.groupby("y")], ignore_index=True)
    order = np.random.default_rng(seed).permutation(len(df))
    if optimizer == "adafactor":
        from transformers.optimization import Adafactor

        opt = Adafactor(model.parameters(), lr=lr, relative_step=False, scale_parameter=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(df) / batch))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(df))
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            enc = tok([PROMPT.format(text=df.t.iloc[i]) for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256)
            tgt = tok([ANSWERS[df.y.iloc[i]] for i in idx], return_tensors="pt", padding=True).input_ids
            tgt = tgt.masked_fill(tgt == tok.pad_token_id, -100)
            loss = model(**enc, labels=tgt).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    return tok, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default="zeroshot,ft-S")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--wright-dev", action="store_true", help="also score zero-shot on Wright dev (native)")
    parser.add_argument("--optimizer", choices=("adamw", "adafactor"), default="adamw")
    args = parser.parse_args(argv)

    import torch
    from sklearn.model_selection import StratifiedKFold
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sdoh = add_evidence(load_sdoh(Path(args.sdoh_dir), "train")).reset_index(drop=True)
    y = sdoh["label_native"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(sdoh, y))

    def with_evidence(texts: pd.Series, fn) -> np.ndarray:
        pred = np.full(len(texts), "UNKNOWN", dtype=object)
        has = (texts.str.len() > 0).to_numpy()
        if has.any():
            pred[has] = fn(texts[has].tolist())
        return pred

    if "zeroshot" in args.arms:
        tok = T5Tokenizer.from_pretrained(args.model, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(args.model).eval()
        started = time.perf_counter()
        zs = with_evidence(sdoh["evid0"], lambda t: score_options(tok, model, t, calibrate=True))
        elapsed = time.perf_counter() - started
        per_fold = []
        for k, (_, te) in enumerate(folds):
            m = metrics_payload(y[te], zs[te])
            per_fold.append({"fold": k, "macro_f1": m["macro_f1"], "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()}})
        payload = {"arm": "zeroshot", "model": args.tag, "per_fold": per_fold, "pred": zs.tolist(), "seconds": round(elapsed, 1),
                   "sentences_per_s": round(int((sdoh["evid0"].str.len() > 0).sum()) / elapsed, 1)}
        if args.wright_dev:
            wright = load(Path(args.wright_dir))
            wright["split"] = assign_splits(wright)
            dev = add_evidence(wright[wright["split"] == "dev"])
            wz = with_evidence(dev["evid0"], lambda t: score_options(tok, model, t, calibrate=True))
            payload["wright_dev_native"] = metrics_payload(dev["label_native"].to_numpy(), wz)
        (out / f"{args.tag}_zeroshot.json").write_text(json.dumps(payload, indent=1, default=str))
        print(f"{args.tag} zeroshot CV-fold macroF1 mean={np.mean([f['macro_f1'] for f in per_fold]):.3f} {payload['sentences_per_s']} sent/s", flush=True)
        if args.wright_dev:
            print(f"{args.tag} zeroshot Wright-dev native macroF1={payload['wright_dev_native']['macro_f1']:.3f}", flush=True)

    if "ft-S" in args.arms:
        for k in [int(f) for f in args.folds.split(",")]:
            path = out / f"{args.tag}_ft-S_fold{k}.json"  # tag carries the optimizer for H14 runs
            if path.exists():
                continue
            tr, te = folds[k]
            a, b = sdoh.iloc[tr], sdoh.iloc[te]
            a = a[a["evid0"].str.len() > 0]
            started = time.perf_counter()
            tok, model = finetune(args.model, a["evid0"].tolist(), a["label_native"].tolist(), epochs=args.epochs, seed=args.seed, lr=args.lr, optimizer=args.optimizer)
            pred = with_evidence(b["evid0"], lambda t: score_options(tok, model, t))
            m = metrics_payload(b["label_native"].to_numpy(), pred)
            path.write_text(json.dumps({"arm": "ft-S", "model": args.tag, "fold": k, "macro_f1": m["macro_f1"],
                                        "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()}, "pred": pred.tolist(),
                                        "seconds": round(time.perf_counter() - started, 1)}, indent=1, default=str))
            print(f"{args.tag} ft-S fold {k} macroF1={m['macro_f1']:.3f} NEVER={m['per_class']['NEVER']['f1']:.3f} {time.perf_counter() - started:.0f}s", flush=True)
            del tok, model  # free this fold's model before the next fold trains (large: 2 x 3 GB otherwise)
            import gc

            gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
