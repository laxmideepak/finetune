#!/usr/bin/env python3
"""H8-ENCODER: fine-tune a local pretrained encoder on evidence sentences, paired with
TF-IDF on identical SDOH-NLI folds (protocol: experiments/h8-pretrained-encoder-cpu/protocol.md).

    python scripts/run_h8_encoder_cv.py --model /path/to/scibert_scivocab_uncased \
        --wright-dir ../smokingbenchmark --sdoh-dir ../SDOH-NLI \
        --out experiments/h8-pretrained-encoder-cpu/results
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_h7_ood import add_evidence, fit_predict, load_sdoh  # noqa: E402
from run_wright_smoke import assign_splits, load  # noqa: E402
from src.label_policy import LABELS  # noqa: E402
from src.metrics import metrics_payload  # noqa: E402

LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def train_encoder(model_dir: str, texts: list[str], labels: list[str], *, epochs: int, max_len: int, seed: int, lr: float = 3e-5, batch: int = 16):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID
    )
    y = np.array([LABEL2ID[label] for label in labels])
    counts = np.bincount(y, minlength=len(LABELS)).astype(float)
    weights = torch.tensor(len(y) / (len(LABELS) * np.maximum(counts, 1.0)), dtype=torch.float)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    enc = tok(texts, truncation=True, max_length=max_len)
    truncated = sum(len(ids) >= max_len for ids in enc["input_ids"])
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=lr)
    steps = epochs * int(np.ceil(len(y) / batch))
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            features = tok.pad({"input_ids": [enc["input_ids"][i] for i in idx], "attention_mask": [enc["attention_mask"][i] for i in idx]}, return_tensors="pt")
            logits = model(**features).logits
            loss = loss_fn(logits, torch.tensor(y[idx]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    model.eval()
    return tok, model, truncated


def predict_encoder(tok, model, texts: list[str], *, max_len: int, batch: int = 32) -> tuple[np.ndarray, float]:
    import torch

    preds = np.empty(len(texts), dtype=object)
    todo = [i for i, t in enumerate(texts) if t.strip()]
    for i, t in enumerate(texts):
        if not t.strip():
            preds[i] = "UNKNOWN"
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(todo), batch):
            idx = todo[start : start + batch]
            features = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            out = model(**features).logits.argmax(-1).tolist()
            for i, k in zip(idx, out):
                preds[i] = LABELS[k]
    elapsed = time.perf_counter() - started
    return preds, (len(todo) / elapsed if elapsed > 0 else float("nan"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--sdoh-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--arms", default="tfidf-S,enc-S,tfidf-WS,enc-WS")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--epochs-s", type=int, default=4, help="epochs for enc-S (H8: 4; H8b: 20)")
    parser.add_argument("--lr", type=float, default=3e-5, help="H8: 3e-5; H8b: 2e-5")
    args = parser.parse_args(argv)

    import torch
    from sklearn.model_selection import StratifiedKFold

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wright = load(Path(args.wright_dir))
    wright["split"] = assign_splits(wright)
    w_train = add_evidence(wright[wright["split"] == "train"])
    sdoh = add_evidence(load_sdoh(Path(args.sdoh_dir), "train")).reset_index(drop=True)
    y = sdoh["label_native"].to_numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(sdoh, y))
    arms = args.arms.split(",")
    wanted = [int(f) for f in args.folds.split(",")]

    for fold in wanted:
        path = out / f"fold{fold}.json"
        record = json.loads(path.read_text()) if path.exists() else {"fold": fold, "arms": {}}
        tr, te = folds[fold]
        a, b = sdoh.iloc[tr], sdoh.iloc[te]
        record["test_ids"] = [int(i) for i in te]
        record["y_true"] = b["label_native"].tolist()
        ws = pd.concat([w_train[["evid0", "label_native"]], a[["evid0", "label_native"]]], ignore_index=True)
        for arm in arms:
            if arm in record["arms"]:
                continue
            started = time.perf_counter()
            info: dict = {}
            if arm == "tfidf-S":
                pred = fit_predict(a, b)
            elif arm == "tfidf-WS":
                pred = fit_predict(ws, b)
            else:
                train_df = a if arm == "enc-S" else ws
                train_df = train_df[train_df["evid0"].str.len() > 0]
                epochs = args.epochs_s if arm == "enc-S" else 3
                tok, model, truncated = train_encoder(args.model, train_df["evid0"].tolist(), train_df["label_native"].tolist(), epochs=epochs, max_len=args.max_len, seed=args.seed, lr=args.lr)
                pred, sps = predict_encoder(tok, model, b["evid0"].tolist(), max_len=args.max_len)
                info = {"epochs": epochs, "train_rows": int(len(train_df)), "train_truncated": int(truncated), "infer_sentences_per_s": round(sps, 1)}
                del model
            payload = metrics_payload(b["label_native"].to_numpy(), pred)
            info.update({"seconds": round(time.perf_counter() - started, 1), "macro_f1": payload["macro_f1"], "per_class_f1": {k: v["f1"] for k, v in payload["per_class"].items()}, "pred": [str(p) for p in pred]})
            record["arms"][arm] = info
            path.write_text(json.dumps(record, indent=1))
            print(f"fold {fold} {arm:9s} macroF1={payload['macro_f1']:.3f} NEVER={info['per_class_f1']['NEVER']:.3f} {info['seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
