"""Train on adjudicated, already extracted examples. No raw EHR download.
CSV columns: text,label,patient_id,note_id,split,leakage_group
split: train/dev/test. leakage_group: connected patient/near-duplicate group.
Run without --test during development; reserve --test for final evaluation.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
)

LABELS = ["CURRENT", "FORMER", "NEVER", "UNKNOWN"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="thomas-sounack/BioClinical-ModernBERT-base")
    ap.add_argument("--revision", default="main", help="Use a pinned commit for final runs")
    ap.add_argument("--out", default="runs/bioclinical")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    required = {"text", "label", "patient_id", "note_id", "split", "leakage_group"}
    if not required.issubset(df.columns):
        raise ValueError(f"Required columns: {sorted(required)}")
    if df.empty or (df[list(required)].apply(lambda s: s.str.strip()).eq("")).any().any():
        raise ValueError("Required fields must be nonempty")
    if not set(df.label) <= set(LABELS) or set(df.split) != {"train", "dev", "test"}:
        raise ValueError("Invalid labels or missing train/dev/test split")
    # leakage_group must be computed BEFORE splitting, using patient identity
    # and meaningful copied-note/near-duplicate clusters, not generic phrases.
    for key in ["patient_id", "note_id", "leakage_group"]:
        if df.groupby(key).split.nunique().max() > 1:
            raise ValueError(f"Cross-split leakage in {key}")
    for split in ["train", "dev", "test"]:
        if set(df.loc[df.split == split, "label"]) != set(LABELS):
            raise ValueError(f"All four classes must be represented in {split}")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    label2id = {name: i for i, name in enumerate(LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, num_labels=4,
        id2label=dict(enumerate(LABELS)), label2id=label2id,
        attn_implementation="sdpa",
    )
    # Fail rather than silently cut off potentially decisive smoking evidence.
    def encode(batch):
        result = tok(batch["text"], truncation=False)
        if any(len(ids) > args.max_length for ids in result["input_ids"]):
            raise ValueError("Input exceeds max-length: re-extract or route to wider context")
        return result
    def make_dataset(split):
        part = df.loc[df.split == split, ["text", "label"]].copy()
        part["labels"] = part.label.map(label2id).astype(int)
        ds = Dataset.from_pandas(part[["text", "labels"]], preserve_index=False)
        return ds.map(encode, batched=True, remove_columns=["text"])
    train_ds, dev_ds = make_dataset("train"), make_dataset("dev")
    def metrics(prediction):
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        yhat = logits.argmax(axis=-1)
        return {"macro_f1": f1_score(prediction.label_ids, yhat,
                                    labels=list(range(4)), average="macro", zero_division=0)}
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training = TrainingArguments(
        output_dir=str(out), learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=2, weight_decay=0.01, warmup_ratio=0.1,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="none", push_to_hub=False, seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training, train_dataset=train_ds, eval_dataset=dev_ds,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(str(out / "model"))
    tok.save_pretrained(out / "model")
    metadata = vars(args) | {
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "label_order": LABELS, "torch_version": torch.__version__,
    }
    (out / "run_config.json").write_text(json.dumps(metadata, indent=2))
    # Dev probabilities support calibration; these files inherit data controls.
    for split in (["dev", "test"] if args.test else ["dev"]):
        pred = trainer.predict(dev_ds if split == "dev" else make_dataset(split))
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        yhat = logits.argmax(-1)
        report = classification_report(pred.label_ids, yhat, labels=list(range(4)),
                                       target_names=LABELS, zero_division=0, output_dict=True)
        report["confusion_matrix_rows_true_cols_pred"] = confusion_matrix(
            pred.label_ids, yhat, labels=list(range(4))).tolist()
        (out / f"{split}_metrics.json").write_text(json.dumps(report, indent=2))
        np.savez_compressed(out / f"{split}_logits.npz", logits=logits, labels=pred.label_ids)
        # This mapping is sensitive, even though note text is omitted.
        df.loc[df.split == split, ["patient_id", "note_id"]].to_csv(
            out / f"{split}_row_mapping.csv", index=False)
    print("Saved model and metrics. Test evaluated:", args.test)

if __name__ == "__main__":
    main()
