"""Train the encoder arm on already-extracted, adjudicated examples.

The CSV must satisfy ``src/data_contract.py`` (stable example_id, corpus_id,
example_mode, label_scope, policy version, source, retrieval version, and
patient/note/leakage identifiers). No raw EHR download happens here.

Checkpoint selection and early stopping use **dev only**. ``--test`` is the
single final locked-test evaluation and additionally requires
``--confirm-final-test``; never use it while comparing models or tuning.

Example:
    python src/train_encoder.py --csv data/examples.csv \
        --model thomas-sounack/BioClinical-ModernBERT-base --out runs/c1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from src.label_policy import LABELS, POLICY_TEXT, POLICY_VERSION
    from src.data_contract import read_and_validate_csv
    from src.provenance import model_provenance, runtime_environment
except ImportError:  # running as a script: python src/train_encoder.py
    from label_policy import LABELS, POLICY_TEXT, POLICY_VERSION
    from data_contract import read_and_validate_csv
    from provenance import model_provenance, runtime_environment


def _policy_fingerprint() -> str:
    return hashlib.sha256(POLICY_TEXT.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--model", default="thomas-sounack/BioClinical-ModernBERT-base")
    parser.add_argument("--model-revision", default="main", help="Pin a commit for final runs")
    parser.add_argument("--out", default="runs/bioclinical")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=-1, help="Smoke runs: stop after N optimizer steps")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--test", action="store_true", help="Final locked-test evaluation only")
    parser.add_argument("--confirm-final-test", action="store_true")
    args = parser.parse_args(argv)

    if args.test and not args.confirm_final_test:
        parser.error("--test is reserved for the final locked evaluation; add --confirm-final-test after model selection")

    import numpy as np
    import pandas as pd
    import torch
    from datasets import Dataset
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
    )

    if args.device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, data_manifest = read_and_validate_csv(args.csv)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision, use_fast=True)

    # Tokenizer-specific overlength audit: counts, never raw text.
    lengths = [len(tok(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in df["text"]]
    overlength = [length for length in lengths if length > args.max_length]
    if overlength:
        raise ValueError(
            f"{len(overlength)} rows exceed max_length={args.max_length} "
            f"(max={max(overlength)}); route overflow to wider context instead of truncating"
        )

    label2id = {name: index for index, name in enumerate(LABELS)}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.model_revision,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=label2id,
        attn_implementation="sdpa",
    )

    def encode(batch):
        result = tok(batch["text"], truncation=False)
        if any(len(ids) > args.max_length for ids in result["input_ids"]):
            raise ValueError("Input exceeds max-length: re-extract or route to wider context")
        return result

    def make_dataset(split: str) -> Dataset:
        part = df.loc[df["split"] == split, ["text", "label"]].copy()
        part["labels"] = part["label"].map(label2id).astype(int)
        dataset = Dataset.from_pandas(part[["text", "labels"]], preserve_index=False)
        return dataset.map(encode, batched=True, remove_columns=["text"])

    train_ds, dev_ds = make_dataset("train"), make_dataset("dev")

    def metrics(prediction):
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(axis=-1)
        return {
            "macro_f1": f1_score(
                prediction.label_ids, predictions, labels=list(range(len(LABELS))),
                average="macro", zero_division=0,
            )
        }

    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    use_bf16 = cuda and torch.cuda.is_bf16_supported()
    use_cpu = args.device == "cpu" or (args.device == "mps" and not mps) or (args.device == "cuda" and not cuda)
    if args.device == "mps" and not mps:
        raise SystemExit("MPS requested but unavailable")
    if args.device == "cuda" and not cuda:
        raise SystemExit("CUDA requested but unavailable")

    smoke_steps = args.max_steps > 0
    training = TrainingArguments(
        output_dir=str(out),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="steps" if smoke_steps else "epoch",
        eval_steps=max(1, args.max_steps) if smoke_steps else None,
        save_strategy="steps" if smoke_steps else "epoch",
        save_steps=max(1, args.max_steps) if smoke_steps else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=cuda and not use_bf16,
        use_cpu=use_cpu,
        report_to="none",
        push_to_hub=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(str(out / "model"))
    tok.save_pretrained(out / "model")

    run_config = {
        "arm": "encoder",
        "args": vars(args),
        "policy_version": POLICY_VERSION,
        "policy_text_sha256": _policy_fingerprint(),
        "label_order": list(LABELS),
        "data_manifest": data_manifest,
        "data_file": str(Path(args.csv).resolve()),
        "token_lengths": {
            "max": max(lengths),
            "n_over_max_length": len(overlength),
            "max_length": args.max_length,
            "tokenizer": args.model,
            "note": "counts use this arm's tokenizer on the same retained text",
        },
        "selection": {
            "checkpoint_metric": "dev macro_f1",
            "early_stopping": "dev macro_f1, patience=2",
            "test_used_for_selection": False,
        },
        "model": model_provenance(model, tok, args.model, args.model_revision),
        "environment": runtime_environment(),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))

    splits = ["dev", "test"] if args.test else ["dev"]
    for split in splits:
        dataset = dev_ds if split == "dev" else make_dataset(split)
        prediction = trainer.predict(dataset)
        logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        predicted = logits.argmax(-1)
        report = classification_report(
            prediction.label_ids, predicted, labels=list(range(len(LABELS))),
            target_names=list(LABELS), zero_division=0, output_dict=True,
        )
        report["confusion_matrix_rows_true_cols_pred"] = confusion_matrix(
            prediction.label_ids, predicted, labels=list(range(len(LABELS)))
        ).tolist()
        (out / f"{split}_metrics.json").write_text(json.dumps(report, indent=2))
        rows = df.loc[df["split"] == split].reset_index(drop=True)
        np.savez_compressed(
            out / f"{split}_logits.npz",
            logits=logits,
            labels=prediction.label_ids,
            example_ids=np.asarray(rows["example_id"], dtype=str),
        )
        # Mapping is sensitive even without note text: keep it out of git and
        # include every identifier needed to join a prediction back to a crop.
        mapping_columns = [
            column for column in (
                "example_id", "corpus_id", "patient_id", "note_id", "crop_id",
                "example_mode", "label_scope", "label", "split",
            ) if column in rows.columns
        ]
        rows[mapping_columns].to_csv(out / f"{split}_row_mapping.csv", index=False)

    print(json.dumps({
        "out": str(out),
        "policy_version": POLICY_VERSION,
        "data_fingerprint": data_manifest["data_fingerprint"],
        "test_evaluated": args.test,
        "metrics": {
            split: json.loads((out / f"{split}_metrics.json").read_text())["macro avg"]["f1-score"]
            for split in splits
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
