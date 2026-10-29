#!/usr/bin/env python3
"""Build an offline, randomly initialised ModernBERT-base-shaped classifier for E4.

Use only when the Hugging Face Hub is unreachable. Throughput depends on the
architecture and sequence lengths, not on weight values, so this model gives a
valid *speed* number for arms C/D (BioClinical-ModernBERT-base / ModernBERT-base
share the base shape) and no accuracy information at all.

The tokenizer is a byte-level BPE trained on the Wright train split with the
ModernBERT vocabulary size (50,368). Its token counts approximate, but do not
equal, the real ModernBERT tokenizer; record that caveat with every result.

Also writes the dev-split crops (retrieval-v2, whitespace retrieval tokenizer)
as the benchmark input CSV.

    python scripts/build_shape_model.py --wright-dir /path/to/smokingbenchmark \
        --splits experiments/E0_E3_wright_retrieval-v2/labels_and_splits.csv \
        --out runs/modernbert-base-shape
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.extract_windows import WhitespaceOffsetTokenizer, extract  # noqa: E402
from src.label_policy import LABELS  # noqa: E402

MODERNBERT_VOCAB = 50368


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wright-dir", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out", default="runs/modernbert-base-shape")
    args = parser.parse_args(argv)

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import ModernBertConfig, ModernBertForSequenceClassification, PreTrainedTokenizerFast

    out = Path(args.out)
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    wright = Path(args.wright_dir)
    splits = pd.read_csv(args.splits)
    note = lambda row: (wright / row.condition / "notes" / f"{row.patient_id}.txt").read_text(encoding="utf-8")  # noqa: E731

    train_texts = [note(r) for r in splits[splits["split"] == "train"].itertuples()]
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    special = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    backend.train_from_iterator(
        train_texts,
        trainers.BpeTrainer(vocab_size=MODERNBERT_VOCAB, special_tokens=special, show_progress=False),
    )
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", backend.token_to_id("[CLS]")), ("[SEP]", backend.token_to_id("[SEP]"))],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(str(model_dir))

    config = ModernBertConfig(
        vocab_size=MODERNBERT_VOCAB,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    model = ModernBertForSequenceClassification(config)
    model.save_pretrained(str(model_dir))
    n_params = sum(p.numel() for p in model.parameters())

    retrieval_tokenizer = WhitespaceOffsetTokenizer()
    rows = []
    for record in splits[splits["split"] == "dev"].itertuples():
        for crop in extract(note(record), retrieval_tokenizer)["crops"]:
            rows.append({"note_id": record.patient_id, "crop_id": crop["crop_id"], "text": crop["text"]})
    crops = pd.DataFrame(rows)
    crops.to_csv(out / "dev_crops.csv", index=False)
    lengths = [len(ids) for ids in tokenizer(crops["text"].tolist())["input_ids"]]
    print(
        f"params={n_params:,} vocab={tokenizer.vocab_size} dev_crops={len(crops)} "
        f"bpe_tokens_p50={pd.Series(lengths).median():.0f} p95={pd.Series(lengths).quantile(0.95):.0f} max={max(lengths)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
