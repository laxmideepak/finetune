"""Optional Qwen3-0.6B LoRA arm. Uses the same validated splits as the encoder.

This script trains only. Classification metrics, invalid-output counting, and
checkpoint selection live in ``src/evaluate_generative.py`` so that the
generative arm gets the same dev-selected evaluation machinery as the encoder.

Environment: install ``requirements-generative.txt`` in a separate environment
from the encoder arm; TRL/transformers version drift is the known risk R5.

Example:
    python src/train_generative.py --csv data/examples.csv --out runs/qwen-lora
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    from src.label_policy import LABELS, POLICY_TEXT, POLICY_VERSION, validate_label
    from src.data_contract import read_and_validate_csv
    from src.provenance import model_provenance, runtime_environment
except ImportError:  # running as a script: python src/train_generative.py
    from label_policy import LABELS, POLICY_TEXT, POLICY_VERSION, validate_label
    from data_contract import read_and_validate_csv
    from provenance import model_provenance, runtime_environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="runs/qwen-lora")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--base-revision", default="main", help="Pin a commit for final runs")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="Smoke runs: stop after N optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args(argv)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    df, data_manifest = read_and_validate_csv(args.csv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, revision=args.base_revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def build_prompt(text: str) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": POLICY_TEXT},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    lengths: list[int] = []
    for row in df.itertuples():
        validate_label(row.label)
        prompt = build_prompt(row.text)
        completion = row.label + tokenizer.eos_token
        lengths.append(len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]))
    overlength = [length for length in lengths if length > args.max_length]
    if overlength:
        raise ValueError(
            f"{len(overlength)} rows exceed max_length={args.max_length} "
            f"(max={max(overlength)}); route overflow to wider context instead of truncating"
        )

    def dataset(split: str) -> Dataset:
        rows = []
        for row in df.loc[df["split"] == split].itertuples():
            prompt = build_prompt(row.text)
            rows.append({"prompt": prompt, "completion": row.label + tokenizer.eos_token})
        if not rows:
            raise ValueError(f"Empty split {split!r}")
        return Dataset.from_list(rows)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, revision=args.base_revision, torch_dtype=dtype
    )
    model.config.use_cache = False

    smoke_steps = args.max_steps > 0
    training = SFTConfig(
        output_dir=str(out),
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        use_cpu=args.device == "cpu",
        eval_strategy="steps" if smoke_steps else "epoch",
        eval_steps=max(1, args.max_steps) if smoke_steps else None,
        save_strategy="steps" if smoke_steps else "epoch",
        save_steps=max(1, args.max_steps) if smoke_steps else None,
        save_total_limit=10,
        load_best_model_at_end=False,
        report_to="none",
        push_to_hub=False,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset("train"),
        eval_dataset=dataset("dev"),
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
        args=training,
    )
    trainer.train()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    chat_template = tokenizer.chat_template
    run_config = {
        "arm": "generative",
        "args": vars(args),
        "policy_version": POLICY_VERSION,
        "policy_text_sha256": hashlib.sha256(POLICY_TEXT.encode("utf-8")).hexdigest(),
        "label_order": list(LABELS),
        "data_manifest": data_manifest,
        "data_file": str(Path(args.csv).resolve()),
        "prompt_template": {
            "system": POLICY_TEXT,
            "user": "{text}",
            "add_generation_prompt": True,
            "enable_thinking": False,
            "chat_template_sha256": hashlib.sha256((chat_template or "").encode("utf-8")).hexdigest(),
            "chat_template": chat_template,
        },
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 8,
            "note": "evaluate_generative.py reads these values; exact-label match only",
        },
        "token_lengths": {
            "max": max(lengths),
            "n_over_max_length": len(overlength),
            "max_length": args.max_length,
            "tokenizer": args.base_model,
        },
        "selection": {
            "checkpoint_metric": "dev macro-F1 from src/evaluate_generative.py",
            "selected_by_train_script": False,
            "note": "all epoch checkpoints retained for dev-side selection; locked test is never used for selection",
        },
        "model": model_provenance(model, tokenizer, args.base_model, args.base_revision),
        "environment": runtime_environment(),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(json.dumps({
        "out": str(out),
        "policy_version": POLICY_VERSION,
        "data_fingerprint": data_manifest["data_fingerprint"],
        "n_checkpoints": len(list(out.glob("checkpoint-*"))),
        "note": "run src/evaluate_generative.py to select a checkpoint on dev",
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
