#!/usr/bin/env python3
"""Offline end-to-end smoke test: tiny model, one train step, save/reload, predict.

This never touches the network. It builds a tiny local BERT (and optionally a
tiny Qwen3 + LoRA adapter), runs one optimizer step through the real CLI, saves
and reloads the model, and checks that predictions have the expected shape.
It also verifies ``--help`` for every entry point.

    python scripts/smoke_train.py                # encoder path
    python scripts/smoke_train.py --generative   # encoder + Qwen3 LoRA path
    python scripts/smoke_train.py --keep         # keep artifacts under runs/

Exit code 0 means every requested check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "smoke_splits.csv"
RUNS_DIR = REPO_ROOT / "runs"

ENTRY_POINTS = (
    "src/train_encoder.py",
    "src/train_generative.py",
    "src/evaluate_generative.py",
    "src/benchmark_encoder.py",
    "src/extract_windows.py",
    "src/compare_models.py",
)

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
}


def _run(command: list[str], *, timeout: int = 900) -> subprocess.CompletedProcess:
    environment = {**os.environ, **OFFLINE_ENV}
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}"
        )
    return completed


def _check_help() -> list[str]:
    checked = []
    for entry in ENTRY_POINTS:
        completed = _run([sys.executable, str(REPO_ROOT / entry), "--help"], timeout=120)
        if "usage:" not in completed.stdout.lower():
            raise RuntimeError(f"{entry} --help did not print usage")
        checked.append(entry)
    return checked


def _vocab(texts: list[str], labels: list[str]) -> dict[str, int]:
    words = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}
    for text in texts:
        words.update(text.split())
    words.update(labels)
    return {word: index for index, word in enumerate(sorted(words))}


def _build_tiny_encoder(model_dir: Path, texts: list[str], labels: list[str]) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    vocab = _vocab(texts, labels)
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        eos_token="[SEP]",
        model_max_length=512,
    )
    tokenizer.save_pretrained(str(model_dir))
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: index for index, label in enumerate(labels)},
    )
    BertForSequenceClassification(config).save_pretrained(str(model_dir))


def _build_tiny_generative(model_dir: Path, texts: list[str], labels: list[str]) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from src.label_policy import POLICY_TEXT

    vocab = _vocab(texts + [POLICY_TEXT], labels)
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[SEP]",
        model_max_length=1024,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    tokenizer.save_pretrained(str(model_dir))
    config = Qwen3Config(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(config).save_pretrained(str(model_dir))


def _smoke_encoder(workdir: Path) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.label_policy import LABELS, POLICY_VERSION

    frame = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    tiny = workdir / "tiny-encoder"
    tiny.mkdir(parents=True, exist_ok=True)
    _build_tiny_encoder(tiny, frame["text"].tolist(), list(LABELS))

    out = workdir / "encoder-run"
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "train_encoder.py"),
            "--csv", str(FIXTURE),
            "--model", str(tiny),
            "--model-revision", "main",
            "--out", str(out),
            "--max-length", "64",
            "--batch", "4",
            "--grad-accum", "1",
            "--epochs", "1",
            "--max-steps", "1",
            "--lr", "1e-3",
            "--device", "cpu",
            "--seed", "42",
        ]
    )

    config = json.loads((out / "run_config.json").read_text())
    assert config["policy_version"] == POLICY_VERSION
    assert config["data_manifest"]["n_rows"] == len(frame)
    assert config["selection"]["test_used_for_selection"] is False
    assert (out / "model" / "config.json").exists()
    assert (out / "dev_metrics.json").exists()
    logits = np.load(out / "dev_logits.npz")
    assert logits["example_ids"].shape[0] == logits["labels"].shape[0]
    mapping = pd.read_csv(out / "dev_row_mapping.csv", dtype=str, keep_default_na=False)
    assert set(mapping["example_id"]) == set(logits["example_ids"])

    reloaded = AutoModelForSequenceClassification.from_pretrained(str(out / "model"))
    tokenizer = AutoTokenizer.from_pretrained(str(out / "model"))
    encoded = tokenizer(
        ["Patient currently smokes one pack per day"],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    with torch.inference_mode():
        predictions = reloaded(**encoded).logits
    assert tuple(predictions.shape) == (1, len(LABELS))
    return {
        "tiny_model": str(tiny),
        "run": str(out),
        "dev_macro_f1": json.loads((out / "dev_metrics.json").read_text())["macro avg"]["f1-score"],
        "prediction_shape": list(predictions.shape),
    }


def _smoke_generative(workdir: Path) -> dict:
    import pandas as pd

    from src.label_policy import LABELS, POLICY_VERSION

    frame = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    tiny = workdir / "tiny-generative"
    tiny.mkdir(parents=True, exist_ok=True)
    _build_tiny_generative(tiny, frame["text"].tolist(), list(LABELS))

    out = workdir / "generative-run"
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "train_generative.py"),
            "--csv", str(FIXTURE),
            "--out", str(out),
            "--base-model", str(tiny),
            "--base-revision", "main",
            "--max-length", "512",
            "--batch", "1",
            "--grad-accum", "1",
            "--epochs", "1",
            "--max-steps", "1",
            "--lr", "1e-3",
            "--device", "cpu",
            "--seed", "42",
        ],
        timeout=1800,
    )
    config = json.loads((out / "run_config.json").read_text())
    assert config["policy_version"] == POLICY_VERSION
    assert config["data_manifest"]["n_rows"] == len(frame)
    assert config["selection"]["selected_by_train_script"] is False

    _run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "evaluate_generative.py"),
            "--run", str(out),
            "--split", "dev",
            "--device", "cpu",
            "--batch", "2",
            "--max-new-tokens", "4",
        ],
        timeout=1800,
    )
    selection = json.loads((out / "checkpoint_selection.json").read_text())
    manifest = json.loads((out / "evaluation_manifest.json").read_text())
    assert selection["selection_split"] == "dev"
    assert manifest["selection_split"] == "dev"
    assert manifest["test_confirmed"] is False
    assert (out / "dev_predictions.csv").exists()
    assert "invalid_output_policy" in manifest

    # The locked test must refuse to run without explicit confirmation.
    blocked = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "src" / "evaluate_generative.py"),
            "--run", str(out),
            "--split", "test",
            "--device", "cpu",
        ],
        cwd=str(REPO_ROOT),
        env={**os.environ, **OFFLINE_ENV},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert blocked.returncode == 2, f"locked-test guard failed: {blocked.returncode}"

    return {
        "tiny_model": str(tiny),
        "run": str(out),
        "selected_checkpoint": selection["selected_checkpoint"],
        "macro_f1": manifest["metrics"]["macro_f1"],
        "invalid_rate": manifest["metrics"]["invalid_rate"],
        "locked_test_guard": "refused without --confirm-locked-test",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generative", action="store_true", help="Also smoke the Qwen3 LoRA path")
    parser.add_argument("--keep", action="store_true", help="Keep artifacts under runs/")
    parser.add_argument("--skip-help", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    if args.keep:
        workdir = RUNS_DIR / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}"
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        workdir = Path(tempfile.mkdtemp(prefix="finetune-smoke-"))
        cleanup = True

    summary: dict[str, object] = {"workdir": str(workdir), "offline": True}
    try:
        summary["help_checked"] = [] if args.skip_help else _check_help()
        summary["encoder"] = _smoke_encoder(workdir)
        if args.generative:
            summary["generative"] = _smoke_generative(workdir)
        summary["elapsed_s"] = round(time.time() - started, 1)
        summary["ok"] = True
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        summary["ok"] = False
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["elapsed_s"] = round(time.time() - started, 1)
        print(json.dumps(summary, indent=2))
        return 1
    finally:
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
