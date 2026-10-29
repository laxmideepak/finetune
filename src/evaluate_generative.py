"""Evaluate and select Qwen3 LoRA checkpoints with the same contract as the encoder.

Selection is always on **dev** macro-F1 (invalid outputs count as errors). The
locked test is evaluated once, after selection, and only with
``--confirm-locked-test``. Invalid generations are kept as ``__INVALID__`` and
reported separately; they are never converted to UNKNOWN.

The adapter is loaded with the exact base-model revision and chat template
recorded by ``train_generative.py``.

Example:
    python src/evaluate_generative.py --run runs/qwen-lora --split dev
    python src/evaluate_generative.py --run runs/qwen-lora --split test \
        --confirm-locked-test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from src.label_policy import LABELS, POLICY_VERSION
    from src.metrics import INVALID, metrics_payload
    from src.provenance import model_provenance, runtime_environment
except ImportError:  # running as a script: python src/evaluate_generative.py
    from label_policy import LABELS, POLICY_VERSION
    from metrics import INVALID, metrics_payload
    from provenance import model_provenance, runtime_environment

_REQUIRED_RUN_CONFIG = ("args", "policy_version", "model", "prompt_template", "decoding")


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS requested but unavailable")
    return requested


def _dtype(device: str):
    import torch

    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _free(device: str) -> None:
    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _load_run(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        raise SystemExit(f"Missing {path}; train_generative.py must write the run manifest")
    config = json.loads(path.read_text())
    missing = [key for key in _REQUIRED_RUN_CONFIG if key not in config]
    if missing:
        raise SystemExit(f"run_config.json is missing {missing}")
    if config["policy_version"] != POLICY_VERSION:
        raise SystemExit(
            f"Policy version mismatch: run={config['policy_version']!r}, current={POLICY_VERSION!r}"
        )
    return config


def _base_revision(config: dict[str, Any]) -> str | None:
    return config["model"].get("resolved_model_commit") or config["args"].get("base_revision")


def _load_tokenizer(run_dir: Path, config: dict[str, Any]):
    from transformers import AutoTokenizer

    base_id = config["args"]["base_model"]
    tokenizer_path = run_dir if (run_dir / "tokenizer_config.json").exists() else base_id
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), revision=_base_revision(config))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    template = config["prompt_template"].get("chat_template")
    if template:
        tokenizer.chat_template = template
    return tokenizer


def _prompt(tokenizer, text: str, config: dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": config["prompt_template"]["system"]},
        {"role": "user", "content": text},
    ]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": bool(config["prompt_template"].get("add_generation_prompt", True)),
    }
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=bool(config["prompt_template"].get("enable_thinking", False)), **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _load_model(run_dir: Path, checkpoint: Path | None, config: dict[str, Any], device: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_id = config["args"]["base_model"]
    adapter = checkpoint or run_dir
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json in {adapter}")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, revision=_base_revision(config), torch_dtype=_dtype(device)
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.to(device).eval()
    model.config.use_cache = True
    return model


def _predict(model, tokenizer, rows, config: dict[str, Any], device: str, batch: int, max_new_tokens: int):
    import torch

    prompts = [_prompt(tokenizer, text, config) for text in rows["text"].tolist()]
    max_input = int(config["args"].get("max_length", 1024)) - max_new_tokens
    too_long = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
    if any(length > max_input for length in too_long):
        raise SystemExit(
            f"{sum(1 for length in too_long if length > max_input)} prompts exceed max_input={max_input}; "
            "route overflow instead of truncating"
        )

    outputs: list[dict[str, str]] = []
    for start in range(0, len(prompts), batch):
        chunk = prompts[start:start + batch]
        encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=False).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True
        )
        for raw in decoded:
            normalized = raw.strip().upper()
            outputs.append({
                "raw_output": raw.strip(),
                "y_pred": normalized if normalized in LABELS else INVALID,
            })
    return outputs


def evaluate_checkpoint(
    run_dir: Path,
    checkpoint: Path | None,
    rows,
    split: str,
    config: dict[str, Any],
    device: str,
    batch: int,
    max_new_tokens: int,
) -> tuple[dict[str, Any], Any]:
    import pandas as pd

    model = _load_model(run_dir, checkpoint, config, device)
    try:
        predictions = _predict(model, _load_tokenizer(run_dir, config), rows, config, device, batch, max_new_tokens)
    finally:
        del model
        _free(device)
    result = rows[["example_id", "corpus_id", "patient_id", "note_id", "crop_id", "example_mode", "split", "label"]].copy()
    result = result.rename(columns={"label": "y_true"})
    result["y_pred"] = [item["y_pred"] for item in predictions]
    result["raw_output"] = [item["raw_output"] for item in predictions]
    result["valid"] = result["y_pred"].ne(INVALID)
    result["label_policy_version"] = POLICY_VERSION
    metrics = metrics_payload(result["y_true"].tolist(), result["y_pred"].tolist())
    return metrics, result


def _checkpoint_candidates(run_dir: Path, explicit: str | None) -> list[Path]:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
        return [path]
    candidates = sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else 0,
    )
    return candidates or [run_dir]


def _score(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["macro_f1"]),
        -float(metrics["invalid_rate"]),
        float(metrics["coverage"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Training output directory containing run_config.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--checkpoint", help="Evaluate one checkpoint instead of auto-selecting on dev")
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Debug only: cap rows per split")
    parser.add_argument("--out", help="Output directory (default: run directory)")
    args = parser.parse_args(argv)

    if args.split == "test" and not args.confirm_locked_test:
        parser.error("locked-test evaluation requires --confirm-locked-test; selection must use dev")
    if args.device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import pandas as pd

    run_dir = Path(args.run)
    config = _load_run(run_dir)
    out = Path(args.out) if args.out else run_dir
    out.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    data_path = config.get("data_file")
    if not data_path or not Path(data_path).exists():
        raise SystemExit(f"Training data file not found: {data_path!r}; cannot evaluate stable example IDs")
    frame = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    frame = frame[frame["split"] == "dev"].copy() if args.split == "dev" else frame[frame["split"].isin(["dev", "test"])].copy()
    if args.limit:
        frame = frame.groupby("split", group_keys=False).head(args.limit)

    candidates = _checkpoint_candidates(run_dir, args.checkpoint)
    selection: list[dict[str, Any]] = []
    best_path: Path | None = None
    best_metrics: dict[str, Any] | None = None
    best_predictions = None
    for candidate in candidates:
        dev_rows = frame[frame["split"] == "dev"].copy()
        if dev_rows.empty:
            raise SystemExit("No dev rows for checkpoint selection")
        metrics, predictions = evaluate_checkpoint(
            run_dir, candidate, dev_rows, "dev", config, device, args.batch, args.max_new_tokens
        )
        selection.append({"checkpoint": str(candidate), "metrics": metrics})
        if best_metrics is None or _score(metrics) > _score(best_metrics):
            best_path, best_metrics, best_predictions = candidate, metrics, predictions
        print(json.dumps({
            "checkpoint": str(candidate),
            "dev_macro_f1": metrics["macro_f1"],
            "dev_invalid_rate": metrics["invalid_rate"],
        }))

    assert best_path is not None and best_metrics is not None and best_predictions is not None
    (out / "checkpoint_selection.json").write_text(json.dumps({
        "selection_split": "dev",
        "metric": "macro_f1 (invalid outputs count as errors)",
        "tie_breakers": ["lower invalid_rate", "higher coverage"],
        "candidates": selection,
        "selected_checkpoint": str(best_path),
    }, indent=2))
    best_predictions.to_csv(out / "dev_predictions.csv", index=False)
    (out / "dev_metrics.json").write_text(json.dumps(best_metrics, indent=2))

    final_metrics = best_metrics
    if args.split == "test":
        test_rows = frame[frame["split"] == "test"].copy()
        if test_rows.empty:
            raise SystemExit("No test rows to evaluate")
        # Reload exactly the dev-selected checkpoint for the single locked-test confirmation.
        test_metrics, test_predictions = evaluate_checkpoint(
            run_dir, best_path, test_rows, "test", config, device, args.batch, args.max_new_tokens
        )
        test_predictions.to_csv(out / "test_predictions.csv", index=False)
        (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        final_metrics = test_metrics

    manifest = {
        "arm": "generative",
        "policy_version": POLICY_VERSION,
        "run_dir": str(run_dir.resolve()),
        "selected_checkpoint": str(best_path),
        "selection_split": "dev",
        "evaluated_split": args.split,
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "batch": args.batch,
        },
        "model": config.get("model"),
        "environment": runtime_environment(),
        "metrics": final_metrics,
        "invalid_output_policy": "kept as __INVALID__; macro_f1 counts them as errors; macro_f1_valid_only excludes them",
        "test_confirmed": bool(args.split == "test" and args.confirm_locked_test),
    }
    (out / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "selected_checkpoint": str(best_path),
        "split": args.split,
        "macro_f1": final_metrics["macro_f1"],
        "macro_f1_valid_only": final_metrics["macro_f1_valid_only"],
        "invalid_rate": final_metrics["invalid_rate"],
        "current_recall": final_metrics["current_recall"],
        "current_precision": final_metrics["current_precision"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
