"""Warm classifier compute benchmark (tokenization + forward, no retrieval).

Reads a CSV with a ``text`` column of representative held-out inputs. The same
``--max-length`` routing rule as inference is enforced: overlength inputs raise
rather than truncate silently.

Hardware/price honesty: a rented-GPU price may only be combined with throughput
measured on CUDA; an MPS price only with MPS; a CPU price only with CPU. The
result is "warm classifier compute cost", excluding retrieval, aggregation,
fallback passes, startup, and I/O.

Examples:
    python src/benchmark_encoder.py --model runs/c1/model --csv tests/fixtures/smoke_texts.csv \
        --device mps --batch 4 --passes 3
    python src/benchmark_encoder.py --model runs/c1/model --csv heldout.csv \
        --device cuda --price-hardware a10g --hourly-price 1.00
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


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


def _resolve_dtype(requested: str, device: str):
    import torch

    if requested == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    if requested == "bf16":
        if device != "cuda" or not torch.cuda.is_bf16_supported():
            raise SystemExit("bf16 requested but not supported on this device")
        return torch.bfloat16
    if requested == "fp16":
        if device == "cpu":
            raise SystemExit("fp16 is not supported for this CPU benchmark path")
        return torch.float16
    return torch.float32


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _device_name(device: str) -> str:
    import torch

    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple MPS ({platform.machine()})"
    return platform.processor() or platform.machine()


def _peak_memory(device: str) -> dict[str, float | None]:
    import torch

    if device == "cuda":
        return {
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        }
    if device == "mps" and hasattr(torch, "mps"):
        return {
            "peak_allocated_gb": torch.mps.current_allocated_memory() / 1e9,
            "peak_reserved_gb": (
                torch.mps.driver_allocated_memory() / 1e9
                if hasattr(torch.mps, "driver_allocated_memory") else None
            ),
        }
    return {"peak_allocated_gb": None, "peak_reserved_gb": None}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "numpy", "pandas"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--hourly-price", type=float, default=None)
    parser.add_argument(
        "--price-hardware",
        choices=("a10g", "l4", "t4", "cuda-other", "mps", "cpu"),
        default=None,
        help="Must match the measured device; Mac throughput cannot be priced as a rented GPU",
    )
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    if args.batch < 1 or args.passes < 1:
        parser.error("--batch and --passes must be >= 1")
    if args.max_length < 1:
        parser.error("--max-length must be >= 1")
    if (args.hourly_price is None) != (args.price_hardware is None):
        parser.error("--hourly-price and --price-hardware must be provided together")
    if args.device == "mps":
        import os

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    frame = pd.read_csv(args.csv, keep_default_na=False)
    if "text" not in frame.columns:
        raise SystemExit("CSV must contain a text column")
    texts = frame["text"].tolist()
    if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
        raise SystemExit("Nonempty text rows required")

    device = _resolve_device(args.device)
    if args.hourly_price is not None:
        gpu_prices = {"a10g", "l4", "t4", "cuda-other"}
        if args.price_hardware in gpu_prices and device != "cuda":
            raise SystemExit("Rented-GPU pricing requires CUDA throughput; do not combine Mac/CPU with GPU prices")
        if args.price_hardware == "mps" and device != "mps":
            raise SystemExit("MPS pricing requires --device mps")
        if args.price_hardware == "cpu" and device != "cpu":
            raise SystemExit("CPU pricing requires --device cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
    overlength = [length for length in lengths if length > args.max_length]
    if overlength:
        raise SystemExit(
            f"{len(overlength)} inputs exceed max_length={args.max_length} (max={max(overlength)}); "
            "route overflow instead of truncating"
        )

    dtype = _resolve_dtype(args.dtype, device)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=args.revision)
    model.to(device=device, dtype=dtype).eval()

    def run(batch: list[str]):
        encoded = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
        non_pad_tokens = int(encoded["attention_mask"].sum().item())
        padded_tokens = int(encoded["input_ids"].numel())
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        return logits, non_pad_tokens, padded_tokens

    # Warm up representative length buckets, not just the first batch.
    order = sorted(range(len(texts)), key=lambda index: lengths[index])
    warmup_indices: list[int] = []
    for fraction in (0.0, 0.5, 1.0):
        start = int(fraction * max(0, len(order) - args.batch))
        warmup_indices.extend(order[start:start + args.batch])
    warmup_batches = [texts[index] for index in warmup_indices]
    for start in range(0, len(warmup_batches), args.batch):
        run(warmup_batches[start:start + args.batch])
    _sync(device)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    pass_results: list[dict[str, float]] = []
    all_batch_latencies: list[float] = []
    for _ in range(args.passes):
        batch_latencies: list[float] = []
        non_pad_tokens_total = 0
        padded_tokens_total = 0
        _sync(device)
        pass_start = time.perf_counter()
        for start in range(0, len(texts), args.batch):
            batch = texts[start:start + args.batch]
            _sync(device)
            batch_start = time.perf_counter()
            _logits, non_pad_tokens, padded_tokens = run(batch)
            _sync(device)
            batch_latencies.append(time.perf_counter() - batch_start)
            non_pad_tokens_total += non_pad_tokens
            padded_tokens_total += padded_tokens
        elapsed = time.perf_counter() - pass_start
        all_batch_latencies.extend(batch_latencies)
        pass_results.append({
            "elapsed_s": elapsed,
            "snippets_per_second": len(texts) / elapsed,
            "non_pad_tokens_per_second": non_pad_tokens_total / elapsed,
            "padded_tokens_per_second": padded_tokens_total / elapsed,
        })

    snippet_rates = [result["snippets_per_second"] for result in pass_results]
    token_rates = [result["non_pad_tokens_per_second"] for result in pass_results]
    mean_rate = float(np.mean(snippet_rates))
    mean_token_rate = float(np.mean(token_rates))
    cost_per_million = (
        args.hourly_price * 1e6 / (3600 * mean_rate) if args.hourly_price is not None else None
    )
    cost_per_million_tokens = (
        args.hourly_price * 1e6 / (3600 * mean_token_rate) if args.hourly_price is not None else None
    )

    result = {
        "device": device,
        "device_name": _device_name(device),
        "dtype": str(dtype).replace("torch.", ""),
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "batch": args.batch,
        "n": len(texts),
        "max_length": args.max_length,
        "max_input_tokens": max(lengths),
        "passes": args.passes,
        "snippets_per_second": {
            "mean": mean_rate,
            "std": float(np.std(snippet_rates)),
            "min": float(np.min(snippet_rates)),
            "max": float(np.max(snippet_rates)),
            "per_pass": snippet_rates,
        },
        "non_pad_tokens_per_second": {
            "mean": mean_token_rate,
            "std": float(np.std(token_rates)),
            "min": float(np.min(token_rates)),
            "max": float(np.max(token_rates)),
        },
        "batch_latency_ms_p50_p95": (np.percentile(all_batch_latencies, [50, 95]) * 1000).tolist(),
        "peak_memory": _peak_memory(device),
        "hourly_price": args.hourly_price,
        "price_hardware": args.price_hardware,
        "cost_per_million_snippets": cost_per_million,
        "cost_per_million_non_pad_tokens": cost_per_million_tokens,
        "versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "scope": "warm classifier compute (tokenization + forward); excludes retrieval, aggregation, fallback passes, startup, and I/O",
        "note": "Batch latency is measured with a device sync per batch; throughput is end-to-end over each pass.",
    }
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
