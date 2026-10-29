"""Warm, sequential-batch end-to-end benchmark including tokenization.
Input CSV: text. Use representative held-out lengths; no PHI printed.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--hourly-price", type=float, required=True)
    args = ap.parse_args()
    texts = pd.read_csv(args.csv, keep_default_na=False).text.tolist()
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError("Nonempty text rows required")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    if device == "cuda" and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
    @torch.inference_mode()
    def run(batch):
        encoded = tok(batch, padding=True, truncation=False, return_tensors="pt").to(device)
        return model(**encoded).logits.argmax(-1).cpu().tolist()
    for _ in range(5):
        run(texts[:args.batch])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    sync()
    start = time.perf_counter()
    for i in range(0, len(texts), args.batch):
        sync()
        t = time.perf_counter()
        run(texts[i:i+args.batch])
        sync()
        times.append(time.perf_counter()-t)
    elapsed = time.perf_counter()-start
    rate = len(texts)/elapsed
    print(json.dumps({"device": device, "batch": args.batch, "n": len(texts),
        "snippets_per_second": rate,
        "batch_latency_ms_p50_p95": (np.percentile(times, [50,95])*1000).tolist(),
        "compute_dollars_per_million": args.hourly_price*1e6/(3600*rate),
        "peak_cuda_allocated_GB": torch.cuda.max_memory_allocated()/1e9 if device=="cuda" else None,
        "note": "Warm batch latency, not per-request serving latency; excludes startup and I/O."}, indent=2))

if __name__ == "__main__":
    main()
