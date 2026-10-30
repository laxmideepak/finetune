# H15-WRIGHT-TEST — protocol (locked before opening the Wright test split)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY (final, held-out)

## Why

The audit found that every Wright number so far (incl. H12's 0.972) comes from the dev split,
which was scored in ~5 experiments. The Wright **test** split (600 notes, patient-disjoint,
seed 20261029) has never been scored. It is opened once here, for the two final pipelines.

## Pipelines (frozen; no change after this commit)

1. **Flan pipeline** — `src/pipeline.py` with Flan-T5-base fine-tuned exactly as in H12's Wright
   stage (Wright train + SDOH-NLI train evidence, native labels, AdamW 3e-4, 3 epochs, batch 16,
   oversampling, seed 42) + quit-resolver-v1 (frozen at eec6f03) + retrieval-v2.
2. **TF-IDF pipeline** — same pipeline with the TF-IDF SVM trained on the same rows.

The trained Flan model is saved this time, and per-note predictions are written.

## Metrics

One-year macro-F1 (v1 and v2 labels), per-class F1, CURRENT recall/precision, native-status
accuracy, per-condition (basic/messy/complex) macro-F1, recent-quitter recall.

## Predictions

1. Both pipelines ≥ 0.95 one-year macro-F1 (v2) on Wright test (dev: 0.972 / 0.973).
2. The two pipelines are within 1.5 points of each other (Wright is saturated for status).
3. CURRENT recall ≥ 0.98 for both.

## Run order

After H14 finishes (CPU and memory). Allocator settings as in H12 (`MALLOC_ARENA_MAX=2`).
