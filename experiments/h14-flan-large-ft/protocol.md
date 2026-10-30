# H14-LARGE — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY on SDOH-NLI train 5-fold CV (same folds as H8/H9);
the SDOH test split is NOT reopened

## Why

H9/H12 made fine-tuned Flan-T5-base (250M) the recommended model (CV 0.868; locked test 0.877).
"Best" requires checking whether the next size up, Flan-T5-large (780M), is better enough to be
worth ~3× the compute. Zero-shot, large beat base by +6.4 (H9). AdamW for 780M parameters needs
≈ 12 GB for weights + optimizer state + gradients; this machine's limit is ≈ 14 GB and two runs
were already OOM-killed. Adafactor (Shazeer & Stern 2018; the optimizer T5 was trained with)
keeps factored second moments and fits.

## Arms (evidence sentences, SDOH-NLI train folds, rank classification as in H9)

| Arm | Model | Optimizer | Other settings |
|---|---|---|---|
| L-ada | Flan-T5-large | Adafactor, lr 1e-3, no relative step, no parameter scaling | 3 epochs, batch 16, warm-up 10% + linear decay, oversampling, seed 42 |
| B-ada (control) | Flan-T5-base | same Adafactor settings | same |
| B-adamw (reference) | Flan-T5-base | H9 recipe (AdamW 3e-4) | existing H9 out-of-fold predictions |

B-ada separates the optimizer change from the size change.

## Predictions

1. L-ada ≥ B-adamw + 2 points pooled macro-F1 (≥ 0.888).
2. B-ada within ±2 points of B-adamw (the optimizer change alone is not the story).
3. L-ada inference ≈ 3× slower than base (≈ 3 sentences/s on 4 threads).

## Decision rule

The recommended model changes to Flan-T5-large only if L-ada beats B-adamw with a paired
bootstrap 95% CI above 0 **and** the cascade cost at the 40% operating point stays below
$5 per million sentences. Otherwise Flan-T5-base stays (lighter, already confirmed).
