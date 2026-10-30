# H16-REPEAT — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY (robustness of the H9/H12 claim)

## Why

The recommendation rests on one 5-fold split of SDOH-NLI train (random_state 0) and a locked
test with only 15 smoking-labelled premises. The audit asked for evidence that the Flan-T5-base
advantage is not an artefact of one split.

## Design

- Three NEW stratified 5-fold splits of SDOH-NLI train: `StratifiedKFold(5, shuffle=True,
  random_state=r)` for r ∈ {1, 2, 3} (r = 0 was H9).
- Arms per fold (identical evidence-sentence input, identical training rows — evidence-bearing
  premises only, the audit's like-for-like rule):
  - Flan-T5-base fine-tuned, frozen H9 recipe (AdamW 3e-4, 3 epochs, batch 16, oversampling, seed 42)
  - TF-IDF word+char LinearSVM
- No-evidence premises ⇒ UNKNOWN for both. SDOH validation/test untouched.

## Metrics

Per repeat: pooled macro-F1 for each arm, Δ (Flan − TF-IDF), paired bootstrap 95% CI
(5,000 resamples), folds won. Across repeats: mean Δ, and whether every repeat's Δ > 0. Also the
3-class (NEVER+UNKNOWN merged) Δ.

## Predictions

1. Flan − TF-IDF > 0 in all three repeats (4-class), mean Δ ≥ +0.05.
2. Flan wins ≥ 12 of the 15 folds.
3. 3-class Δ > 0 in all three repeats.

## Refutation

Any repeat with Δ ≤ 0, or fewer than 10/15 fold wins — the advantage would then depend on the split.
