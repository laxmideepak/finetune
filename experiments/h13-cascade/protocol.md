# H13-CASCADE — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** EXPLORATORY (SDOH-NLI train 5-fold CV, out-of-fold predictions only;
the locked test was already used once by H12 and is not reopened)

## Why

H12: fine-tuned Flan-T5-base is clearly more accurate on human-written text, but costs
≈ $5 per million evidence sentences on CPU vs ≈ $0.04 per million notes for the TF-IDF
pipeline. Most evidence sentences are easy ("Never smoker.", "Tob: current, 40 PY").

## Hypothesis

**H13:** routing only the sentences where TF-IDF is unsure to Flan-T5-base recovers most of
Flan's accuracy at a fraction of its cost.

## Design

- Confidence = margin between the top two LinearSVM decision scores (TF-IDF trained on the
  fold's training part, same model as H8 TF-IDF S).
- For call budgets b ∈ {10%, 20%, 30%, 40%, 50%} of evidence-bearing premises: send the b%
  lowest-margin premises in each test fold to Flan-T5-base (its H9 out-of-fold prediction),
  keep TF-IDF for the rest. Budgets are set per fold from that fold's own margins (no
  threshold is tuned on labels).
- Report pooled macro-F1, NEVER F1, Δ vs TF-IDF and vs Flan-alone, and cost per million
  sentences = TF-IDF cost + b × Flan cost.

## Prediction

At b = 30%, the cascade keeps ≥ 80% of Flan's CV gain over TF-IDF (≥ 0.851 pooled macro-F1)
at ≤ 35% of Flan-alone cost.

## Refutation

At b = 50% the cascade still keeps < 50% of the gain — TF-IDF's margin does not identify
the sentences where it is wrong.
