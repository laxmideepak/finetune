# H18-WSP — protocol (locked before running)

**Date locked:** 2026-11-03 · **Type:** CONFIRMATORY (SDOH-NLI train 5-fold CV, H9 folds)

## Motivation

"Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2026-11-02; logs at
github.com/RikiyaT/lm_understand-logs) edits model weights directly, without gradients, from
activations read through a small forward-pass budget. Their strongest result (Exp. 8) removes one
behaviour direction from a few MLP output matrices (W′ = W − α u uᵀ W) and fixes a looping failure;
their Gemma follow-up shows edits remove obstacles but do not add missing abilities.

Our clearest single-mechanism failure is the same kind: zero-shot Flan-T5-large (H9) maps
denial-only premises ("does not smoke", "denies smoking") to NEVER (82/457 UNKNOWN), while being
strong on true NEVER (16/19). This is a policy distinction the model has the language for.

## Hypothesis

**H18:** a rank-1, gradient-free weight edit of one MLP output matrix in Flan-T5-large's encoder,
computed in closed form from activations, separates denial-only from lifetime-never inputs and
raises zero-shot macro-F1 — without any training.

## Edit (fixed procedure; computed per fold from the fold's TRAINING part only)

- Model: Flan-T5-large (780M), zero-shot scoring exactly as H9/H10 (OPTIONS prompt, contextual
  calibration with "N/A", rank classification). Evidence-sentence input.
- Layer: the last encoder block's MLP output matrix `encoder.block[23].layer[1].DenseReluDense.wo`
  (1,024 × 2,816 = 2.88M weights; the edit is rank 1).
- From the fold's training premises with evidence, label groups D = UNKNOWN(denial-only) and
  N = NEVER:
  - key k = mean MLP hidden activation (gated-GELU output, averaged over prompt tokens) of D minus N;
  - value v = mean block-output residual (averaged over tokens) of D minus N.
  - ΔW = α · v kᵀ / (kᵀ k); W′ = W + ΔW (pushes denial-like inputs further from never-like inputs,
    gated by a denial-specific feature).
- α chosen from {0, 1, 2, 4, 8} by macro-F1 on the fold's training part (forward passes only; no
  backward pass, no optimiser, no autograd). α = 0 means "no edit".
- Weight-change share reported (rank-1 changes every entry of one matrix; we also report the
  fraction of total model weights in that matrix).

## Arms (pooled out-of-fold, same folds as H9; compared on identical premises)

ZS-large (H9) · ZS-large + rule guard (H10) · **ZS-large + weight edit (H18)** · Flan-T5-base fine-tuned (H9).

## Predictions

1. Weight-edited ZS-large ≥ ZS-large + 4 points pooled macro-F1 (≥ 0.718).
2. It does not beat the rule guard (0.750) by more than 2 points — an edit learned from ~30 contrast
   examples is a softer version of the same policy rule.
3. It stays below fine-tuned Flan-T5-base (0.868) — editing fixes one obstacle, it does not add the
   abilities fine-tuning teaches (cf. the paper's Gemma result).

## Refutation

Edited ZS-large ≤ ZS-large + 1 point: a rank-1 closed-form edit at this layer does not move the
denial/never decision.
