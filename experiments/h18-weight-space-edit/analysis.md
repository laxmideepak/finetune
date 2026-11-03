# H18-WSP — analysis

**Verdict: REFUTED.** The fixed rank-1, gradient-free edit never helped: on every fold the
training-set grid chose α = 0 (no edit), so the out-of-fold result equals unedited zero-shot
Flan-T5-large.

| Fold | train macro-F1 by α (0 / 1 / 2 / 4 / 8) | chosen α | test macro-F1 |
|:-:|---|:-:|:-:|
| 0 | 0.671 / 0.447 / 0.360 / 0.349 / 0.304 | 0 | 0.707 |
| 1 | 0.676 / 0.654 / 0.641 / 0.612 / 0.269 | 0 | 0.690 |
| 2 | 0.665 / 0.315 / 0.348 / 0.307 / 0.373 | 0 | 0.751 |
| 3 | 0.680 / 0.609 / 0.416 / 0.344 / 0.291 | 0 | 0.671 |
| 4 | 0.698 / 0.678 / 0.622 / 0.376 / 0.273 | 0 | 0.594 |

Pooled: unchanged from H9 zero-shot large (0.678). For reference: + rule guard 0.750 (H10),
Flan-T5-base fine-tuned 0.868 (H9).

## Predictions vs outcome

1. Edit ≥ zero-shot + 4 points ❌ (edit rejected on every fold)
2. Not more than 2 points above the guard ✅ (trivially)
3. Below fine-tuned Flan-T5-base ✅

## Reading

- Even the smallest edit (α = 1) lowered training-set macro-F1 on every fold, sometimes by more than
  20 points: amplifying the denial-vs-never difference in the last encoder block's MLP output
  pushes all inputs off the distribution the decoder expects, instead of moving only denial
  inputs towards "unknown".
- This was a single fixed closed-form attempt, not the paper's setup (a frontier model reasoning
  over the weights for up to 158 turns with up to 50 forward passes, choosing where and how to edit).
  It is a weak test of Weight-Space Programming as such, and a fair negative for a one-shot recipe.
- Consistent with the paper's own Gemma finding: an edit can remove an obstacle (a loop), but the
  denial/never boundary is a learned policy distinction, which supervision supplies cheaply here
  (Flan-T5-base fine-tuned, ~6 CPU-minutes per fold, 0.868). The external rule guard (0.750) gets the
  same policy fix with no risk to the model.
