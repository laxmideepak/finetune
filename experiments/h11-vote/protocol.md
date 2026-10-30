# H11-VOTE — protocol (locked before Flan-T5-base fine-tuning finished; 2 of 5 folds seen)

**Date locked:** 2026-10-30 · **Type:** EXPLORATORY (uses existing out-of-fold predictions; no new training)

## Why

The three strongest families make different errors on SDOH-NLI: TF-IDF learns the corpus's
own label habits; fine-tuned Flan-T5 reads present-tense durations and negation; guarded
zero-shot Flan-T5-large is strongest on NEVER without training. A majority vote is the
cheapest way to test whether those errors are complementary.

## Rule (fixed now)

Voters on each SDOH-NLI train premise (out-of-fold predictions, same 5 folds):
1. TF-IDF (S) — H8
2. Flan-T5-base fine-tuned — H9 (whatever its final score)
3. Flan-T5-large zero-shot + policy guard — H10

Label = the label at least two voters give; if all three differ, use voter 2.

## Prediction

Vote pooled macro-F1 ≥ best single voter + 1 point, NEVER F1 ≥ 0.55.

## Note

Exploratory: the voter set was chosen after seeing H7–H10, and no held-out split is used.
If it wins, it must be confirmed on SDOH-NLI test (locked) before any claim.
