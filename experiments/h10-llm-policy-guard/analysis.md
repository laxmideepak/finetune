# H10-GUARD — analysis

**Verdict: REFUTED (confirmatory, SDOH-NLI validation) / partly supported (exploratory).**

| Arm | SDOH validation macro-F1 (n=109, confirmatory) | smoking-bearing | train 5-fold CV (exploratory) |
|---|:-:|:-:|:-:|
| Flan-T5-base zero-shot | — | — | 0.616 |
| Flan-T5-base ZS + guard | — | — | 0.716 |
| Flan-T5-large zero-shot | 0.529 | 0.398 | 0.683 |
| Flan-T5-large ZS + guard | 0.504 | 0.456 | 0.751 |
| **TF-IDF S (supervised)** | **0.831** | **0.819** | **0.776** |
| Keyword rules | 0.666 | 0.629 | — |

Wright dev (native): ZS-large 0.809 → +guard 0.830; the guard changed 6 predictions, 0 wrongly.

## Predictions vs outcome

1. ZS-large + guard ≥ TF-IDF + 2 on train CV ❌ (−2.5)
2. Guard adds ≥ 10 points on validation ❌ (−2.5; the only NEVER premise in validation lacks a
   lifetime cue the rules recognise, so the guard flipped it to UNKNOWN)
3. Guard rarely wrong on Wright ✅ (6 changes, 0 wrong)

## What we learned

1. **The policy guard is the right idea but not enough.** Across the larger train set it adds
   +6.8 (large) / +10.0 (base) points by fixing denial-only → NEVER, but the guard's lifetime
   lexicon also removes some true NEVERs.
2. **Zero-shot Flan-T5 misreads durations as the past**: "She smokes one pack a day for last
   30 years" → FORMER. Supervised data teaches the model that construction; zero-shot does not.
3. **Part of TF-IDF's edge is learning SDOH-NLI's own label habits**: several validation
   "CURRENT" items are past-tense or pack-years-only ("smoked 1 pack per day for 15 years",
   "120pk-yr smoking"), which our policy would not call CURRENT. A model that follows our
   policy is penalised on these items.
4. **Conclusion so far (H7–H10)**: on human-written text with ~240 smoking examples, no
   tested pretrained model beats the supervised bag-of-words classifier *zero-shot or with a
   generic fine-tune*. Fine-tuned Flan-T5 (H9) and the small-data SciBERT recipe (H8b) are
   the remaining tests.
