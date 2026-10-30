# H16-REPEAT — analysis

**Verdict: SUPPORTED (2 of 3 predictions; no refutation criterion met).** Fine-tuned Flan-T5-base
beats like-for-like TF-IDF on every one of four independent 5-fold splits of SDOH-NLI.

| Split | Flan-T5-base | TF-IDF (same rows) | Δ 4-class [95% CI] | Δ 3-class (NEVER+UNKNOWN merged) [95% CI] | folds won |
|---|:-:|:-:|:-:|:-:|:-:|
| r=0 (H9, audit-corrected) | 0.868 | 0.791 | +0.077 [+0.016, +0.142] | +0.057 [+0.012, +0.104] | 5/5 |
| r=1 | 0.857 | 0.802 | +0.055 [−0.012, +0.122] | +0.005 [−0.040, +0.048] | 3/5 |
| r=2 | 0.822 | 0.759 | +0.062 [+0.000, +0.128] | +0.048 [+0.005, +0.095] | 4/5 |
| r=3 | 0.884 | 0.774 | +0.110 [+0.046, +0.178] | +0.029 [−0.007, +0.068] | 4/5 |

New repeats (r = 1–3): mean Δ **+0.076**; all three Δ > 0 (4-class and 3-class); folds won 11/15
(16/20 including r = 0).

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | Δ > 0 in all three repeats, mean ≥ +0.05 | ✅ +0.055 / +0.062 / +0.110, mean +0.076 |
| 2 | ≥ 12 of 15 fold wins | ❌ 11/15 (refutation threshold was < 10) |
| 3 | 3-class Δ > 0 in all three repeats | ✅ but small: +0.005 / +0.048 / +0.029 |

## Reading

- The 4-class advantage does not depend on one lucky split: four independent splits all favour
  Flan by +0.055 to +0.110.
- Most of the advantage comes from the NEVER / UNKNOWN / denial boundary (the language-heavy part).
  On CURRENT vs FORMER vs other alone, Flan's edge is consistent in sign but small (+0.005 to
  +0.057) and not significant in two of four splits.
- With ~240 smoking examples per training set, single-fold results swing by ±0.1; repeated splits
  are needed for any claim, and this is now done.

Run note: a container restart interrupted r = 3 after fold 2; completed folds were kept on disk and
the run resumed (each fold is an independent train/test, so nothing else changed).
