# H14-LARGE — analysis

**Verdict: REFUTED. Flan-T5-base stays the recommended model.**

Pooled SDOH-NLI train 5-fold CV (same folds as H8/H9), evidence sentences, rank classification:

| Arm | pooled macro-F1 | folds | NEVER F1 | vs TF-IDF (S) |
|---|:-:|---|:-:|---|
| B-adamw: Flan-T5-base, AdamW 3e-4 (H9) | **0.868** | 0.794 / 0.939 / 0.855 / 0.867 / 0.901 | 0.68 | +0.090 [+0.030, +0.151] |
| B-ada: Flan-T5-base, Adafactor 1e-3 (control) | 0.854 | 0.818 / 0.860 / 0.891 / 0.867 / 0.829 | 0.67 | +0.076 [+0.023, +0.133] |
| L-ada: Flan-T5-large (780M), Adafactor 1e-3 | 0.813 | 0.693 / 0.911 / 0.831 / 0.881 / 0.877 | 0.47 | +0.036 [−0.033, +0.103] |

Paired bootstrap (5,000 resamples over premises): L-ada − B-adamw = **−0.054 [−0.105, +0.001]**;
L-ada − B-ada (same optimizer) = −0.041 [−0.091, +0.012].

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | L-ada ≥ B-adamw + 2 points | ❌ −5.4 points |
| 2 | B-ada within ±2 of B-adamw | ✅ −1.4 |
| 3 | Large ≈ 3× slower | ✅ training ~36 min/fold vs ~11 min (base, Adafactor) |

## Reading

On ~240 smoking sentences per training set, the 780M model does not use its extra capacity; its
NEVER class (≈15 training examples per fold) is learned worst (F1 0.47). The optimizer change
alone costs only 1.4 points, so the gap is due to size, not recipe. Bigger is not better at this
data size; Flan-T5-base (250M) is both lighter and more accurate. A larger model might win with
more labelled data (e.g. i2b2-2006), which remains a future test.

Run note: the first large run was restarted after fold 0 to fix a memory bug (the previous
fold's model stayed in memory); fold 0's saved result was kept, the rest reran with the fix.
