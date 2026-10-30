# H9-FLAN — analysis

**Verdict: SUPPORTED for fine-tuned Flan-T5-base; zero-shot predictions as expected.**

Pooled out-of-fold SDOH-NLI train 5-fold CV (n = 570; same folds as H8), paired bootstrap vs
TF-IDF (`scripts/compare_sdoh_cv.py`, `experiments/sdoh-cv-leaderboard.json`):

| Arm | pooled macro-F1 | folds | NEVER F1 | Δ vs TF-IDF [95% CI] |
|---|:-:|---|:-:|---|
| **Flan-T5-base fine-tuned (250M)** | **0.868** | 0.794 / 0.939 / 0.855 / 0.867 / 0.901 | **0.68** | **+0.090 [+0.030, +0.151]**, p≤0 = 0.001 |
| Flan-T5-small fine-tuned (77M) | 0.795 | 0.818 / 0.688 / 0.856 / 0.816 / 0.809 | 0.49 | +0.017 [−0.045, +0.079] |
| TF-IDF (S) — reference | 0.779 | 0.752 / 0.785 / 0.758 / 0.730 / 0.855 | 0.49 | — |
| Flan-T5-large zero-shot (780M) | 0.678 | | 0.26 | −0.100 [−0.169, −0.033] |
| Flan-T5-base zero-shot | 0.614 | | 0.23 | −0.164 |
| Flan-T5-small zero-shot | 0.435 (fold mean) | | | |

Zero-shot on Wright dev (native): small 0.594, base 0.822, large 0.809. Throughput
(zero-shot, calibrated, 4 threads, shared CPU): small 30, base 10.4, large 2.8 sentences/s.

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | ZS-base in 0.55–0.75, below TF-IDF | ✅ 0.616 |
| 2 | ZS-large ≥ ZS-base + 5 | ✅ +6.4 (pooled) |
| 3 | FT-base ≥ TF-IDF + 3 with NEVER +0.15 | ✅ +9.0, NEVER +0.19 |
| 4 | FT-base ≥ SciBERT (H8 generic) | ✅ 0.868 vs 0.682 |

## What we learned

1. **A small instruction-tuned seq2seq LM, fine-tuned on ~240 smoking sentences, beats the
   supervised bag-of-words model significantly on human-written text** — the first model to do
   so in this project. It wins every fold.
2. **Size matters more for fine-tuning than for zero-shot here**: small (77M) ties TF-IDF;
   base (250M) wins clearly; large (780M) was not fine-tuned on CPU in this wave.
3. **Instruction tuning + answer-string targets beat a classification head** at this data size
   (Flan-base FT 0.868 vs SciBERT generic 0.682): the model starts from a prompt it already
   understands, so ~70 updates are enough; the BERT head starts from random weights.
4. Fine-tuning takes ~6 min per fold on 4 CPU threads — no GPU was needed.

Next: H12-FINAL (locked SDOH test, Wright pipeline, cost).
