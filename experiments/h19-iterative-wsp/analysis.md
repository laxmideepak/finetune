# H19-ITERATIVE-WSP — analysis

**Verdict: SUPPORTED (all three predictions).** Iterative weight-space programming made a real,
held-out improvement to zero-shot Flan-T5-large without any training — but supervised fine-tuning of
a smaller model remains far better.

## The run (16 turns, 22 of 50 forward passes; full log in `log/calls.jsonl`, scripts in `turn_scripts/`)

1. Baseline on development (SDOH train folds 1–4): macro-F1 0.671; the dominant error is
   denial-only → NEVER (65 of 97).
2. Decomposed the "never smoked" vs "unknown" decision along d = W_U[never] − W_U[unknown]: the push
   comes from decoder cross-attention (blocks 19 and 22), and is the same for wrong denials and true
   NEVER — the model represents them alike.
3. Per-head analysis: heads 19.1, 19.7 and 22.1 read the encoder state of the **"never" option word**
   in the prompt's OPTIONS list (not the patient's sentence directly).
4. A direction in that option-word state separates denials from true NEVER (mean-difference LOO
   accuracy 0.75; shrinkage-LDA LOO AUC 0.93).
5. Edit: rank-1 changes to the value weights of the three heads, W_v,h ← W_v,h − β a_h ŵᵀ, gated by
   that direction. Three corrections were needed, each found from a failed pass:
   - centre the direction against the calibration ("N/A") input and the shared component (turn 7);
   - make a_h orthogonal to the current/former answer directions so those classes are untouched
     (turn 10);
   - use the shrinkage-LDA direction, neutralised for current/former premises (turns 12–15).
6. Development: 0.671 → **0.771** (denial → NEVER 65 → 15; true NEVER 16/16). A second edit for
   CURRENT → FORMER was not attempted: only ~5 genuine cases, several others are SDOH label quirks.
7. Final edit: 196,608 of 849M weights (**0.023%**), frozen and committed (360f3b2, ed06952) before
   evaluation.

## One-time evaluation (SDOH train fold 0 + validation, n = 223, never seen by the editor)

| Arm | macro-F1 | NEVER F1 | fold 0 | validation | Δ vs zero-shot [95% CI] |
|---|:-:|:-:|:-:|:-:|---|
| Flan-T5-large zero-shot | 0.619 | 0.18 | 0.707 | 0.529 | — |
| zero-shot + rule guard | 0.634 | 0.18 | 0.736 | 0.504 | +0.015 [−0.053, +0.090] |
| **zero-shot + iterative weight edit** | **0.673** | **0.35** | 0.752 | 0.571 | **+0.054 [+0.003, +0.105]** |
| weight edit + rule guard (exploratory) | 0.688 | 0.40 | 0.770 | 0.517 | +0.069 [−0.058, +0.222] |
| TF-IDF (same development data) | 0.769 | 0.40 | 0.752 | 0.823 | +0.150 |
| **Flan-T5-base fine-tuned (same development data)** | **0.786** | 0.35 | 0.794 | 0.777 | **+0.167 [+0.084, +0.254]** |

Weight edit − fine-tuned Flan-T5-base: **−0.113 [−0.211, −0.014]**.

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | edit ≥ zero-shot + 5 points | ✅ +5.4 (CI lower bound +0.3) |
| 2 | edit below fine-tuned Flan-T5-base | ✅ −11.3, significant |
| 3 | edit changes < 1% of weights | ✅ 0.023% |

## What we learned

1. **Weight-space programming works here in the way the paper says it should**: it removed a specific,
   mechanistically located obstacle (denials conflated with lifetime non-use) in a model that already
   had the language for it — +5.4 points held-out, with 0.023% of the weights and no training. The
   development gain (+10) shrank by about half on unseen data: the direction was estimated from ~106
   examples.
2. **It does not replace fine-tuning.** Fine-tuning a smaller model for ~6 CPU-minutes on the same data
   is +11 points better and teaches what an edit cannot (e.g. present-tense durations).
3. **Mechanistic finding:** in Flan-T5 the 4-way decision is read off the encoder's representation of
   the *answer-option words*, by three cross-attention heads — useful for auditing any fine-tuned
   version too.
4. Editor note: Claude, not GPT-6 Astra, ran this harness; 16 turns and ~4 hours of CPU time.
