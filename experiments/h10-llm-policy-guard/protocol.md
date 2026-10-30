# H10-GUARD — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY on SDOH-NLI validation; EXPLORATORY on train CV

## Observation that motivates it (aggregate confusion only, no item-level tuning)

H9 zero-shot Flan-T5-large on SDOH-NLI train (no training): NEVER recall 16/19, CURRENT
47/59, FORMER 32/35 — far better than TF-IDF on the language-heavy classes — but 82 of 457
UNKNOWN premises are predicted NEVER. Most UNKNOWN premises with evidence are
"denial-only" ("currently not a smoker", "denies smoking"), which our policy maps to
UNKNOWN because denial does not establish lifetime non-use. That is a policy rule, not a
language-understanding gap — the same split as H5 (model reads language, a deterministic
layer applies the policy).

## Hypothesis

**H10-GUARD:** a zero-shot instruction-tuned LM followed by a policy guard
(NEVER without an explicit lifetime-never cue ⇒ UNKNOWN(denial-only)) beats the
supervised TF-IDF model on human-written text, without any task training.

Guard definition (fixed now, from `src/status_rules.NEVER`, which was written from the
label policy before H9 and frozen at H7): if the model says NEVER and the evidence text
does not match `status_rules.NEVER`, output UNKNOWN. Nothing else changes.

## Arms

| Arm | Description |
|---|---|
| ZS-large | Flan-T5-large zero-shot (H9 setup) |
| ZS-large + guard | ZS-large then the guard |
| ZS-base + guard | Flan-T5-base zero-shot then the guard |
| TF-IDF S | supervised reference (trained on SDOH-NLI train) |
| Rules | keyword rules (H7) |

## Evaluation

- **Confirmatory:** SDOH-NLI validation (n = 109; never used for any Flan run or for guard
  design). Note its single NEVER premise: report per-class scores and smoking-bearing
  macro-F1 alongside macro-F1.
- **Exploratory:** the same arms on SDOH-NLI train 5-fold CV folds (zero-shot arms have no
  training, so CV only matters for TF-IDF), and on Wright dev (native).

## Predictions

1. ZS-large + guard ≥ TF-IDF S on train CV mean (0.776) by ≥ 2 points.
2. On validation, ZS-large + guard ≥ ZS-large + 10 points (the guard removes the
   denial→NEVER errors).
3. The guard changes ≤ 2% of Wright dev predictions in the wrong direction (Wright NEVER
   statements mostly carry lifetime cues).

## Refutation

ZS-large + guard below TF-IDF S on train CV — then supervised in-domain data still beats a
policy-guarded zero-shot LM on this corpus.
