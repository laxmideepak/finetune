# H12-FINAL — protocol (locked before opening the SDOH-NLI test split)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY (final)

## Candidate

**Fine-tuned Flan-T5-base (250M)** on evidence sentences — the H9 leader on SDOH-NLI
5-fold CV: pooled macro-F1 0.868 vs TF-IDF 0.779 (Δ +0.090, paired-bootstrap 95% CI
+0.030…+0.151; wins 5/5 folds; NEVER F1 0.68 vs 0.49). H11 (3-way vote, 0.840) did not beat
it; H8 SciBERT (generic recipe) and all zero-shot arms were significantly worse.

## What is frozen (no changes after this commit)

- Recipe: `scripts/run_h9_flan_cv.py` `finetune()` — OPTIONS prompt, answer strings, 3 epochs,
  lr 3e-4, AdamW wd 0.01, warm-up 10%, batch 16, minority oversampling, seed 42; rank
  classification without calibration; no evidence ⇒ UNKNOWN.
- Training data: all SDOH-NLI train premises with evidence (n ≈ 239 + denial/no-mention rows
  that have evidence).
- Reference: TF-IDF SVM on the same evidence sentences and data.

## Tests

1. **SDOH-NLI test (n = 108; locked until now; opened once with `--confirm-locked-test`).**
   Primary metric macro-F1; also smoking-bearing macro-F1, per-class F1, CURRENT R/P;
   paired bootstrap (5,000 resamples over premises) for Flan − TF-IDF.
2. SDOH-NLI validation (n = 109; never used for any fine-tuned Flan run) — same metrics.
3. **Wright end-to-end (dev, one-year rule):** Flan-T5-base fine-tuned on Wright train +
   SDOH-NLI train evidence (native labels) as the status classifier inside
   `src/pipeline.py` with quit-resolver-v1; compare with the TF-IDF pipeline (0.973 v2).
4. **Efficiency:** warm CPU inference throughput (sentences/s, 4 threads) and $/M evidence
   sentences at an assumed $0.17/h 4-vCPU machine, vs TF-IDF.

## Predictions

1. SDOH test: Flan-base ≥ TF-IDF + 3 macro-F1 points (point estimate); NEVER and CURRENT
   F1 not lower than TF-IDF. (n = 108 is small; the CI may include 0.)
2. SDOH validation: Flan-base ≥ TF-IDF (0.831).
3. Wright one-year macro-F1 with Flan-base inside the pipeline ≥ 0.96 (no regression
   beyond 1.5 points vs 0.973).
4. Flan-base CPU throughput ≥ 10 evidence sentences/s; cost per million sentences < $10.

## Decision rule (pre-registered)

If predictions 1 and 3 hold, Flan-T5-base (fine-tuned) becomes the recommended status model
for human-written text, with TF-IDF as the low-cost fallback. If prediction 1 fails, the
recommendation stays TF-IDF + rules and Flan-base is reported as "promising on CV, not
confirmed".
