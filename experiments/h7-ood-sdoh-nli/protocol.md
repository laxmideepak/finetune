# H7-OOD — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY

## Why

H6 left Wright saturated (native status 600/600 on dev). Wright is LLM/template-written;
a model that is perfect there may still fail on human-written clinical text (risk R7;
precedents −13 pts n2c2/UW cross-domain, 0.97 → 0.86 SmokeBERT external). We need an
independent, human-written test.

## Data: SDOH-NLI (Google Research, CC-BY-4.0)

Source: `github.com/google-research-datasets/SDOH-NLI` (train / validation / test CSVs;
premises are social-history passages from MTSamples transcriptions). Each premise is paired
with four smoking hypotheses with True/False labels. **Deviation from design v0.9.1** (which
listed SDOH-NLI as "format only, not a label source"): the four hypotheses map
deterministically to our native taxonomy, so it becomes a second evaluation corpus.

Mapping (recorded in the harmonization register; note-level, native — no one-year rule,
because SDOH-NLI has no note dates):

| currently smoker | currently NOT smoker | was smoker in past | was NOT smoker in past | → label |
|:-:|:-:|:-:|:-:|---|
| T | · | · | · | CURRENT |
| F | T | T | F | FORMER |
| F | T | F | T | NEVER |
| F | T | F | F | UNKNOWN (denial-only) |
| F | F | T | F | UNKNOWN (past use, current status not stated) |
| F | F | F | F | UNKNOWN (no-mention) |

Splits: SDOH-NLI's own splits. **Validation = dev for this study; test stays locked.**
Labels are used as released; they are not our annotation and may disagree with our policy
at the edges.

## Arms (same model family as H6: evidence sentences → TF-IDF word+char → LinearSVM, native labels)

| Arm | Training data | Evaluated on |
|---|---|---|
| W→S (zero-shot transfer) | Wright train (frozen H6 configuration) | SDOH-NLI validation |
| S→S | SDOH-NLI train | SDOH-NLI validation |
| W+S→S | Wright train + SDOH-NLI train | SDOH-NLI validation |
| Rules | no training: keyword status rules (current / former / never / denial cues) | SDOH-NLI validation |
| W+S→W (regression check) | Wright train + SDOH-NLI train | Wright dev (native) |

Evidence sentences come from the unchanged extractor (retrieval-v2); a premise with no
used trigger is predicted UNKNOWN.

## Predictions

1. W→S drops ≥ 8 macro-F1 points below its Wright score (1.000 native) — synthetic →
   human-written shift.
2. S→S beats W→S, despite ~240 smoking-bearing training premises.
3. W+S→S is the best arm on SDOH-NLI validation and costs ≤ 1 point on Wright dev.
4. The rules arm is competitive with W→S (both lack in-domain data).

## Refutation

W→S within 3 points of 1.000 would mean Wright transfers to human text better than expected
(and the synthetic/real gap is small for status). W+S worse than S→S would mean synthetic
data hurts.

## Metrics

Native 4-class macro-F1 (primary), per-class F1, CURRENT recall/precision; paired
bootstrap CI over premises for W+S vs S→S; retrieval-negative rate on smoking-bearing premises.
