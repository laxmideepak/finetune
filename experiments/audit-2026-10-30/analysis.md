# Independent audit of the headline claims (2026-10-30)

Ten agents (four reviewers, one angle each, then a skeptic per major finding) audited the claims
without editing files. Raw output: `audit_workflow_output.json`. **No major finding was refuted.**

## Checked and OK

- No data leakage that inflates any headline: 0 raw premise overlaps across SDOH-NLI splits;
  the 8 normalised test twins of train premises are 7 UNKNOWN + 1 CURRENT that both arms trained on;
  no twin with a conflicting label. CV folds are built identically across H8/H9/H13/compare scripts,
  and fold predictions are mapped by index correctly.
- Wright splits are patient-disjoint (3,000 patients, 0 shared), with negligible copy-forward text
  shared across splits.
- H12 trains only on SDOH train; the locked test is gated behind `--confirm-locked-test`.

## Confirmed corrections (and what we changed)

| # | Finding | Correction |
|---|---|---|
| 1 | The headline TF-IDF baseline (0.779) is trained on more rows (it includes no-evidence premises) than Flan. | Like-for-like TF-IDF = **0.791**; Flan-T5-base Δ = **+0.077 [+0.016, +0.142]**, p≤0 = 0.006, 5/5 folds. README and findings now quote this. |
| 2 | The locked-test gap (0.877 vs 0.616) rests on 15 smoking-labelled premises; ~64% of it comes from 2 NEVER items, and validation shows no difference (+0.015, CI includes 0). | Now reported as fragile, with the item counts, and not as the effect size. |
| 3 | SDOH-NLI numbers are native status (no one-year rule) but sat under the pipeline heading. | Labelled "status classifier, native labels". |
| 4 | About half the CV gap comes from NEVER, whose SDOH-NLI annotation boundary is inconsistent. | New sensitivity check: with NEVER+UNKNOWN merged (3-class), Flan still wins, **0.934 vs 0.878, Δ +0.057 [+0.012, +0.104]**, p≤0 = 0.006 (`h12-final-confirmation/results/audit_sensitivity_cv.json`). |
| 5 | The cascade (40%) was exploratory and its operating point was chosen after seeing CV. | README now marks the cascade as exploratory and a suggested setting, not a confirmed one. |
| 6 | The Wright 0.972 figure is the dev split, reused across ~5 experiments, not a held-out test. | Relabelled "Wright dev"; the Wright test split will be opened once for the final pipeline (H15). |

Minor (logged, not yet fixed): normalised near-duplicate premises (punctuation, he/she) across
SDOH splits and folds, all UNKNOWN except one; per-item test predictions were not saved in H12.
