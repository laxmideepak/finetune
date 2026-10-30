# H6-EVIDENCE — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY · **Data:** Wright synthetic (comparability set, never gold)

## Motivation (from train-split cross-validation only)

After H5, the remaining error is native status classification on short template notes
(`basic`: 94.0% 5-fold CV accuracy on train vs 99.0–99.5% on `messy`/`complex`).
Reading the CV errors shows why: in template notes a whole short section becomes one crop,
so the bag of words mixes the smoking sentence with distractors such as "Retired,
**former** nurse", family history, problem lists and medication lists.

## Hypothesis

**H6-EVIDENCE:** classifying only the *evidence sentences* — the sentences that contain a
used retrieval trigger — instead of the whole merged crop improves native-status accuracy,
because it removes distractor words that sit in the same section but not in the same
sentence as the smoking evidence. Mechanism: same features, smaller and cleaner input.

## Arms (same split and seed as E-wave-1 / H5)

| Arm | Input to TF-IDF + LinearSVM (native labels) | Then |
|---|---|---|
| S-crop (H5 reference) | merged crops | quit-resolver-v1 on crops |
| S-evid | trigger sentences only, in source order | quit-resolver-v1 on crops |
| S-evid+1 | trigger sentences ±1 neighbouring sentence (same section) | quit-resolver-v1 on crops |

Resolver unchanged (quit-resolver-v1, frozen at eec6f03). Retrieval unchanged (retrieval-v2).

## Development discipline

Development and any tuning use 5-fold CV on the train split only. Dev is scored once per
arm after the evidence builder is committed. Test stays locked.

## Predictions

1. Train-CV native accuracy on `basic` rises from 0.940 to ≥ 0.970 with S-evid.
2. Dev one-year macro-F1 (v2 labels) of S-evid + resolver ≥ 0.960 (H5: 0.949), with
   `basic` ≥ 0.93 (H5: 0.898); `messy`/`complex` not worse by more than 1 point.
3. S-evid+1 sits between S-evid and S-crop on `basic` (context re-admits distractors) but
   helps notes whose status sentence has no trigger word.

## Refutation

S-evid ≤ S-crop + 0.5 point on dev macro-F1, or `messy`/`complex` lose > 1 point —
meaning the crop context carries signal that sentence-only input loses.
