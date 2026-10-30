# H6-EVIDENCE — analysis (dev scored once after freezing the builder at the commit before this file)

**Verdict: SUPPORTED (confirmatory), stronger than predicted.**

## Results

Train 5-fold CV, native-status accuracy (development signal):

| Input | all | basic | messy | complex |
|---|:-:|:-:|:-:|:-:|
| S-crop (merged crops) | 0.975 | 0.940 | 0.995 | 0.990 |
| **S-evid (trigger sentences)** | **0.993** | **0.980** | 0.998 | 1.000 |
| S-evid+1 (±1 sentence) | 0.991 | 0.980 | 0.995 | 0.998 |

Dev (n = 600), native status → quit-resolver-v1 on crops → one-year label:

| Arm | native acc | macro-F1 v1 | macro-F1 v2 | CURRENT R / P | basic | messy | complex |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| S-crop + R (H5 reference) | 0.978 | 0.947 | 0.949 | 0.956 / 1.000 | 0.898 | 0.982 | 0.972 |
| **S-evid + R** | **1.000** | **0.971** | **0.973** | **1.000 / 1.000** | **0.949** | **1.000** | 0.972 |
| S-evid+1 + R | 1.000 | 0.971 | 0.973 | 1.000 / 1.000 | 0.949 | 1.000 | 0.972 |

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | basic CV accuracy 0.940 → ≥ 0.970 | ✅ 0.980 |
| 2 | dev v2 macro-F1 ≥ 0.960, basic ≥ 0.93, messy/complex not worse by > 1 pt | ✅ 0.973, basic 0.949, messy +1.8, complex ±0 |
| 3 | S-evid+1 between S-evid and S-crop on basic | ❌ equal to S-evid (same on dev; −0.002 on CV overall) |

## What we learned

1. **Merged crops hurt in template notes.** Smaller, evidence-only input removes words like
   "Retired, *former* nurse" that share a section with the smoking sentence. Native status
   on Wright dev is now 600/600.
2. **All remaining error is the quit-time step** (0.973 = the gold-status + resolver ceiling
   on crops from H5). Reading the full note for the quit time adds ~0.8 points (H5).
3. **Wright is saturated for status classification.** A bag-of-words model gets every dev
   note right, so this dataset can no longer separate TF-IDF from neural encoders or LLMs.
   Any claim about "the best model" now needs harder, human-written data
   (SDOH-NLI premises, then i2b2-2006 / SHAC after DUAs).
4. Caution: synthetic notes repeat a small set of phrasings. A 100% score here is an upper
   bound for this data, not an expectation for real EHR text (risk R7).

## Next

- H7-OOD: test the frozen Wright pipeline zero-shot on an independent, human-written
  corpus (SDOH-NLI smoking premises from MTSamples), then train on it.
- H2-CONTEXT for the resolver: let the quit-time step read FORMER notes' full section.
