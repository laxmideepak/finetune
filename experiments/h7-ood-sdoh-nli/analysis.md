# H7-OOD — analysis

**Verdict: synthetic → human transfer gap CONFIRMED; "Wright + SDOH is best" NOT supported.**

## Data reality

SDOH-NLI (GitHub mirror, CC-BY-4.0), mapped to native 4-class labels per the protocol:

| split | premises | CURRENT | FORMER | NEVER | UNKNOWN (denial / no-mention / other) |
|---|:-:|:-:|:-:|:-:|:-:|
| train | 570 | 59 | 35 | 19 | 457 (120 / 331 / 6) |
| validation | 109 | 12 | 8 | **1** | 88 (21 / 67 / 0) |

The validation split has a single NEVER premise, so its macro-F1 is dominated by one example.
The extractor (retrieval-v2) found evidence for 239/239 smoking-bearing train premises and
42/42 validation premises (no retrieval misses).

**Label-policy mismatches seen in SDOH-NLI train** (kept as released, not corrected): "Cigar
smoker" → current smoker (our policy: cigars do not count); "Grandmother is a heavy smoker"
→ current smoker (third party); "50+ pack-yr cigarette use" → current smoker (no current-time
statement). These cap any model's agreement with SDOH labels below 100%.

## Confirmatory results (SDOH-NLI validation, n = 109, scored once)

| Arm | macro-F1 | smoking-bearing macro-F1 | CURRENT R / P | NEVER F1 |
|---|:-:|:-:|:-:|:-:|
| W→S (Wright-trained, zero-shot) | 0.653 | 0.437 | 0.75 / 1.00 | 0.09 |
| **S→S** (SDOH train) | **0.831** | **0.819** | 0.67 / 1.00 | 0.67 |
| W+S→S | 0.705 | 0.689 | 0.83 / 1.00 | 0.00 |
| Rules (keyword, no training) | 0.666 | 0.629 | 0.75 / 1.00 | 0.00 |
| W+S→Wright dev (regression check) | 1.000 | — | 1.00 / 1.00 | 1.00 |

Paired bootstrap W+S − S: −0.080, 95% CI [−0.253, +0.094] (not significant).

## Exploratory: 5 × 5-fold CV on SDOH-NLI train (n = 570; steadier)

| Arm | macro-F1 (mean ± sd) | CURRENT | FORMER | NEVER | UNKNOWN |
|---|:-:|:-:|:-:|:-:|:-:|
| W→S | 0.599 ± 0.048 | 0.70 | 0.62 | 0.24 | 0.84 |
| **S→S** | **0.793 ± 0.080** | 0.84 | 0.84 | 0.52 | 0.97 |
| W+S→S | 0.779 ± 0.071 | 0.84 | 0.85 | 0.46 | 0.97 |
| Rules* | 0.772 ± 0.066 | 0.71 | 0.88 | 0.56 | 0.95 |

\*The rules were adjusted on SDOH train phrasings before the freeze, so their CV score is
optimistic. W+S beats S in 11/25 folds (mean −0.014): adding Wright neither helps nor hurts.

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | W→S drops ≥ 8 points below Wright (1.000) | ✅ −35 (validation), −40 (CV) |
| 2 | S→S beats W→S | ✅ +18 (validation), +19 (CV) |
| 3 | W+S best on SDOH, ≤ 1 pt cost on Wright | ❌ W+S ≈ S (CV −1.4, 11/25 folds); no Wright cost (1.000) |
| 4 | Rules ≈ W→S | ✅ 0.666 vs 0.653 (validation) |

## What we learned

1. **The synthetic → human gap is large**: a model perfect on Wright scores 0.60–0.65 on
   human-written MTSamples text. Wright cannot stand in for real notes when ranking models.
2. **Bag-of-words saturates near 0.79–0.83 on human text with ~240 smoking examples.** The
   failures are language-understanding cases: lifetime denial ("Denies ever using illicit
   drugs and tobacco" = NEVER), product type (cigars), third parties, and scope of negation.
   NEVER is the weakest class everywhere (F1 0.24–0.56).
3. **Synthetic data does not add value for the bag-of-words model** on human text (W+S ≈ S).
   Whether synthetic data helps a *pretrained* encoder is open (it may, by teaching the label
   policy rather than surface words).
4. **This is the evidence for the GPU/pretrained step**: the remaining errors are exactly what
   pretrained encoders and small LLMs are good at, and SDOH-NLI + Wright give a measurable
   target (CV macro-F1 > 0.79 on SDOH, ≥ 0.97 one-year on Wright).

## Next

- GPU case + ready-to-run notebook: BioClinical-ModernBERT (arm C), ModernBERT (D),
  Ettin-32M (E), Qwen3-0.6B LoRA (G) on SDOH-NLI CV + Wright, same evidence-sentence input,
  same resolver. Success = beat 0.793 SDOH CV macro-F1 without losing Wright.
- Keep SDOH-NLI test locked for the final comparison.
