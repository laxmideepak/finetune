# H12-FINAL — analysis

**Verdict: CONFIRMED. Under the pre-registered decision rule, fine-tuned Flan-T5-base becomes
the recommended status model for human-written text, with TF-IDF as the low-cost fallback
(and first stage of the H13 cascade).**

## Results

| Test | Flan-T5-base FT | TF-IDF | Paired Δ [95% CI] |
|---|:-:|:-:|---|
| **SDOH-NLI test (locked, opened once; n = 108)** | **0.877** | 0.616 | **+0.261 [+0.024, +0.470]**, p≤0 = 0.012 |
| ↳ CURRENT recall / precision | 1.00 / 1.00 | 0.90 / 0.75 | |
| ↳ NEVER F1 | 0.67 | 0.00 | |
| SDOH-NLI validation (n = 109) | 0.846 | 0.831 | +0.015 [−0.110, +0.138] |
| Wright dev, full pipeline, one-year v2 (n = 600) | 0.972 | 0.973 | — |
| ↳ native-status accuracy | 0.998 (1 error) | 1.000 | |
| ↳ CURRENT recall / precision | 1.00 / 1.00 | 1.00 / 1.00 | |

Efficiency (4 CPU threads, assumed $0.17/h): Flan-T5-base 9.3 evidence sentences/s ≈ $5.08 per
million sentences (248M parameters). Training: ~3 min on SDOH-NLI; ~3.7 h on Wright+SDOH (2,944
oversampled rows of ~60-word evidence, 552 steps at ~24 s/step on CPU).

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | SDOH test Flan ≥ TF-IDF + 3; NEVER/CURRENT F1 not lower | ✅ +26.1; CURRENT 1.00 vs 0.82, NEVER 0.67 vs 0.00 |
| 2 | SDOH validation Flan ≥ TF-IDF (0.831) | ✅ 0.846 (CI includes 0) |
| 3 | Wright pipeline ≥ 0.96, ≤ 1.5 pt regression | ✅ 0.972 (−0.2) |
| 4 | ≥ 10 sentences/s and < $10/M | ◐ 9.3/s (just under); $5.08/M ✓ |

## Caveats

- SDOH-NLI test has only 15 smoking-labelled premises (10 CURRENT, 3 FORMER, 2 NEVER); the gap's
  size is uncertain even though its sign is clear.
- SDOH-NLI labels differ from our policy at the edges (cigars, third parties, pack-years-only).
- Run history: the first H12 process was OOM-killed in the Wright stage after the SDOH stages had
  printed (results recovered from the log into `h12_sdoh.json`); the Wright stage was rerun alone
  with allocator settings that do not change the computation (`MALLOC_ARENA_MAX=2`,
  `MALLOC_TRIM_THRESHOLD_`), same recipe and seed.
