# H15-WRIGHT-TEST — analysis (locked Wright test split, opened once)

**Verdict: CONFIRMED (all three predictions).** On 600 never-before-scored Wright test notes, both
final pipelines keep the one-year rule working at ≥ 0.96 macro-F1, and they are tied.

| Pipeline | one-year v2 | one-year v1 | CURRENT R / P | native acc. | basic | messy | complex | recent quitters |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Flan-T5-base + quit-resolver-v1 | **0.965** | 0.958 | 0.994 / 0.994 | 0.998 | 0.964 | 0.986 | 0.947 | 5/6 |
| TF-IDF + quit-resolver-v1 | **0.967** | 0.959 | 0.994 / 1.000 | 0.998 | 0.964 | 0.986 | 0.951 | 5/6 |

(dev, for reference: 0.972 / 0.973.) Per-note predictions: `results/h15_per_note.json`.

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | both ≥ 0.95 (v2) | ✅ 0.965 / 0.967 |
| 2 | within 1.5 points of each other | ✅ 0.1 points apart |
| 3 | CURRENT recall ≥ 0.98 | ✅ 0.994 both |

## Reading

- The dev → test drop is small (0.972 → 0.965), so the H5/H6 design choices tuned on Wright dev
  did not overfit it.
- Status is essentially solved on synthetic notes by either classifier (599/600); the remaining
  errors are in the quit-date step (FORMER vs UNKNOWN timing, one recent quitter missed) and are
  shared by both pipelines — they come from quit-resolver-v1, not the classifier.
- Together with H12 (human-written text: Flan clearly better) this completes the evidence for the
  recommendation: Flan-T5-base wins where language is hard and loses nothing where it is easy.

Run note: a container restart killed the first H15 run during training, before any test note was
scored; it was relaunched from scratch with the same frozen recipe.
