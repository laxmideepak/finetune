# H13-CASCADE — analysis (exploratory, SDOH-NLI train 5-fold CV, out-of-fold predictions)

**Verdict: SUPPORTED (exploratory).** Routing the lowest-margin TF-IDF decisions to the
fine-tuned Flan-T5-base keeps most of Flan's accuracy at a fraction of its CPU cost.

| Flan call share | pooled macro-F1 | NEVER F1 | share of Flan gain kept | ≈ $/M sentences |
|:-:|:-:|:-:|:-:|:-:|
| 0% (TF-IDF only) | 0.791 | 0.55 | 0% | 0.04 |
| 10% | 0.820 | 0.59 | 37% | 0.55 |
| 20% | 0.840 | 0.62 | 64% | 1.06 |
| **30%** | **0.858** | 0.67 | **87%** | **1.56** |
| **40%** | **0.865** | 0.70 | **97%** | **2.07** |
| 50% | 0.863 | 0.68 | 94% | 2.58 |
| 100% (Flan only) | 0.868 | 0.68 | 100% | 5.12 |

(TF-IDF here is trained only on evidence-bearing rows, so its pooled score is 0.791 rather than
H8's 0.779; Flan predictions are H9's out-of-fold outputs.) Costs use H12's measured Flan
throughput (9.3 sentences/s, 4 threads) at an assumed $0.17/h.

Prediction (≥ 80% of the gain at ≤ 35% of Flan cost at b = 30%) ✅: 87% at 30% cost.

Reading: the SVM's margin is a good uncertainty signal — its low-margin sentences are where
it is wrong and Flan is right. Recommended operating point: **Flan on ~40% of evidence
sentences** (97% of the gain, ~$2 per million). Needs confirmation on new labelled data (the
SDOH test was already used once and is not reopened).
