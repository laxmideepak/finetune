# H8-ENCODER — analysis (interim: enc-WS has fold 0 only; folds 1–4 run last in the queue)

**Verdict on the primary arm: REFUTED.** With the fixed generic recipe (4 epochs, lr 3e-5,
~48 updates on ~190 evidence sentences per fold), SciBERT is worse than TF-IDF on every fold.

| Arm | fold 0 | 1 | 2 | 3 | 4 | mean | NEVER F1 (mean) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| TF-IDF (S) | 0.752 | 0.785 | 0.758 | 0.730 | 0.855 | **0.776** | 0.48 |
| SciBERT (S) | 0.672 | 0.671 | 0.678 | 0.579 | 0.762 | 0.673 | 0.06 |
| TF-IDF (W+S) | 0.706 | 0.723 | 0.773 | 0.784 | 0.885 | 0.774 | 0.42 |
| SciBERT (W+S) | 0.704 | — | — | — | — | (1 fold) | 0.00 |

Predictions: (1) SciBERT ≥ 0.85 ❌ 0.673; (2) NEVER +0.15 ❌ (−0.41); (3) W+S helps SciBERT —
pending; (4) CPU inference ≥ 20 sentences/s — see fold JSONs (`infer_sentences_per_s`).

Reading: SciBERT almost never predicts NEVER (19 NEVER premises in all of train, ~15 per
training fold). With ~48 optimizer updates the classifier head has not learned the minority
classes — the failure mode Mosbach et al. (ICLR 2021) attribute to too few iterations.
This was anticipated in H8b (locked at 749c2ab before folds 1–4 were seen), which reruns
SciBERT with 20 epochs and 3 seeds. Mixed W+S training also costs ~38 min per fold on CPU.
