# H8b-RECIPE — protocol (locked before H8 finished; runs regardless of H8's outcome)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY (follow-up)

## Why

H8 fixed a generic fine-tuning recipe (4 epochs, lr 3e-5). Premises without a trigger are
not model inputs, so each SDOH-NLI fold trains on only ~190 evidence sentences: 12 updates
per epoch, ~48 in total. Mosbach, Andriushchenko & Klakow (ICLR 2021, "On the Stability of
Fine-tuning BERT") show that small-data BERT fine-tuning fails mainly from too few
iterations (and Adam without bias correction), and recommend training for many more
updates (they use 20 epochs on small GLUE sets) with warm-up and a small learning rate.
Only fold 0 of H8 had been seen when this was written (SciBERT 0.672 vs TF-IDF 0.752).

## Hypothesis

**H8b:** with a small-data recipe (≈ 20 epochs, ~240 updates), SciBERT fine-tuned on
evidence sentences beats TF-IDF on the same SDOH-NLI folds.

## Design

Identical to H8 (folds, evidence input, class-weighted loss, max_len 128, seed 42,
last-epoch model, TF-IDF reference from H8), except: **epochs 20, lr 2e-5**, warm-up 10%,
AdamW (PyTorch AdamW applies bias correction). SDOH-only training (enc-S). Three seeds
(42, 43, 44) to measure seed variance, as the instability literature requires.

## Predictions

1. Mean over seeds and folds: SciBERT-20ep ≥ TF-IDF S + 3 macro-F1 points.
2. NEVER F1 ≥ TF-IDF + 0.15.
3. Seed standard deviation of fold-mean macro-F1 ≤ 0.03.

## Refutation

SciBERT-20ep ≤ TF-IDF S + 1 point: with ~190 training sentences per fold, a 110M
scientific encoder does not beat bag-of-words; labelled data is the binding constraint.
