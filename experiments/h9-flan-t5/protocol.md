# H9-FLAN — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY

## Why

The goal is the best *small, efficient language model* for 4-class smoking status. Small
decoder LLMs (Qwen3-0.6B etc.) are on the blocked Hugging Face Hub, but Google's
instruction-tuned **Flan-T5** checkpoints are public on GCS (`gs://t5-data`, T5X format).
`scripts/fetch_flan_t5.py` converts them to Hugging Face format (0 missing / 0 unexpected
weights; a translation prompt returns German as a sanity check).

## Arms (same SDOH-NLI train folds and evidence-sentence input as H8)

| Arm | Model | Training |
|---|---|---|
| ZS-small / ZS-base / ZS-large | Flan-T5 77M / 250M / 780M | none (zero-shot) |
| FT-base | Flan-T5-base | fine-tuned per fold on SDOH train (evidence → answer string), 3 epochs, lr 3e-4, AdamW, batch 16, minority classes oversampled, seed 42 |
| FT-small | Flan-T5-small | same recipe |

Prediction = rank classification over the four answer strings ("current smoker",
"former smoker", "never smoked", "unknown") with Flan's OPTIONS prompt; zero-shot arms use
contextual calibration (subtract the scores for the content-free input "N/A"). The prompt
and calibration were fixed on 8 hand-written policy sentences only (Flan-base 6/8, small
4/8); no evaluation data was used. No evidence ⇒ UNKNOWN without calling the model.

Also reported: zero-shot on Wright dev (native labels) for each size; CPU throughput.

## Predictions

1. Zero-shot cannot know our policy edges (denial-only = UNKNOWN, third-party = UNKNOWN):
   ZS-base CV macro-F1 between 0.55 and 0.75, below TF-IDF S (≈ 0.79 mean CV).
2. ZS-large ≥ ZS-base + 5 points (instruction following scales).
3. FT-base ≥ TF-IDF S + 3 points on the same folds, with NEVER F1 ≥ +0.15 over TF-IDF.
4. FT-base ≥ SciBERT (H8, enc-S) on the same folds.

## Refutation

FT-base ≤ TF-IDF S on the same folds — then on ~460 training sentences no tested
pretrained model beats bag-of-words, and the deciding factor is labelled data, not models.
