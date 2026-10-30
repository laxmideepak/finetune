# H8-ENCODER — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY

## Why

H7: bag-of-words saturates at ≈ 0.79 macro-F1 (5-fold CV) on human-written SDOH-NLI; the
errors need language understanding (lifetime denial, product type, third parties). The
Hugging Face Hub is blocked here, but **SciBERT** (allenai/scibert_scivocab_uncased,
110M, pretrained on 1.14M scientific papers, mostly biomedical) is reachable inside the
scispaCy `en_core_sci_scibert-0.5.4` package (AWS S3). It was unpacked to a standard
Hugging Face folder (0 missing / 0 unexpected weights). Caveat: scispaCy further trained
these weights for biomedical parsing/NER, so this is "SciBERT + scispaCy training".

Because H6 reduced the input to evidence sentences (short), CPU fine-tuning is feasible:
measured 2.7 s per training step (batch 16, 64 tokens, 4 threads).

## Hypothesis

**H8-ENCODER:** a pretrained biomedical encoder fine-tuned on evidence sentences beats the
TF-IDF SVM on human-written text, mainly on NEVER / denial cases, because it represents
negation scope and lifetime qualifiers instead of surface n-grams.

## Design (paired against TF-IDF on identical folds)

- Data: SDOH-NLI train (n = 570, mapped to native 4-class as in H7), StratifiedKFold(5,
  shuffle, random_state=0) — the first repeat of the H7 exploratory CV.
- Input: evidence sentences (extractor retrieval-v2, `src/evidence.py`, context 0); no
  evidence ⇒ UNKNOWN without calling the model.
- Arms per fold: TF-IDF SVM (S), SciBERT (S), TF-IDF SVM (W+S), SciBERT (W+S), where W =
  Wright train evidence sentences with native labels.
- SciBERT recipe (fixed, no tuning): max_len 128 (truncation counted and reported), AdamW
  lr 3e-5, weight decay 0.01, linear warm-up 10%, batch 16, epochs 4 (S) / 3 (W+S),
  class-weighted cross-entropy (inverse frequency, same spirit as `class_weight="balanced"`),
  seed 42, last-epoch model (no early stopping — there is no separate dev in a fold).
- SDOH-NLI validation and test stay untouched by this experiment.

## Predictions

1. SciBERT (S) mean CV macro-F1 ≥ 0.85 vs TF-IDF (S) on the same folds (≈ 0.79).
2. SciBERT improves NEVER F1 by ≥ 0.15 over TF-IDF (S).
3. Unlike TF-IDF, SciBERT benefits from adding Wright: SciBERT (W+S) ≥ SciBERT (S) + 1 pt.
4. Inference throughput on CPU for evidence sentences ≥ 20 sentences/s (batch 32).

## Refutation

SciBERT (S) ≤ TF-IDF (S) + 2 points mean CV — a pretrained scientific encoder adds little on
~460 training examples, pointing to data, not model capacity, as the constraint.

## Outputs

Per-fold predictions and metrics → `experiments/h8-pretrained-encoder-cpu/results/`;
wall-clock training time per fold; CPU inference throughput.
