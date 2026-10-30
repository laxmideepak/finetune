# H17-FIELD — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY (widen the candidate set)

## Why

The recommendation (fine-tuned Flan-T5-base) was compared against a small field. Everything else
reachable here and light enough for CPU is added now, on the same terms:

| Candidate | Size | What it tests | Source |
|---|---|---|---|
| T5 v1.1 base, LM-adapted (`t5_1_1_lm100k_base`) | 250M | Same architecture as Flan-T5-base **without instruction tuning** — does instruction tuning matter? | GCS `t5-data` T5X |
| ByT5-small (`byt5_small`) | 300M | Byte-level input — robust to clinical typos and shorthand ("smoekr", "pkyr")? | GCS `t5-data` T5X |
| SciBERT, small-data recipe (= H8b, locked at 749c2ab) | 110M | A fairly trained encoder with a classification head | scispaCy (S3) |

Excluded as not lightweight on CPU: Flan-T5-XL/XXL, T5 3B/11B, mT5 (250k vocabulary), XL ByT5.

## Design (identical to H9 unless stated)

- SDOH-NLI train, StratifiedKFold(5, shuffle, random_state=0) — the H9 folds — so every candidate
  is paired with Flan-T5-base (H9) and like-for-like TF-IDF on the same premises.
- Evidence-sentence input; no evidence ⇒ UNKNOWN.
- Seq2seq candidates: H9 recipe exactly (OPTIONS prompt, answer strings, AdamW 3e-4, 3 epochs,
  batch 16, oversampling, seed 42, rank classification).
- SciBERT: H8b recipe (20 epochs, lr 2e-5, class-weighted CE, seed 42; seeds 43/44 only if time).
- Conversion: `scripts/fetch_flan_t5.py` generalised to any T5 v1.1-layout T5X checkpoint with
  dimensions inferred from the arrays; a translation/sanity prompt is not meaningful for
  non-instruction models, so conversion is checked by 0 missing / 0 unexpected weights.

## Predictions

1. Flan-T5-base stays first: every new candidate's pooled macro-F1 ≤ Flan-T5-base (0.868).
2. LM-adapted T5 v1.1 base < Flan-T5-base by ≥ 3 points (instruction tuning helps at ~240 examples).
3. ByT5-small ≥ Flan-T5-small (0.795) — byte input helps with shorthand — but < Flan-T5-base.
4. SciBERT (20 epochs) ≥ TF-IDF like-for-like (0.791) — a fair recipe fixes H8's under-training.

## Decision rule

If any candidate beats Flan-T5-base with a paired-bootstrap 95% CI above 0, it becomes the
recommendation (subject to cost). Otherwise Flan-T5-base stays.
