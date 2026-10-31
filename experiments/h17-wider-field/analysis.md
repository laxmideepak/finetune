# H17-FIELD — analysis

**Verdict: Flan-T5-base stays first (decision rule: no candidate beats it). Predictions 1–2 hold,
3–4 fail.**

Pooled SDOH-NLI train 5-fold CV (H9 folds, evidence sentences, same terms as Flan-T5-base):

| Candidate | Size | pooled macro-F1 | folds | NEVER F1 | vs Flan-T5-base |
|---|:-:|:-:|---|:-:|---|
| **Flan-T5-base fine-tuned (reference, H9)** | 250M | **0.868** | 0.794 / 0.939 / 0.855 / 0.867 / 0.901 | 0.68 | — |
| T5 v1.1 base, LM-adapted (no instruction tuning) | 250M | 0.786 | 0.752 / 0.718 / 0.831 / 0.843 / 0.803 | 0.53 | **−0.081 [−0.134, −0.035]** |
| TF-IDF (like-for-like reference) | — | 0.791 | | 0.55 | |
| SciBERT, small-data recipe (20 epochs, seed 42) | 110M | 0.702 | 0.690 / 0.649 / 0.773 / 0.612 / 0.747 | 0.14 | — |
| ByT5-small (byte-level) — **stopped after 2/5 folds** | 300M | (0.362, 0.297) | — | ≤ 0.19 | — |

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | Flan-T5-base stays first | ✅ |
| 2 | LM-adapted T5 v1.1 base < Flan-T5-base by ≥ 3 points | ✅ −8.1 [−13.4, −3.5] |
| 3 | ByT5-small ≥ Flan-T5-small (0.795) | ❌ ~0.33 on 2 folds |
| 4 | SciBERT 20 epochs ≥ TF-IDF (0.791) | ❌ 0.702 |

## Reading

1. **Instruction tuning is what makes the small seq2seq model win.** The same 250M architecture
   without Flan instruction tuning scores like bag-of-words (0.786); with it, 0.868. Starting
   from a model that already understands "what is the patient's smoking status? OPTIONS: …"
   matters when there are only ~240 training sentences.
2. **The SciBERT encoder does not catch up with a longer schedule** (0.682 → 0.702): its NEVER
   class stays near zero. A randomly initialised classification head on ~190 sentences is the
   bottleneck, not the number of epochs.
3. **ByT5 under the fixed recipe fails.** Byte sequences are ~5× longer, so the subword-tuned
   recipe (3 epochs, lr 3e-4, max 256 positions ≈ 256 bytes) truncates and under-trains it; this
   is not a verdict on byte models in general.

## Deviation

ByT5-small was stopped after 2 of 5 folds (~1 h per fold on CPU). Both folds were > 40 points
below Flan-T5-base on the same folds, so the remaining folds could not change the decision.
SciBERT seeds 43/44 (optional in the protocol) were not run.

## Field covered so far (all on the same folds)

Flan-T5 small / base / large (fine-tuned and zero-shot, ± policy guard), T5 v1.1 LM-adapted base,
ByT5-small, SciBERT (two recipes), TF-IDF, keyword rules, a 3-model vote and a cascade.
Not testable here: Hugging Face-only models (ModernBERT, BioClinical-ModernBERT, Qwen, Ettin) and
models too large for a lightweight CPU deployment (Flan-T5-XL/XXL, T5 3B/11B).
