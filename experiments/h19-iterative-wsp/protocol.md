# H19-ITERATIVE-WSP — protocol (locked before any analysis or forward pass)

**Date locked:** 2026-11-04 · **Type:** CONFIRMATORY (one final scoring on untouched data)

## Question

Can an LLM editor, reasoning over the weights over many turns with a forward-pass budget — the
"Weight-Space Programming" harness of Takehi et al. (2026), Exp. 8 — make **zero-shot Flan-T5-large
(780M, no task training)** good at 4-class smoking status *without any gradient training*? H18 tested
one fixed closed-form recipe (rejected on every fold); this is the full iterative version.

## Editor and harness (modelled on `lm_understand-logs/exp8_math`)

- **Editor:** Claude (this session), not GPT-6 Astra. Same rules as their Exp. 8.
- **Allowed:** reading any weight tensor and computing on it (numpy/torch on CPU); saving edited
  tensors (`save_edit`); a **forward-pass tool** with a budget of **50 passes**. One pass = the
  (edited and/or original) model scores up to 512 development premises (predictions, per-class
  report, per-item answer scores), or returns activations (encoder/decoder hidden states, MLP
  activations, attention) for up to 512 development texts.
- **Forbidden:** gradient descent or any training (no backward passes, autograd, optimisers, or
  iterative numeric fitting of weights to a loss); running the model outside the forward-pass tool;
  edits to the token embedding, the output (LM head) matrix, or the final norms; edits that only
  multiply a whole tensor by a constant; looking at the evaluation data.
- **Logging:** every forward pass and edit is appended to `experiments/h19-iterative-wsp/log/` with the
  request, the returned summary, and a short plain-English explanation of the step (as their harness
  required "EXPLAIN EVERY STEP").

## Data

- **Development:** SDOH-NLI train premises in H9 folds 1–4 (≈ 456; evidence sentences as input).
- **Evaluation (scored once, after the final edit is frozen and committed):** SDOH-NLI train fold 0
  (114) + SDOH-NLI validation (109) = 223 premises. Locked test is not reopened.

## Comparators (same 223 evaluation premises)

zero-shot Flan-T5-large · zero-shot + rule guard (H10) · **Flan-T5-large + iterative weight edit** ·
Flan-T5-base fine-tuned on the same development premises (H9 recipe).

## Predictions

1. The iterative edit beats unedited zero-shot Flan-T5-large by ≥ 5 points macro-F1 (it can fix the
   denial→NEVER obstacle the model has the language for).
2. It does not reach fine-tuned Flan-T5-base (fine-tuning adds abilities; editing removes obstacles).
3. The final edit changes < 1% of the model's weights.

## Refutation

Edited ≤ unedited + 1 point on the evaluation set: the iterative harness cannot make this edit work
here either.

## Stopping

The run ends when the 50 passes are used or the editor submits early; the final edit is committed
before the evaluation script is run.
