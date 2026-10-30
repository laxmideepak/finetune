# Zero-DUA smoke wave — results (2026-10-29)

Scope: design contract v0.9.1 §14 (E0–E4), run in a cloud Linux container
(4 vCPU, 15 GB RAM, **no GPU**, **Hugging Face Hub blocked by network policy**).
Data: Wright synthetic LCS benchmark (`github.com/vclic/smokingbenchmark`,
3,000 notes, CC-BY-4.0) — pipeline-comparability set only, **never gold**
(LLM-adjudicated references; §9-4.3).

| Exp | Status here | Why |
|---|---|---|
| E0 retrieval sanity | ✅ run (on Wright, not SDOH-NLI) | SDOH-NLI lives on the HF Hub (blocked); Wright notes are a stricter test anyway (real note layouts, shorthand, distractors) |
| E1 encoder-stack smoke | ⚠️ partial | Pinned stack installs and passes `check_env.py` + `smoke_train.py` on Linux/CPU; BioClinical-ModernBERT weights not downloadable (Hub blocked); MPS not available here |
| E2 end-to-end dry run | ⚠️ partial | Triggers → crops → classifier → note label → metrics ran end-to-end with arm A as the classifier; the C-config fine-tune needs the real weights + a GPU |
| E3 arm-A floor | ✅ run | TF-IDF word+char → LR / LinearSVM |
| E4 cost baseline | ⚠️ CPU only | ModernBERT-base-*shaped* model with random weights (speed only, no accuracy); GPU numbers still needed |

## Setup

- Splits: patient-disjoint 60/20/20 (1,800/600/600), stratified by condition × native label, seed 20261029. **Only dev is scored; test stays locked.**
- Label views:
  - *native* — Wright's labels (any documented quit ⇒ FORMER).
  - *one-year* — our `smoking-status-v1`, derived mechanically from the answer key
    (`quit_date` / `quit_year_{min,max}` vs `note_date`) through
    `label_policy.status_from_cessation`. If the possible quit window straddles the
    365-day line, or no quit time is documented ⇒ `UNKNOWN(timing-unresolved)`.
- Native → one-year mapping over all 3,000 notes: of 1,171 native FORMER, **958 stay FORMER, 24 become CURRENT (quit < 1 year), 189 become UNKNOWN (timing-unresolved)**. Final one-year counts: FORMER 958 / CURRENT 900 / NEVER 703 / UNKNOWN 439.
- Retrieval tokenizer: offline whitespace tokenizer (Hub blocked) ⇒ crop "tokens" below are **words**.

Reproduce:

```bash
python scripts/run_wright_smoke.py --wright-dir <smokingbenchmark clone> --out experiments/E0_E3_wright_retrieval-v2
python scripts/build_shape_model.py --wright-dir <clone> --splits experiments/E0_E3_wright_retrieval-v2/labels_and_splits.csv --out runs/modernbert-base-shape
python src/benchmark_encoder.py --model runs/modernbert-base-shape/model --csv runs/modernbert-base-shape/dev_crops_le512.csv --device cpu --dtype fp32 --batch 8 --passes 2
```

## E0 — retrieval sanity

**Finding 1 (fixed): the v1 lexicon missed common shorthand.** 27 of 2,750 known-status
notes (1.0%; 3.0% of the template "basic" condition) produced no crop at all. The misses
were `Tob: current`, `+tob`, `Former tob use`, `No tob hx`, `lifelong nonsmoker`,
`55 pkyr`, `22 pk-yr`, `40-50 pack-yrs`. `\bsmok` cannot match inside `nonsmoker`, and
`tobacc\w*` does not cover `tob`.

→ `retrieval-v2` adds `(non-?)smok\w*`, `tob`, `pack-y(ea)?rs?`, `pk-y(ea)?rs?`, `pkyrs?`.
All 57 unit tests still pass. Remaining misses: **3/2,750 (0.11%)** —
`nevr smoekr` (misspelling), and two notes where "He quit 38 years ago" / "He quit when
he had his stent" sits in a sentence without any tobacco word (cross-sentence evidence;
cf. SHAC's <2% cross-sentence events).

| Metric (retrieval-v2) | all | basic | messy | complex |
|---|---|---|---|---|
| crops / note (mean) | 1.82 | — | — | — |
| notes with no crop | 128 | — | — | — |
| retrieval FN rate, known-status notes | 0.11% | 0.33% | 0% | 0% |
| crop words p50 / p95 / max | 148 / 353 / 512 | 90 / 186 / 248 | 172 / 368 / 511 | 175 / 380 / 512 |
| crops > 256 words | 839 | 0 | 413 | 426 |
| truncated crops | 0 | 0 | 0 | 0 |
| notes routed to wider context | 23 | 0 | 6 | 17 |
| crop / note size | 0.63 | 0.48 | 0.66 | 0.66 |

Extraction speed: ~340 notes/s on one CPU core (pure Python) — not a cost driver.

**Finding 2: crops are larger than the "±100-token" mental model.** Median 148 words,
p95 ≈ 353, and ~0.63 of the note survives into crops on the LLM-written notes. Triggers
are dense (8–11 per note in messy/complex) and merge into big clusters (≈5 triggers per
crop). This matters for E's max_len 256 plan: 28% of crops exceed 256 *words* already.

**Finding 3: the 512 budget is measured in the wrong tokenizer.** The extractor caps crops
at 512 retrieval-tokenizer units. Re-tokenized with a BPE tokenizer (~1.5 model tokens per
word here), **53 of 1,072 dev crops (5%) exceed 512 model tokens**, and
`benchmark_encoder.py` correctly refuses them. With the real ModernBERT tokenizer the ratio
should be lower (larger vocabulary), but the rule must be: *the retrieval tokenizer must be
the model tokenizer, or the budget must be set in the smallest model's units with margin.*
Re-run E0 with `--tokenizer thomas-sounack/BioClinical-ModernBERT-base` once the Hub is reachable.

## E3 — arm A floor (TF-IDF), dev split (n = 600)

| Labels | Input | Model | macro-F1 | CURRENT R | CURRENT P | basic | messy | complex |
|---|---|---|---|---|---|---|---|---|
| native | crops | SVM | **0.979** | 0.954 | 1.000 | 0.951 | 0.977 | 1.000 |
| native | crops | LR | 0.970 | 0.931 | 0.994 | 0.923 | 0.977 | 1.000 |
| native | full note | SVM | 0.952 | 0.937 | 1.000 | 0.867 | 0.989 | 1.000 |
| native | full note | LR | 0.928 | 0.920 | 0.994 | 0.821 | 0.977 | 0.988 |
| one-year | crops | SVM | **0.888** | 0.923 | 0.994 | 0.833 | 0.912 | 0.914 |
| one-year | crops | LR | 0.880 | 0.917 | 0.994 | 0.817 | 0.901 | 0.914 |
| one-year | full note | SVM | 0.848 | 0.890 | 1.000 | 0.743 | 0.910 | 0.893 |
| one-year | full note | LR | 0.847 | 0.890 | 1.000 | 0.735 | 0.917 | 0.893 |

(retrieval-v1 → v2 lifted native|crops|SVM from 0.967 to 0.979 and native|crops|LR from 0.961 to 0.970; one-year rows moved ≤0.001.)

One-year | crops | SVM confusion (rows = truth, cols = prediction: CUR, FOR, NEV, UNK):

```
CURRENT  167  11   1   2
FORMER     0 189   1   2
NEVER      0   1 140   0
UNKNOWN    1  31   3  51
```

**Finding 4: crops beat full notes for the floor arm too** (+2.7 to +4.2 macro-F1 points,
largest on short template notes). Same direction as the KOTI evidence-in-window result.

**Finding 5: the floor is very high on Wright's native labels (0.98).** Synthetic notes
spell status out in a few fixed phrasings, so bag-of-n-grams solves it. This is the
predicted "A ≈ C" danger zone: Wright native labels cannot separate the neural arms.
Do not use them to rank arms; keep them as a pipeline check only.

**Finding 6: the one-year rule is where the difficulty is.** Under our policy the same
model drops ~9 points. The errors are almost all date arithmetic: **30 of 36**
`UNKNOWN(timing-unresolved)` notes are predicted FORMER (no-mention UNKNOWN: 48/50
correct), and **all 6** recent quitters in dev (quit < 1 year ⇒ CURRENT) are predicted
FORMER; the other 5 CURRENT→FORMER errors are native current smokers. A bag of words
cannot compare a quit date with the note date. Two design consequences:
1. **The note's reference date must be in the model input** (e.g. a `[NOTE DATE: …]`
   prefix). Crops often do not contain it, and no arm can apply the one-year rule without it.
2. The one-year slice is the right stress test for arms C/D/E/G, and the most likely place
   where a small generative arm or a rule-based date resolver earns its cost. Register as an
   ablation: crop + date prefix vs crop + rule-based quit-date resolver.

## E4 — cost baseline (CPU only, provisional)

ModernBERT-base-shaped classifier (149.6M parameters, random weights, local BPE tokenizer),
1,019 dev crops ≤ 512 model tokens, fp32, 4 vCPU, warm, tokenization + forward only:

| batch | snippets / s | non-pad tokens / s | batch latency p50 / p95 |
|---|---|---|---|
| 8 | 2.24 | 502 | 3.5 s / 5.5 s |
| 32 | 1.79 | 401 | 19.2 s / 21.7 s |

Batch 32 is *slower* than batch 8 on CPU (more padding per batch, no unpadding gain in
the eager CPU path), so CPU serving should use small batches or length-bucketing.

$/M = 1e6 × H / (3600 × R). At R ≈ 2.2/s on 4 vCPU, a general-purpose 4-vCPU cloud VM at
roughly $0.15–0.20/h gives **≈ $19–25 per million snippets** — CPU fp32 is not a
competitive serving option for the base-size encoder. GPU (T4/L4/A10G, bf16, unpadding)
numbers are still required for H0-ARCH; they need the real weights and a GPU.
Arm A for contrast: ~350 notes/s for featurize + predict on one core (≈ $0.15/M at the same
VM price) — the cost floor any neural arm has to justify.

## What this wave could not do here, and what unblocks it

- **Hugging Face Hub blocked** (`huggingface.co`): needed for BioClinical-ModernBERT /
  ModernBERT / Ettin / Qwen3 weights, their tokenizers, and SDOH-NLI. Unblock: allow the
  host in the cloud environment's network settings, or run E1/E2/E4 on the M4 / HF Jobs.
- **No GPU here**: E1 (MPS) and the GPU half of E4 run on the M4 and a rented GPU.
- `peft`/`trl` not installed (separate env by design, R5); arm G untested.

## Next moves

1. Re-run E0 with the real ModernBERT tokenizer as the retrieval tokenizer; re-check the >512 count.
2. Add the note-date prefix to the crop/input contract (`data_contract.py` `note_date` is already an optional column; make it part of the visible text for one-year-rule training).
3. E2 proper on GPU: arm C (3 epochs, toy) on Wright one-year labels + native labels; report both, rank on one-year.
4. E4 on T4/L4/A10G with the real model.
