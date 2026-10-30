<div align="center">

# 🚭 finetune

### Lightweight Smoking-Status Classification from EHR Snippets

*How small can an open-source model get while still reading smoking status correctly —<br/>at a million snippets, for the lowest cost per million?*

<br/>

![Status](https://img.shields.io/badge/status-E--wave--1%20done-2ea44f?style=for-the-badge)
![Design](https://img.shields.io/badge/design%20contract-v0.9.1-1f6feb?style=for-the-badge)
![Retrieval](https://img.shields.io/badge/retrieval-v2-8250df?style=for-the-badge)
![Tests](https://img.shields.io/badge/tests-57%20passing-2ea44f?style=for-the-badge)

![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/torch-2.8.0-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/transformers-4.57.0-FFD21E?logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0-F7931E?logo=scikitlearn&logoColor=white)
![PHI](https://img.shields.io/badge/raw%20ePHI%20to%20cloud-never-critical)

<br/>

[**The problem**](#-the-problem) ·
[**Pipeline**](#-pipeline-at-a-glance) ·
[**Model arms**](#-model-arms) ·
[**Latest results**](#-latest-results--e-wave-1) ·
[**Evidence**](#-what-the-evidence-says) ·
[**Quickstart**](#-quickstart) ·
[**Repo map**](#-repository-map) ·
[**Roadmap**](#-roadmap)

</div>

---

## 🎯 The problem

A production EHR pipeline pre-filters clinical notes with regex (`smok`, `tobacco`, `tob`, …), cuts out a sentence-aligned window of roughly ±100 tokens around each match, and asks a model for **exactly one label**:

| | Label | Meaning (note-level, at the note's reference time) |
|:-:|---|---|
| 🔴 | **`CURRENT`** | Explicit ongoing combustible-tobacco use. **One-year rule: quit < 1 year ago ⇒ `CURRENT`.** |
| 🟠 | **`FORMER`** | Explicit prior use with cessation ≥ 1 year before the note. |
| 🟢 | **`NEVER`** | Explicit lifetime never-use. *Missing evidence is never read as NEVER.* |
| ⚪ | **`UNKNOWN`** | No mention · retrieval-negative · timing unresolved · family-only · denial-only · conflicting. The reason is stored separately. |

> [!IMPORTANT]
> **The deciding number is cost per million snippets.** Latency is secondary. Clinical reliability is not: `CURRENT` recall/precision and the `UNKNOWN` reasons are first-class metrics.

<details>
<summary><b>Why this is hard</b></summary>

<br/>

- Public labeled data is tiny and behind data-use agreements (~500 notes for the canonical 2006 challenge).
- Clinical notes are 33–54% duplicated text, and smoking history is often copy-forwarded.
- Prompt-based classifiers plateau well below classifier-head fine-tuning.
- Every corpus uses a different label taxonomy, and they disagree exactly at the boundaries.
- The one-year rule needs **date arithmetic** — the note date versus the quit date.

</details>

---

## 🧭 Pipeline at a glance

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#eef4ff','primaryBorderColor':'#1f6feb','lineColor':'#57606a','fontSize':'14px'}}}%%
flowchart LR
    A["🔒 0 · Ingest<br/>privacy + provenance"] --> B["🗂️ 1 · Preprocess<br/>sections + note date"]
    B --> C["🔎 2 · Trigger retrieval<br/>recall-first regex + audit"]
    C --> D["✂️ 3 · Crops<br/>sentence-aligned ±100<br/>dedupe · boilerplate out"]
    D --> E["🏷️ 4 · Labels<br/>4-way + one-year rule"]
    E --> F["🧠 5 · Model arms<br/>A · B · C · D · E · G"]
    F --> G["🧮 6 · Aggregate<br/>crops → note"]
    G --> H["🚦 7 · Abstain<br/>θ → wider context → review"]
    H --> I["📏 8 · Evaluate<br/>patient-disjoint · locked tests"]
    I --> J["💵 9 · Serve & cost<br/>$ / M decided here"]

    classDef hot fill:#fff1e5,stroke:#d1242f,color:#24292f;
    class C,J hot;
```

Stage contracts, boundary tests B1–B11 and risks R1–R7 → **[`docs/design/pipeline-design.md`](docs/design/pipeline-design.md)** · more diagrams → **[`docs/design/pipeline-diagram.md`](docs/design/pipeline-diagram.md)**

---

## 🧠 Model arms

| Arm | Model | Role | Configuration |
|:-:|---|---|---|
| **C** ⭐ | BioClinical-ModernBERT-base + 4-way head | **Primary** | BF16 · 3 seeds · eff. batch 32 · ≤ 5 epochs · max_len 512 · refuses truncation |
| **D** | ModernBERT-base | Ablation | Isolates the value of clinical pre-training |
| **E** | Ettin-32M | Cost upset | Crops at max_len 256 |
| **A** | TF-IDF word + char n-grams → LR / LinearSVM | Floor guard | If A ≈ C, the neural gain is noise — redesign the crops |
| **B** | MiniLM frozen 384-d + LR head | Fallback only | Deprioritised after the few-shot evidence |
| **G** | Qwen3-0.6B SFT LoRA | Challenger | Source-order crops; instruction as prefix only |

> [!NOTE]
> **Deployment rule (pre-registered):** the smallest model whose macro-F1 is non-inferior to the best (paired patient-cluster bootstrap, margin 1 point) **without** losing more than 2 points of `CURRENT` recall — reported with its $/M.

---

## 📊 Latest results — E-wave-1

<sub>2026-10-29 · Wright synthetic benchmark (3,000 notes, CC-BY-4.0) · patient-disjoint 60/20/20 · dev split only, test locked · <b>comparability set, never gold</b></sub>

<table>
<tr>
<td width="50%" valign="top">

#### 🔎 E0 — retrieval
| | v1 | **v2** |
|---|:-:|:-:|
| Known-status notes with **no** crop | 27 / 2,750 | **3 / 2,750** |
| Miss rate | 0.98% | **0.11%** |
| Crop words p50 / p95 | 148 / 354 | 148 / 353 |
| Truncated crops | 0 | 0 |

v2 adds the shorthand v1 missed: `Tob:` · `+tob` · `nonsmoker` · `pkyr` · `pk-yr` · `pack-yrs`.

</td>
<td width="50%" valign="top">

#### 🧪 E3 — TF-IDF floor (macro-F1)
| Labels | Crops | Full note |
|---|:-:|:-:|
| Wright native | **0.979** | 0.952 |
| Our one-year rule | **0.888** | 0.848 |

Crops beat full notes in every setting (+2.7 to +4.2 points).

</td>
</tr>
</table>

> [!WARNING]
> **Under the one-year rule the errors are all date arithmetic.** The floor model called **6 of 6** recent quitters `FORMER` (should be `CURRENT`), and **30 of 36** "quit date unclear" notes `FORMER` (should be `UNKNOWN`). ➜ **The note's reference date must be part of the model input.**

| | Finding | Consequence |
|:-:|---|---|
| 1 | Wright native labels are easy (TF-IDF 0.98) | Cannot rank neural arms — use as a pipeline check only |
| 2 | 5% of crops exceed 512 *model* tokens after re-tokenizing | Retrieval tokenizer must equal the model tokenizer |
| 3 | CPU, base-size encoder: **2.2 snippets/s** on 4 vCPU (≈ **$19–25 / M**) | CPU fp32 is not competitive; GPU numbers needed |
| 4 | TF-IDF: ~350 notes/s on one core (≈ $0.15 / M) | The cost floor every neural arm must justify |

Full write-up → **[`docs/results/e-wave-1-results-2026-10-29.md`](docs/results/e-wave-1-results-2026-10-29.md)**

---

## 🔬 What the evidence says

<sub>Every row was checked against the full text or abstract (study queue S-1…S-25 in the design contract).</sub>

| Finding | Source | Design consequence |
|---|---|---|
| Classifier-head full fine-tuning reaches F1 ≈ 0.85–0.90 on 4-class smoking | i2b2-2006 winners · Palmer 2019 | Encoder-first; full fine-tuning is the primary path |
| Prompt-verbalizer classification plateaus ~0.67 micro-F1; few-shot 0.35–0.47 macro-F1 | Landi & Alleva 2312.09469 · KOTI 2025 | Few-shot arms are fallbacks, never primary |
| Evidence-in-window beats tail truncation by +15.8–21.6% F1 | KOTI full text | Trigger-anchored crops |
| Full cross-patient dedup **hurts** smoking F1 (−11.9% / −32.6%); within-note dedup helps (+14.2%) | Landi & Alleva 2312.09469 | Within-note dedup + boilerplate filter only |
| 3–8% same-patient leakage across folds when splits aren't patient-stratified | MSEP, JMIR 2026 | Patient-disjoint splits are mandatory |
| Cross-corpus drop 0.901 → 0.774 (−13 points) | n2c2/UW 2022 | One locked test per corpus, no blended metrics |
| Fine-tuned ClinicalBERT 0.97 vs gpt-oss-20B zero-shot 0.72–0.79 on quantitative fields | SmokeBERT, JCO CCI 2025 | Teacher scope = 4-class status only |
| 19% of current tobacco users are documented **only** in narrative text | Lybarger, JAMIA 2023 | Extraction is worth it; retrieval audit lane |

---

## 🗄️ Datasets & access

| Corpus | Size | Access | Role here |
|---|---|---|---|
| **i2b2/n2c2-2006 Smoking** | ~502 discharge summaries | 🔐 DUA (manual signing in progress) | Primary gold benchmark |
| **SHAC** (n2c2/UW 2022) | 4,405 social-history sections | 🔐 PhysioNet + UW DUA | Tobacco events `{none, current, past}` |
| **Wright synthetic LCS** | 3,000 notes + labels | 🟢 CC-BY-4.0 · [`vclic/smokingbenchmark`](https://github.com/vclic/smokingbenchmark) | E-wave pipeline set + one-year-rule stress test — **never gold** |
| **SDOH-NLI** | 29,335 items | 🟢 CC-BY-4.0 | Format smoke only — not a label source |
| **MIMIC-III** | — | 🔐 Credentialed | Deferred (Phase 3+) |

> [!CAUTION]
> Label taxonomies disagree at the boundaries (e.g. Wright: *any* quit ⇒ FORMER; ours: quit < 1 year ⇒ CURRENT). Each corpus's labels are kept separate; mappings live in the harmonization register (§9-4.3 of the design contract).

---

## 🛡️ Privacy & compliance

- 🔒 **No raw ePHI leaves the approved environment.** `policy_can_egress()` defaults to `false`; cloud calls need an executed BAA or de-identified/public data.
- ☁️ **Teacher route (ratified):** cloud APIs on public/de-identified corpora only; a privately hosted gpt-oss-20B is a Phase-3 decision.
- 🏷️ **Teacher scope = 4-class status only.**
- ✂️ Cropping is **not** de-identification; HHS Expert Determination or Safe Harbor applies.
- 🚫 No automatic uploads to the Hugging Face Hub (`push_to_hub=False` everywhere).

---

## 💵 Cost model

```text
$ / M snippets = 1,000,000 × H / (3600 × R)
                 H = hardware $/hour     R = snippets/second (warm, incl. tokenization)
```

| Reference point | $ / M |
|---|--:|
| Cloud frontier LLMs, full notes (published) | ≈ 120 – 6,610 |
| Base-size encoder, CPU fp32, 4 vCPU (E4, measured) | ≈ 19 – 25 |
| TF-IDF floor, one CPU core (E3, measured) | ≈ 0.15 |
| Base-size encoder on T4 / L4 / A10G | ⏳ pending |

Fine-tuning budget: **$50–200 total**.

---

## 🚀 Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt          # encoder stack + pytest
# generative arm lives in a separate env:    pip install -r requirements/generative.txt
```

```bash
python scripts/check_env.py                  # verify pins, devices, label policy, data contract
python -m pytest -q tests                    # 57 unit tests
python scripts/smoke_train.py                # offline tiny-model train → save → reload → predict
python src/extract_windows.py --demo         # see the crop extractor on a built-in note
```

<details>
<summary><b>Re-run E-wave-1 on the Wright benchmark</b></summary>

<br/>

```bash
git clone --depth 1 https://github.com/vclic/smokingbenchmark ../smokingbenchmark

# E0 retrieval sanity + E3 TF-IDF floor
python scripts/run_wright_smoke.py --wright-dir ../smokingbenchmark \
    --out experiments/E0_E3_wright_retrieval-v2

# E4 speed baseline (random-weight, ModernBERT-base-shaped; speed only)
python scripts/build_shape_model.py --wright-dir ../smokingbenchmark \
    --splits experiments/E0_E3_wright_retrieval-v2/labels_and_splits.csv \
    --out runs/modernbert-base-shape
python src/benchmark_encoder.py --model runs/modernbert-base-shape/model \
    --csv runs/modernbert-base-shape/dev_crops_le512.csv --device cpu --batch 8 --passes 2
```

</details>

---

## 📁 Repository map

```text
finetune/
├── 📄 README.md
├── 📦 requirements.txt            → requirements/core.txt
├── 📦 requirements/
│   ├── core.txt                  # pinned encoder / inference stack
│   ├── dev.txt                   # core + pytest
│   └── generative.txt            # core + peft + trl (separate env)
│
├── 🧠 src/                        # library + entry points
│   ├── label_policy.py           # ★ single source of truth for labels & the one-year rule
│   ├── data_contract.py          # shared dataset contract for every entry point
│   ├── extract_windows.py        # Stage 3 crop extractor (retrieval-v2)
│   ├── train_encoder.py          # arms C / D / E
│   ├── train_generative.py       # arm G (Qwen3 LoRA)
│   ├── evaluate_generative.py
│   ├── benchmark_encoder.py      # warm throughput → $/M
│   ├── compare_models.py         # paired patient-cluster bootstrap
│   ├── metrics.py
│   └── provenance.py
│
├── 🛠️ scripts/
│   ├── check_env.py              # environment readiness check
│   ├── smoke_train.py            # offline end-to-end smoke
│   ├── run_wright_smoke.py       # E0 + E3 on the Wright benchmark
│   └── build_shape_model.py      # E4 offline speed model
│
├── ✅ tests/                      # 57 unit tests + synthetic fixture
│
├── 📚 docs/
│   ├── MASTER_CONTEXT.md         # ★ start here: user, scope, phases, hypotheses
│   ├── design/                   # pipeline-design.md (contract v0.9.1) · pipeline-diagram.md
│   ├── results/                  # experiment write-ups (E-wave-1)
│   ├── literature/               # survey.md — verified literature + pricing
│   ├── external-reports/         # two external expert reports (verbatim)
│   └── tracking/                 # findings.md · research-log.md · research-state.yaml
│
└── 🔒 data/ · experiments/ · paper/ · to_human/   # git-ignored workspaces (never committed)
```

**Read order for newcomers:** [`MASTER_CONTEXT`](docs/MASTER_CONTEXT.md) → [`pipeline-design`](docs/design/pipeline-design.md) → [`pipeline-diagram`](docs/design/pipeline-diagram.md) → [`E-wave-1 results`](docs/results/e-wave-1-results-2026-10-29.md) → [`findings`](docs/tracking/findings.md) → [`research-log`](docs/tracking/research-log.md)

---

## 🗺️ Roadmap

| Phase | Content | Status |
|:-:|---|:-:|
| 1 | Research: datasets, models, methods, architecture, costs | ✅ design v0.9.1 |
| E | **Zero-DUA smokes E0–E4** on the Wright benchmark | 🟡 E0 · E3 done · E1 · E2 · E4 need model weights + GPU |
| 2 | DUA execution (i2b2 + SHAC); dataset kit + labeling policy | ⏳ user-side signing |
| 3 | Fine-tune arms A–G on rented GPU ($50–200); comparison report | ⏳ after Phase 2 |
| 4 | Winner deployment recipe + INT8 slot + real-environment validation | ⏳ |
| 5 | Real-EHR adaptation (self-training / active learning) — PHI never leaves | ⏳ |

**Next moves**
- [ ] Re-run E0 with the real ModernBERT tokenizer as the retrieval tokenizer
- [ ] Put the note's reference date into the visible model input
- [ ] E2: arm C toy fine-tune on Wright (one-year + native labels) on GPU
- [ ] E4: throughput on T4 / L4 / A10G with the real model

---

## 📐 Evaluation contract

- Patient-disjoint, hash-deduped, template-cluster-checked splits; **one locked test set per corpus**, untouched by any optimization.
- **macro-F1 primary** · per-class P/R/F1 · `CURRENT` recall and precision separately · ECE + risk-coverage · `UNKNOWN`-reason breakdown · retrieval false-negative rate.
- Patient-cluster bootstrap CIs; paired comparisons.
- A teacher model's output is **never** ground truth in evaluation.

---

<div align="center">

**License:** TBD — third-party datasets keep their own licenses and DUAs; nothing gated is redistributed here.

**Cite:** *laxmideepak/finetune — Lightweight Smoking-Status Classification from EHR Snippets (2026)*, https://github.com/laxmideepak/finetune

Maintained by [@laxmideepak](https://github.com/laxmideepak) · issues and design critiques welcome — every claim here should come with a source or an experiment ID.

</div>
