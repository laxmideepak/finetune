# MASTER CONTEXT — Lightweight LLM for Smoking Status Classification from EHR Notes

> Single source of truth for this project. Read this file first in every new session.
> Created: 2026-10-28. Status: research plan approved; expert researcher report received and integrated (§9). Phase 1 (research) starting.

---

## 1. The User and Their Pipeline

- **Who**: ML Engineer (Deepak) building production EHR NLP. Limited hands-on LLM fine-tuning experience; needs both rigorous research and a practical implementation roadmap.
- **Existing pipeline** (their real production system):
  1. Clinical notes prefiltered by regex keywords (`smok`, `tobacco`).
  2. For every keyword match: extract ~±100 tokens of surrounding context (snippet) — done to minimize input length and inference cost.
  3. Snippets → language model.
  4. Model must output exactly one of 4 classes: **CURRENT / FORMER / NEVER / UNKNOWN**.
- **Scale objective**: millions of clinical notes; classifier must be highly accurate, extremely lightweight, fast, cost-efficient, clinically reliable.

## 2. Confirmed Scoping Decisions (answered 2026-10-28)

| Question | Answer |
|---|---|
| Labeled data | **None.** No dataset in hand. Considering i2b2; user unsure whether a DUA is required (plan says yes — verify exact portal/terms). |
| Teacher model (distillation / edge-case labeling) | **Undecided** — research question with a decision matrix: cloud LLM on de-identified public data vs. privately-hosted open teacher for real-PHI snippets. |
| Deployment target at scale | Flexible — "can buy compute going ahead if we need" → cost-per-million-snippets is the deciding metric, latency secondary. |
| Experimental compute budget | **$50–200** for the fine-tuning comparison arms; free/cheap tiers where possible (M4 local used for dev-prototype first). |

## 3. Environment Facts (verified first-hand, 2026-10-28)

- Local dev machine: **Apple M4, 16 GB unified memory, MPS backend** (this is where OpenCode runs; the user's production compute is elsewhere/none decided).
- Stack before this project: `torch 2.8.0` (MPS ok), `transformers 4.57.0`. **Nota bene:** `peft`, `accelerate`, `datasets` were installed and then uninstalled during cleanup of a previous (aborted) project; the `transformers` install pre-existed and remains
  → when starting Phase 2 experiments, **reinstall `pip install peft accelerate datasets` locally or run experiments on rented GPU.**
- Prior verified (dot facts) from earlier (deleted) sandbox work:
  - Local HF cache pre-existed and still contains: `Qwen2.5-0.5B-Instruct` and `all-MiniLM-L6-v2`. (The latter is relevant as a sentence-embedding base; the former is a candidate small-LLM classifier.)
  - **MPS OOM lesson**: a 512-token × batch-8 fp32 LoRA training loop on SmolLM2-360M + 16 GB machine OOM'd at ~20 GB allocated. Mitigations if prototyping locally: bf16 compute dtype, gradient checkpointing, batch 2–4 + grad accumulation, max_len 256–384, or run experiments on rented GPU instead.

## 4. Research Objectives (verbatim scope from user, key items)

- **A. Best small open-source models** for this task (≤3B preference, focus ≤1B). Compare small instruction-tuned LLMs vs. clinical/biomedical encoders (ClinicalBERT, PubMedBERT, ModernBERT clinical variants, DeBERTa-v3-small) vs. quantized/CPU-suitable. Explicit requirement: **do not assume generative LLM > encoder**. Scorecard dimensions: classification-accuracy potential, params, memory, throughput/inference speed, license, fine-tuning complexity, deployment cost.
- **B. Available smoking datasets**: i2b2-2006 Smoking Challenge (canonical, 4-class mapped, DUA-gated, tiny — ~501 labelled discharge reports, verify size/access path), PhysioNet/MIMIC (derived smoking labels/notes, DUA), HF sweeps, academic data. Report size, label definitions, annotation quality, access requirements, license, suitability to the 4-class problem. Also design how to construct training snippets from regex-extracted short clinical text.
- **C. Fine-tuning & optimization techniques**: full FT vs LoRA/QLoRA, PEFT/TRL/SFT, knowledge distillation from larger teacher, DSPy prompt optimization (and teacher labeling), synthetic data/augmentation, weak supervision + active learning, parameter-efficient classifier heads, quantization/inference optimization. Tailor to: narrow 4-class short-snippet task, limited labels, million-scale throughput.
- **D. Dataset creation & clinical edge cases** — labeling policy must explicitly handle: negation ("denies smoking"), historical vs active smoking, never-smoker vs undocumented, family history/third-party mentions, conflicting statements, temporal references/cessation dates, multiple keyword matches per note, ±100-token window adequacy, regex false negatives (lexicon gaps: *cigarette(s), cigar, pipe, cigs, pack-year, packs/day, nicotine, vaping, e-cigarette, chewing tobacco, "quit smoking", "never smoked", …*), cases with **no** "smok/tobacco" keywords. **Hard rule: missing smoking evidence → UNKNOWN, never inferred NEVER.**
- **E. Training infra & tech stack**: compare HF ecosystem (incl. HF Jobs), Colab, Kaggle, Azure ML / Azure GPU VMs, RunPod, Modal, Lambda, and other modern options on GPU needs, cost, DX, data-privacy, PHI compliance. **PHI firewall**: cloud/public-data experiments vs. real-EHR work must be separated; real case reviews/adaptation happen only in the user's approved secure environment; my recommendation is cloud for public-data + experiments only.
- **F. Evaluation design**: 6-arm comparison — (1) zero-shot large LLM, (2) DSPy prompt-optimized, (3) fine-tuned small-LLM, (4) fine-tuned encoder classifier, (5) distilled lightweight model, (6) hybrid regex/rules-only. Metrics: macro-F1, per-class precision/recall, confusion matrices, clinical error analysis, throughput, latency, cost-per-million-snippets. Mandatory: **patient-level data-leakage prevention** and generalization tests to unseen clinical doc styles.

## 5. Approved Project Phases

1. **Phase 1 (now, read-only research)**: verify i2b2 portal/DUA reality + MIMIC terms + full HF/GitHub dataset sweep; build model scorecards; methods review for C. Produce the 8 requested deliverables (top-5 models ranked; dataset w/ links + access; methods comparison; architecture recommendation; beginner step-by-step guide with Python; dataset-size recommendation + experimental plan; GPU requirements & costs incl. training + inference-at-scale; staged roadmap with what to build first).
2. **Phase 2**: (user signs DUA) dataset construction kit + explicit labeling-policy doc (covers item D edge-case rules).
3. **Phase 3**: fine-tune the arms from item F on rented GPU within $50–200 budget; produce comparison report.
4. **Phase 4**: winner deployment recipe, incl. quantized/CPU fallback if warranted. Validate in the user's secure environment.
5. **Phase 5**: real-EHR adaptation plan — self-training/active learning + teacher-policy decision executed inside the private environment (PHI never leaves).

## 6. Working Hypotheses (to confirm or refute via Phase 1–3)

- Fine-tuned **clinical encoder** (ModernBERT / BERT-clinical family) will beat small fine-tuned LLMs on label-efficiency, throughput, and cost for a 4-class short-snippet problem — production candidate.
- Public 4-class labeled data will be **tiny and DUA-gated** → the real contribution is the dataset construction pipeline: weak supervision/labeling functions + (undecided→to-be-decided) teacher labeling + small clinician/manual review.
- Larger-teacher **distillation** = mostly a data augmentation play (hard/edge-case coverage) rather than logit-distillation, given a 4-class softmax head.
- ±100 tokens will be **mostly** enough but insufficient for some cases (structured quit-date columns, negation scopes) → evaluate a small "long-context fallback" lane with a cheap reranker/window-extension policy rather than always widening the window.
- Expecting `UNKNOWN` class + conflict/timeline rules to drive most clinical error cost → invest labeling policy doc in that quadrant.

## 7. Deliverables checklist (user's requested 8 items)

- [ ] 1. Top-5 models ranked for this use case, justified, with HF links
- [ ] 2. Best datasets with direct links + access restrictions
- [ ] 3. Fine-tuning / distillation / prompt-optimization methods comparison
- [ ] 4. Recommended end-to-end architecture (regex → snippet → classifier → post-processing)
- [ ] 5. Beginner-friendly step-by-step implementation guide (Python + configs + eval)
- [ ] 6. Realistic dataset-size recommendation + experimental plan
- [ ] 7. GPU requirements, fine-tuning cost estimate, inference cost at scale
- [ ] 8. Staged roadmap (baseline → dataset prep → fine-tune → optimize → validate → deploy)

## 8. Non-goals / constraints

- Do NOT use unlicensed/leaked copies of i2b2/n2c2 data. DUA-compliant use only.
- Do NOT process real PHI outside the user's approved environment.
- No medical/clinical validation claims without evidence — models are research-grade until tested in the user's environment with clinical oversight.
- English-only clinical notes assumed (confirm with user if multilingual is needed).

## 9. External Researcher Report — Integration (received 2026-10-28)

Verbatim source saved at `docs/external-reports/external-report-2026-10-28.md`. These items
**amend / refine the plan above**. Verification status marked on factual claims.

### 9.1 Adopted decisions (now binding for this project)

- **Experiment order (supersedes §6 ordering):** MiniLM + SetFit (lightest baseline) → **Qwen3-0.6B + LoRA** (first generative arm) → Qwen3.5-0.8B (newer challenger) → larger (Qwen3.5-2B) only if quality target missed. Encoder/clinical baselines (Bio_ClinicalBERT + head) run alongside.
- **Prediction-target definition:** classify smoking status **as documented in this note at the note's reference time** — NOT longitudinal patient-level history. Longitudinal reconciliation, if ever needed, is a separate dated reconciliation layer, never blended into snippet classification. Never apply "current overrides former" across the record.
- **Label policy headings (proposed by report, to be ratified as our guide):** current / former (requires documented cessation) / never (requires explicit lifetime non-smoking) / unknown (insufficient or conflicting). i2b2 carries a **one-year rule** (smoking within preceding year = current) — a model trained on i2b2 labels inherits it unless we relabel; we must choose our own cessation rule deliberately (handle "quit last week"). Scope decisions to settle in the guide: cigarettes vs all tobacco vs combustibles, cigars/pipes, vaping/smokeless as separate variables, exclude marijuana + secondhand exposure, patient vs relatives, conflicting statements, copied historical sections.
- **Richer internal record** exposed to audit (status + `evidence_offsets` + optional `unknown_reason` + `conflict_flag`), 4 labels only to downstream consumers.
- **Unknown vs uncertainty is a hard distinction:** "note says unknown" is a *label*; "model is uncertain" is signal for **abstention → review/expanded-context** path. Build a confidence/abstention lane into the architecture.
- **Retrieval is a tuned pipeline, not fixed ±100:** tokenizer-offset windows (not whitespace counts), collect ALL matches, merge overlapping windows preserving order, keep section headers + dates, preserve negation/cessation clauses at sentence boundaries, visualize boundaries between distant sections, treat triggers as retrieval hints only, audit regex-negative notes.
- **Training rule:** never give a small window a full-note label when the supporting evidence lies outside the window. Options: label visible evidence / expand context / concatenate all retained windows in one example.
- **Evaluation = whole pipeline** (retrieval + classifier), in addition to snippet-classifier-only numbers; report regex-positive cohort and end-to-end separately; evaluate unknown subtypes; patient-level splits BEFORE any window/teacher-derivative creation; watch near-duplicate/copied-note leakage; re-evaluate after quantization.
- **Safety-relevant error tracking:** current→never, former→never, family-history→patient misattribution get dedicated confusion analysis.
- **Teacher-policy decision** (was ambiguous in §2): validate teacher first on human-labeled dev data; teacher then labels permitted local notes; review disagreements; student trains on gold + reviewed teacher labels (hard-label distillation first). Locked test set stays outside ALL of it (no teacher selection, no prompt optimization, no training on it).
- **DSPy initial role:** optimize teacher instructions/examples only (using dev set), not required for the first student.
- **PHI/HIPAA wall (upgraded):** corpus cropping does NOT de-identify; BAA + safeguards required for cloud ePHI; no raw notes in trackers/logs/console; no automatic Hub uploads; DUAs reviewed separately even for de-identified data.

### 9.2 New model candidates added to the sweep (§4-A)

+ `Qwen/Qwen3-0.6B` (Apache-2.0) — first generative arm
+ `google/gemma-3-270m-it` (~268M, gated, Gemma license) — challenger
+ `Qwen3.5-0.8B`, `Qwen3.5-2B` — **verify existence/specs/licensing** (report asserts; unverified)
+ `sentence-transformers/all-MiniLM-L6-v2` + SetFit — **already in local HF cache**
+ Bio_ClinicalBERT (encoder head) — through Phase 1 sweep

### 9.3 Dataset facts added (§4-B)

- **i2b2/n2c2-2006**: report says **502** de-identified discharge summaries via `portal.dbmi.hms.harvard.edu/projects/n2c2-2006/` (my earlier estimate ~501 — **verify exact count**). Not a clean 4-label set: Smoker class ambiguous (current vs former), Non-smoker needs definition check; harmonize, don't silently mix.
- **SHAC / n2c2-SDOH** (tobacco events, event-level): GitHub `Lybarger/sdoh_extraction`; release availability **unverified** — do not plan around it.
- **MIMIC-IV-Note**: 331,794 discharge summaries (report) — source corpus only; needs our annotation/teacher labels; credentialing + DUA.
- **No verified good HF smoking 4-class dataset** — matches my own sweep expectation.
- **Local gold set is the priority for deployment**: guideline pilot 100–200 notes → dev corpus 1k–2k → locked patient-disjoint test → deliberate challenge set.

### 9.4 Infra notes (§4-E)

- Pilot ONE modest GPU first; no A100/H100 purchases until a memory/throughput pilot says otherwise.
- Still to verify from the report's inventory: HF Jobs/sandboxes details, Azure ML managed-network specifics, Unsloth Qwen3.5 route.

### 9.5 Milestone list (adopted as the execution sequence)

1. Write the label guide (cessation rule, "non-smoker" definition, unspecified history, product scope)
2. Annotate small local pilot, preserving evidence spans
3. Retrieval-strategy comparison (±50 / ±100 / ±200 / sentence-or-section-aware / union+social-history) measured on evidence retention, contradiction retention, token budget, end quality — **before** fine-tuning
4. Train MiniLM/SetFit + Qwen3-0.6B LoRA on identical patient-disjoint splits
5. Add Qwen3.5-0.8B challenger
6. Distillation only if needed
7. Deploy smallest model meeting the predefined quality target, with expanded-context review for difficult cases
   → Deliverable = **validated smoking-status extraction system**, not a checkpoint.

## 10. External Researcher Guide #2 — Integration (received 2026-10-28)

Second expert guide ingested. Verbatim source: `docs/external-reports/external-report2-guide-2026-10-28.md`.
Its four runnable starter scripts are already extracted, syntax-checked, and saved in `src/`:
`extract_windows.py`, `train_encoder.py`, `train_generative.py`, `benchmark_encoder.py`
(syntax-verified; per the guide itself, **not** yet executed with models/CUDA — treat as starters, not a validated pipeline).

### 10.1 Supersedes/updates §9 in these places

- **Model ranking (amends §9.2 order):** the guide's top-5 ranked — (1) `thomas-sounack/BioClinical-ModernBERT-base` (~150M, MIT) → first accuracy candidate; (2) Ettin encoder 32M → compact/distillation student; (3) ModernBERT-base → general-domain control (tests whether clinical pretraining helps); (4) all-MiniLM-L6-v2 → CPU baseline (**256-token sentence-transformer truncation — check snippet lengths**); (5) Qwen3.5-0.8B → generative challenger (has a vision encoder; benchmark the actually-loaded text path). Others: Bio_ClinicalBERT (110M, older arch), Qwen3-0.6B (easiest first decoder SFT — matches report #1), SmolLM2-360M-Instruct, Ettin 17M (final compression experiment), mmBERT-small.
- **Key architectural stance:** start with a **fully fine-tuned encoder + 4-way head**, generative LLM only as measured challenger/annotation assistant/fallback. **TRL is not needed for encoder arms** — cross-entropy full FT via Trainer is the encoder method; TRL SFTTrainer (`completion_only_loss=True`) is for the decoder arm only.
- **Baseline provenance for the encoder recipe (from the guide's `train_encoder.py` defaults):** lr 2e-5, effective batch 32, ≤5 epochs w/ early stopping on **dev macro-F1**, 10% warmup, weight decay 0.01, max_len 512, SDPA attention; LR sweep {1e-5, 2e-5, 5e-5} if needed; **3 seeds** for the encoder comparison arm; class weights only if imbalance causes measured failures.
- **Data-size plan (refines §9.3 / report #1):** 200–300 **double-annotated** policy pilot → ~2,000 train (aim 400–500/class, natural prevalence *not forced*) → 500 dev → **frozen test ~1,000 from untouched patients** at natural prevalence → challenge set 200–500 → expansion 5k–10k only if learning curves justify. Counts are evidence units/notes, **report unique-patient counts alongside**; CI example: 95% recall on 200 positives ≈ ±3pp; use patient-cluster bootstrap for paired model differences.
- **Snippet label vs full-note label = two linked targets (NEW, critical):** a crop that lacks the evidence supporting the full-note label gets `UNKNOWN/context-insufficient`, NOT the survived label. This prevents training the model to guess.
- **UNKNOWN policy refinement (stricter than report #1):** "denies smoking"/"nonsmoker" alone → **UNKNOWN** (it excludes current but cannot separate FORMER from NEVER). Exceptions rule: explicit past-smoking statement + clearly-current ambiguous "denies" → **FORMER**. Nicotine-replacement/counseling ≠ current use; temporary inpatient abstinence ≠ cessation; "quit last year, restarted last month" = CURRENT; "plans to quit" = CURRENT only when ongoing use documented. No universal CURRENT-wins; corrections need explicit/later-event narrative support.
- **i2b2 four-label mapping policy now explicit** (replaces §9.3's rough mapping): CURRENT → re-adjudicate (one-year rule includes recent quitters); PAST → FORMER iff evidence establishes prior use + cessation; **SMOKER (timing unspecified) → UNKNOWN with reason "ever-use established"** — never silently CURRENT; NON-SMOKER → NEVER only if definition/evidence indicates lifetime non-use. Deliver TWO benchmarks: original-label + separately named harmonized 4-class. Record source label → new label + conversion rationale per row.
- **Additional datasets (facts from the guide, scales verified by its author):** i2b2-2014 heart-disease **1,304 longitudinal notes / 296 patients** w/ smoking risk-factor annotations (longitudinal gold, DUA); SHAC/n2c2-UW-2022 = **4,405 social-history sections** (1,877 MIMIC-III + 2,528 UW; 3,067/447/891 splits; access two-gated: PhysioNet + UW; README wording contradictions — confirm with maintainers); **SDOH-NLI = 1,398 snippets / 29,635 premise–hypothesis pairs, CC-BY-4.0 → lowest-friction public source** (mirror row count 29,335 vs original — reconcile; group all pairs from one source note); MIMIC-III v1.4 ≈ 2M notes; MIMIC-IV-Note v2.2 = 331,794 DS + 2,321,355 radiology. **i2b2-2006 = 502 docs (398 train + 104 test, two pulmonologists)** — both reports agree; my Phase-1 verification still confirms before relying.
- **Published-evidence reality check (§3 of guide):** i2b2-2006 test set = **104 notes with only 11 current / 11 former** — minority-class evidence is thin; a January-2026 study (cTAKES vs 20B LLM, 250 local summaries, κ=0.91) merged current+former and used a broader tobacco definition — **it does not validate a sub-1B four-class cropped-input system**. Research gap we fill: no controlled comparison exists for our exact setting (4 labels + ±100 token crops + patient-disjoint evaluation + cost-per-million).
- **Cheap-control arms added to the comparison matrix** (amends §4-F to 7 arms): G = TF-IDF + logistic regression / linear SVM (word+char n-grams, **mandatory** floor baseline) + MiniLM frozen embeddings+head / SetFit. Hybrid arm F must report all routed cases (incl. fallbacks) and their costs — never only the accepted easy subset.
- **Methods guidance refined:** SetFit = few-shot experiment only (8/class not guaranteed to suffice); QLoRA unnecessary for sub-1B on 24 GB (ordinary LoRA simpler; 4-bit kernels can be slower — benchmark); soft-label distillation formula `L = α·CE(gold) + (1−α)·T²·KL(teacher ‖ student)` with α and temperature as dev-set choices; hard-label distill starting ratio **pseudo-labels ≤ human-labeled count**; teacher self-reported confidence is NOT calibrated accuracy; normalize LLM token log-probs over the full candidate label set before treating as class probabilities. DPO/RLHF/agents/multi-GPU = out of scope initially. Multi-task heads (subject/product/temporal) = sensible later experiment only if error analysis shows persistent failure.
- **Retrieval (extends §9.1):** broaden triggers to `cig\w*`, `hookah`, `shisha`, `snuff`, `chew\w*` etc. (see `src/extract_windows.py` CANDIDATES regex). **"PPD" ambiguity** (tuberculin test / postpartum day) — retrieval favors recall, classifier resolves sense. Retrieval-negative → UNKNOWN w/ retrieval-negative reason; audit filter-negatives with a random sample + enriched sample (report FN prevalence from the random one only). Note-level reconciliation: unrelated UNKNOWN evidence must NOT cancel clear patient-specific evidence. Cohort prediction: use only notes dated before the index date.
- **Cost model (concrete now):** `cost/million = 1e6 × H / (3600 × R)`, H = all-in machine rate, R = measured throughput. Decoder serving: count prompt tokens (policy+few-shot), bound output to label tokens; `calls/million notes ≈ 1e6 · q · k` (q = retrieved fraction, k = windows per note — measure both); fallback cost ≈ `C_student + f·C_teacher + routing` — small f matters if teacher is much slower.
- **Verified GPU pricing (guide's snapshot, verify at purchase time):** HF Jobs: T4 $0.40/h, L4 $0.80/h, A10G $1/h, A100-80GB $2.50/h. Modal: T4 $0.59, L4 $0.80, A10 $1.10 (HIPAA only in Enterprise). Lambda: A10-24GB $1.29/h, A6000 $1.09/h, A100-40GB $1.99/h. RunPod BAA: custom agreements ≥$3,000/mo — **first choice for real-PHI work stays institution-approved Azure**, not these vendors.
- **Training-budget envelopes (planning, not measurements):** 17–32M encoder 0.1–1 GPU-h ($0.04–1); 110–150M 0.25–2 GPU-h; 0.6–0.8B LoRA 0.5–4 GPU-h ($0.20–4); 2–3B LoRA/QLoRA 1–8 GPU-h ($1–20). Initial public-data multi-run sweep budget: **$20–100** (fits inside the $50–200 budget). Method: time 100–200 steps → `epochs × ceil(N/eff_batch) × median_step + eval + checkpoints + startup`.
- **Decision rule (adopted):** deploy the **smallest model whose paired evaluation stays within ~1 macro-F1 point** of the strongest baseline **without an unacceptable CURRENT-recall drop**; if CI too wide, collect more independent eval data instead of declaring equivalence.
- **Environment pins (from guide's starters):** Python 3.11 + `transformers==4.57.3` for encoder arms; Qwen3.5 needs its own current compatible env (not covered by that pin); correspondences verified for `processing_class` / `eval_strategy` / `DataCollatorWithPadding`; confirm exact TRL/PEFT signatures via a public-data smoke run before assuming.

### 10.2 Guide #1 vs Guide #2 conflicts and resolutions

| Topic | Guide #1 | Guide #2 | Resolution |
|---|---|---|---|
| First model | Qwen3-0.6B LoRA first | BioClinical ModernBERT full FT first | Guide #2's encoder-first is adopted as primary (matches your "lightweight, millions-of-snippets" goal); Qwen3-0.6B stays the first *generative* arm |
| "Denies smoking" detail | UNKNOWN unless lifetime non-use documented | Same, adds FORMER escape rule (past + ambiguous current denial → FORMER) | Adopt Guide #2's ruling table |
| i2b2 size | 502 | 502 (398/104) | Consensus; verify during Phase 1 |
| Teacher role | DSPy-optimized teacher | DSPy MIPROv2/GEPA + typed Literal signature | Adopt Guide #2's concreteness; keep locked test outside everything |

### 10.3 Open items Phase 1 must verify (grows §9.5)

- i2b2/n2c2 2006 = 502 (398/104);	i2b2-2014 dataset terms; SHAC access status (README contradictions); SDOH-NLI mirror count mismatch; Ettin + BioClinical-ModernBERT + Qwen3.5 model cards/licenses/commits; the provider price snapshot.

### 10.4 Roadmap (Guide #2's exit-condition version adopted)

| Phase | Output | Exit condition |
|---|---|---|
| Baseline & policy | 4-label policy, 200–300 adjudicated pilot, LLM + TF-IDF baselines | Reviewers agree on current/former, ambiguous negation, scope, conflicts |
| Data prep | dataset access, frozen leakage-aware splits, retrieval audit, crop/full-note pairs | No split leakage; context-loss quantified |
| First fine-tune | BioClinical-ModernBERT, ModernBERT, Ettin-32M runs | Comparable dev results, per-class errors, reproducible models |
| Targeted expansion | active learning; optional teacher labels | Gains survive independent human dev eval |
| Optional decoder/distill | Qwen3/3.5 comparison; 150M→32M compression | Benefit justifies complexity or saves measured cost |
| Optimization | batching/precision/runtime benchmarks + calibrated routing | Accuracy + critical recall within tolerance |
| Validation | frozen human test, challenge set, unseen style/site/time | Error bounds + fallback workload acceptable |
| Deployment | approved batch job, provenance, monitoring, rollback | Stable end-to-end throughput + sampled quality |

**First build (Guide #2's concrete proposal, adopted):** one approved Azure GPU job → BioClinical-ModernBERT-base full FT with 4-way head on human-adjudicated data → conservative UNKNOWN policy → ±100 vs ±200/section context study → Ettin-32M alongside (deploy it if within tolerance, else keep 150M). Add DSPy teacher labeling, QLoRA, or generative fallback only when a measured limitation calls for it.
