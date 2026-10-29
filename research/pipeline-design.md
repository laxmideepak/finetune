# Pipeline Design v0.9.1 — Smoking-Status Extraction from EHR Note Crops

> v0.9.1 (same day): full-text integration pass — S-7/S-11/S-12/S-13 fully ingested, cross-dataset harmonization register added (§9-4.3), risk R7 added, study queue extended to S-24, E2 benchmark located.
Compiled 2026-10-28 by the design-research pass (user directive: "design and research the pipeline extensively before implementation").
Method: applied brainstorming-research-ideas frameworks (F1–F10) to our specific pipeline; every factual claim is anchored to a verified primary source (see §16). This document is the design contract for the implementation stage; changes go through the decision log in research-log.md.

---

## 0. The pitch (F10 — Explain-It Test)

> **Sentence 1 (Problem):** Hospitals record smoking status only in unstructured note text, so at million-note scale researchers either pay $120–6,610 per million for cloud LLMs to read whole notes (verified 2026 cloud price range) or accept regex-level garbage — and no published controlled comparison exists for the four-label, cropped-input, patient-disjoint setting every deployed system actually needs.
>
> **Sentence 2 (Insight):** We systematically find the smallest open model or training strategy that classifies four smoking-status labels from cheap regex-triggered ±100-token crops — encoder-first, distilled if justified, whole-pipeline evaluated — measuring the real frontier of "accuracy per dollar per snippet" where the 2026 literature has only scattered points.

Skeptic's predicted first objection: *"gpt-oss-20B already got F1 0.99 zero-shot on tobacco (JAMIA Open 2026) — why build anything?"* Response prepared: that result is (a) whole notes in a 3-class setting, (b) regardless of cost (2.5–14.5 s/note verified — at 14 s/note, 1M notes ≈ 89 GPU-hrs ≈ days of single-GPU time and ~$100–900 rented), and above all (c) **the study set no $/million benchmark and no crop-level training-data rule**; our contribution is exactly the controlled, patient-disjoint, cost-normalized comparison and the conservative-UNKNOWN training policy, not another point-estimate.

## 1. Problem-first check (F1)

- **Who suffers:** any clinical-research group phenotyping millions of notes for smoking status (epidemiology, lung-cancer screening, drug-outcome adjustment of confounders — verified pipeline in arXiv 2411.03004 where smoking status is a *causal confounder* whose classifier error propagates to effect estimates).
- **Is it unsolved?** No system in the verified literature trains a sub-150M classifier on evidence-labeled crops with a conservative UNKNOWN policy and reports $/million + CURRENT-recall jointly. MSEP (JMIR 2026) is closest in spirit (sentence-level, deployed) but: 6 conditions, CamemBERT-only, no cost frontier, no crop-vs-full ablation. i2b2-2006 is the shared benchmark but is 3-class-ambiguous + Unknown-flooded + 20-year-old.
- **Person who needs this:** the user — million-snippet pipelines at their institution, cost-per-snippet decisive, no giant LLM budget, real-EHR work confined to approved environments.

**Verdict: problem-first, validated.** Proceed to design.

## 2. Design tensions (F3) — the six that shape this pipeline

| # | Tension | Naive resolution (rejected) | Our reconciliation (designed in §9) |
|---|---|---|---|
| T1 | **Accuracy ↔ Cost-per-snippet** | "Use a big LLM" (prototype prices at 1M scale) | Deploy smallest model within ~1 macro-F1 of best arm w/o CURRENT-recall drop (adopted rule from guide#2, §5); measure $/M for every arm on the same hardware (§9-9) |
| T2 | **Crop compression ↔ Evidence completeness** (±100 tokens vs decisive evidence outside crop) | Fixed ±100 everywhere | Dual targets: snippet-label (crop-supported, may = context-insufficient) vs full-note-label; retrieval ablation ±50/100/200/section-aware (§9-3,9-4); explicit boundary markers; truncation refusal (guide#2 script behavior) |
| T3 | **"Note says unknown" ↔ "Model is uncertain"** | Merge abstention into the UNKNOWN class | UNKNOWN is a *label*; model uncertainty is a *routing signal* → abstention lane max-softmax threshold on calibrated dev + expanded-context fallback + human review queue (§9-7) |
| T4 | **Recall-first triggers ↔ Precision of classifier input** | Regex filters as the classifier's truth | Triggers are retrieval hints ONLY; audit lane measures trigger recall on a regex-negative sample before deployment attributes are trusted (§9-2; precedent: i2b2-2006 winners used exactly this limited-feature insight) |
| T5 | **Institutional privacy ↔ Development velocity** | "De-identified ⇒ cloud OK" or "crop ⇒ anonymous" | Public/de-identified/synthetic data anywhere; real ePHI only in BAA-approved environment; neither cropping nor de-identification automates cloud eligibility (HHS guidance; guide#2 §9 DUA wall) |
| T6 | **One 4-way softmax ↔ Per-class evidence heterogeneity (FORMER needs cessation, NEVER needs lifetime absence, UNKNOWN is heterogeneous)** | Force one head for all classes | Primary: single 4-way head (simplicity test, F7). Registered challenger (H1-adjacent): MSEP-style decomposition into per-class binary heads (verified deployed at 0.94 macro-F by MSEP) run as ablation E-x after baseline exists |

## 3. Why now (F5 — What Changed)

| Prior assumption invalidated | What changed | Consequence for us |
|---|---|---|
| "Small encoders can't be both fast and accurate" | ModernBERT (Dec 2024, verified): RoPE+alternating attention, 8192 ctx, unpadding/packing, variable-length fast, small-GPU friendly; BioClinical-ModernBERT-base (150M, MIT, MIMIC-corpus MLM, Social-History bench 58.5) exists | The 150M clinical encoder first-build is materially stronger than 2018-lineage BERT encoders; justify clinical-pretraining by testing plain ModernBERT-base as control (arm D) |
| "Fine-tuning decoders is inconvenient/relative-heavy" | TRL SFT (v-era docs verified): completion-only loss default for prompt-completion, adapter LR ~1e-4, packing, DFT loss option; Qwen3-0.6B first-class support | Generative challenger arm is cheap to build correctly (guardrails: version pinning resolved in smoke test) |
| "Cheapest extraction needs narrow rules" | 2026 datapoints verified: cloud LLM extraction $0.12–6.61/1k notes; cheapest structured-model $0.61–0.64/1k (Wright medRxiv un-reviewed) | Our ≥10×-cheaper-per-snippet H0-ARCH target is now quantifiable against real 2026 numbers |
| "Zero-shot medical LLM needs big frontier API" | gpt-oss-20B (open, Apache-2.0-class) 0.99 F1 tobacco vs cTAKES 0.97 zero-shot, on-prem (JAMIA Open 2026 verified); Llama-3.3-70B 98.6% among 30 vars (NPJ Precis Oncol verified) | Teacher route needs no ePHI egress: privately-hosted open-weights teacher (H4) is a first-class design path, not a fallback |
| "i2b2-2006 access = online form" | Portal TLS cert expired (verified first-hand); legacy i2b2.org 500s | Manual DUA by user in browser; meanwhile development proceeds on zero-DUA corpora (§11) |

## 4. Boundary probing (F6) — fault-injection matrix for OUR pipeline

Each row = a boundary our system must survive or knowingly fail-safe; test enters the eval plan (§9-8).

| # | Stress input | Expected failure mode naive system | Designed behavior | Test to add |
|---|---|---|---|---|
| B1 | "denies smoking" with explicit past-smoking sentence elsewhere in note | crop answers off one sentence; note-level flips FORMER↔UNKNOWN | Crop keeps NO status but flags evidence; aggregation uses explicit-past rule (verified i2b2 rule: past+temporal → FORMER when consistent) | synthetic pair test |
| B2 | "quit smoking four months ago" | model reads "quit" ⇒ FORMER | i2b2 one-year rule (verified: 4-month-quitter = CURRENT class in the 2006 gold) — our policy must ratify or reject this and annotate the consequence | challenge set entry |
| B3 | "smokes 1 ppd" inside Family History section | relatives misattributed to patient | section-presence in crop + aggregation rule: family-history-only evidence ⇒ note-level UNKNOWN + conflict/review flag | challenge set entry |
| B4 | "secondhand smoke exposure" | NEVER or CURRENT confusion | verified i2b2 rule: secondhand ⇒ Non-Smoker-family; but our 4-label policy routes to UNKNOWN unless lifetime non-use explicit — ratify (recommended: unknown, exposed≠never) | challenge set entry |
| B5 | marijuana-only smoking | GENEROUS jamais→NEVER | verified i2b2 rule: marijuana excluded from tobacco smoking status | challenge set entry |
| B6 | regex-negative note containing "smoker's house", "smoke alarm", "no smoking today due to surgery" (ad-hoc triggers) | retrieval miss or false positive | regex-v2 recall-first patterns + unknown_reason=retrieval-negative + audit of random regex-negative sample | random sample audit (n=200)
| B7 | near-duplicate/copied notes across patients (templates) | train-test leakage inflating scores | patient-disjoint splits + note-hash dedupe + template-cluster checks (verified cautionary instance: MSEP's 3–8% same-patient sentence overlap across CV folds — real-world teams miss this) | leakage checker script
| B8 | crops where evidence text sits exactly at boundary | silent truncation of the decisive clause | truncation refusal (raise, never silently elide — guide#2 script), boundary markers `«…»` visible | fuzz test
| B9 | mixed-length batch on small GPU | padding waste / OOM | ModernBERT unpadding+BF16; MPS: small batch, grad-ckpt (verified OOM lesson) | throughput benchmark E4
| B10 | model over-confident on ambiguous crop | review-channel never triggers | calibration via temperature/logit threshold on dev; report AURC-style risk-coverage curve | calibration eval
| B11 | quantization changes错误 | deploy-time accuracy collapse | re-evaluate after quantization (adopted rule, guide#2) | post-quant eval slot

## 5. Simplicity test (F7) — arm justification and pruning trigger

7-arm matrix (guide#2 baseline matrix) justified row-by-row; each arm exists to answer a named question, not for arm-count prestige:

| Arm | Question it answers | Prune if |
|---|---|---|
| A TF-IDF+logit/SVM (mandatory floor) | "Does any learned noise beat n-gram rules at all?" (MSEP verified: rules WIN sometimes) | never — mandatory sanity floor |
| B MiniLM frozen + logit head | "Do generic 22.7M embeddings carry enough?" (256-wp truncation verified card) | after E-x measurement, if <A by >3 macro-F1 |
| C BioClinical-ModernBERT-base full FT | "Does clinical domain pretraining + modern arch win?" (primary candidate) | never — primary |
| D ModernBERT-base full FT (plain) | "How much does clinical pretraining actually buy?" | after C/D paired result recorded |
| E Ettin-32M full FT | "How small can we go within ~1 F1?" (distillation student candidate) | after paired C/E result |
| F Hybrid router (rules route easy crops, encoder resolves remainder) | "Cheap routing saves what fraction without accuracy cost?" | after F-cost analysis; report routed cases incl. fallback (adopted reporting rule) |
| G Qwen3-0.6B LoRA SFT | "Does decoder flexibility beat encoder+head on crops?" | after paired C/G result |

Sequencing adopted: E1–E2 zero-DUA smokes → arms A/B/C/D (first fine-tune wave on first real labeled data) → E/F/G second wave → G→Qwen3.5-0.8B only if label errors favor generative. **3 seeds** for encoder comparison analyses; patient-cluster bootstrap for pair differences (adopted from guide#2).

## 6. Stakeholder rotation (F8) — requirements extracted per stakeholder

| Stakeholder | Top requirements → design hooks |
|---|---|
| **End researcher** (million-snippet analyses) | numbers per million at predictable cost; unknown_reason for imputations; audit trail — §9-7/9-9 + provenance fields (§9-4) |
| **Clinician/informaticist (policy ratifier)** | label guide grounded in a cited standard (i2b2 verified rules), conservative UNKNOWN, review lane — §9-4/9-7 |
| **Developer (us)** | typed schemas, one canonical windows JSONL, single point of truth in MASTER_CONTEXT, version-pinned env — Stage contracts §9 |
| **Operator (ML-Ops at scale)** | throughput R measurement, batch sizing, $/M reporting, resume-able state files — §9-9 |
| **Regulator / privacy office** | BAA boundary, no raw notes in logs, DUA review, no auto-Hub uploads — Stage 0 (§9-0) + T5 |
| **Adversary (what could break us)** | label leakage across patient splits (B7), teacher-policy error inheritance (F-detect via different, human-only test), quiet truncation (B8) — all mapped to tests |

## 7. Composition / decomposition insights (F9)

- **Decompose 4-way → per-status binary heads**: MSEP (verified, deployed macro-F>0.94 on 5/6 conditions) proves per-class heads are production-viable; registered as our T6 ablation after baseline. Simplicity test says: do NOT start there. Start with single head.
- **Compose KOTI-style evidence placement + our windows**: trigger-anchored windows over tail-truncated notes — the +8–24% encoder F1 headline (Alleva BMC MIDM 2025) is a 5-task average; **smoking-specific (S-9 full text): the significant smoking gain is keyword-anchored truncation vs naive tail-truncation (+15.8–21.6% F1, p<0.001), NOT template insertion itself (+2.9–3.4% ns, p=0.50)** — our ±100-token trigger-anchored crops already deliver the smoking benefit; template-position effects are encoder claims where crops aren't sorted by source order by default (evidence-density ordering; boundaries preserved).
- **Compose CHOP's trigger-focused argument resolution (ocad012 2nd place)**: they improved argument subtype resolution by setting segment IDs=1 on the target trigger; ModernBERT has no token_type_ids (S-4) so the encoder arm cannot copy this directly — the generative arm (G) can exploit the same idea by inlining the trigger in the prompt, and the encoder arm leans on our evidence-density ordering instead.
- **Decompose aggregation** into explicit precedence rules (affirmative patient-specific evidence > negated statement resolution > family-history exclusion > temporal one-year rule application) instead of learned weighting — matches the verified 2006 winners' structure (sentence-level→rules) and keeps the aggregation auditable at the human level.
- **Compose measurement-error correction (Lee & Wood-Doughty arXiv 2411.03004)**: our pipeline emits confusion-matrix artifacts enabling downstream cat-error correction without extra work — slicing a research bonus out of eval boxes.

## 8. Abstraction ladder outlook (F2 — for the paper/reuse)

- **Down (instantiated)**: the exact deliverable — smoking status, 4 labels, crops, patient-disjoint eval, $/M frontier (the benchmark paper of record).
- **Up (generalized)**: "evidence-snippet SDOH-status extraction at million-note scale" — the same machinery applies to alcohol/drugs/housing (SHAC covers tobacco events explicitly; verified released+DUA path). Design keeps all stage contracts status-agnostic (trigger regex + label guide are the only swap-ins).
- **Sideways (adjacent)**: pack-year extraction (Wright verified need), quit-date extraction, family-history status — same pipeline, per-status label guides.

## 9. Architecture v0.9 — stage-by-stage design (the core)

### Stage 0 — Ingest, privacy & provenance envelope
- Mechanism: corpus in → (a) PHI-screening checklist file (institution env vars: ENCLAVE=1 forbids external fetches), (b) per-note persisted provenance blob: {source_db, note_id, patient_id, note_date, de_id_status, ingest_hash}. Storage: local parquet/jsonl only.
- **Decisions:** no raw notes to cloud without an executed BAA checklist = hard gate (` polity_can_egress()` returns false by default); no auto-Hub uploads (all arms default push_to_hub=False — re-verified in TRL docs defaults for SFTConfig: push_to_hub False).
- Failure mode logs redacted (adopted rule: no raw notes in exception messages).
- Open: institutional GPU/quota — user-side action item.

### Stage 1 — Preprocessing & normalization
- Mechanism: sentence splitting (section-aware splitter: keeps recognized *section headers* as anchors —וכל "Social History", "Family History", "Past Medical History"); date detection (Heuristic + regex years/quarters) annotating "note reference time".
- Decision: no heavyweight dep Matchers yet (no CACI); plain regex + header list sufficient for smoke wave; swap to formal SecTag-style module later if measured mis-splitting >2% (metric defined in study queue item S-7).

### Stage 2 — Trigger retrieval v2 (recall-first, hints only)
- Mechanism (from `src/extract_windows.py` verified regex + guide#2 broadened lexicon): `smok\w*, tobac\w*, cigar\w*, cigaret\w*, nicotine, pack[s]?[- ]?year, [0-9]+ *(ppd|pack[s]? *(a|per)?day), cigs?\w*, hookah, shisha, snuff, chew\w*, (vape|vaping|e-?cig\w*)` (vaping flagged as separate variable, label-scope decision pending); capture ALL matches; emit:
  - `trigger_metadata`: {pattern_id, span, section_id, sent_id, is_negated_family_margin (negation handling left to classifier, retrieval stays recall-first)}
- Audit lane: sample regex-negative notes (random n=200 + enriched n=50 prioritized) — tagged FN prevalence; report separately (adopted rule).
- Verified vocabulary/template gaps to cover from ocad012's cross-domain error analysis: "toxic habits" (UW synonym phrase for substance/tobacco use); "nicotine" already in lexicon; and **unpopulated template slots ("Tobacco use: _") which generated cross-zone FPs** → rule: empty-value template lines are evidence of unknown_reason=no-mention, never affirmative.
- Decision: no stepping-up to embedding retrieval (MiniLM/all-mpnet) in v0.9 — match i2b2 winners' insight that feature-set is small; revisit if audit FN prevalence > 5%.

### Stage 3 — Crop construction
- Granularity decision (ratification candidate, §15): **sentence-aligned merged windows** — expand each match to sentence boundary, merge overlapping spans preserving source order, cap at budget (±100 tokens default), hard-cap total to 512 tokens per example (ModernBERT 8k ignored deliberately: crops cheap).
- Boundary markers `«…»` retained; KOTI-style: instruction/exemplar placed adjacent to top-evidence sentence when formatting for generative arm; encoder arm unaffected (CLS). **G-arm source-order caveat (S-9 full text): reordering input around the trigger significantly HURT the decoder models on the smoking task (BioGPT −10.1% F1, p=0.018; ClinicalT5 −7.8%, p=0.051) → generative arm keeps source order inside the crop and places the instruction as a prefix before the evidence, never mid-sequence; template-position tricks are encoder-only claims.**
- `unused evidence rope`: If total retained evidence exceeds budget → create TWO examples: (crop-level evidence-rich first) + (note-context example = all retained windows, capped at 1024) — no data thrown away silently (guide#2's non-truncation rule).
- Failure modes: B8 handled by raise-on-truncate in training prep; B9 handled at training-loop config.

### Stage 4 — Labeling & dual targets (the data contract)
- Two rows per source note (when both derivable):
  - `label_note`: full-note 4-label at note reference time per our policy (§9-4.1)
  - `label_crop`: per-crop label from evidence *within* crop; if the full-note label's supporting evidence lies outside crop ⇒ `context-insufficient` (maps to UNKNOWN at training time w/ reason; flagged hidden-correctable at eval)
- 9-4.1 Label policy v1 (for ratification w/ clinical reviewer; i2b2-2006 verified rules where applicable):
  - CURRENT: explicit ongoing (i2b2: within 1 yr incl. recent quitters — ratify adopting or rejecting the one-year rule; recommended adopt for v1 given research continuity with i2b2-labeled corpora)
  - FORMER: explicit prior use + cessation documented
  - NEVER: explicit lifetime non-use ("no tobacco"/"never smoked"; secondhand counts toward smoking *exposure*, status stays patient's own tobacco)
  - UNKNOWN: missing/unresolvable/third-party-only/contradictory/retrieval-negative
  - Escape rules (needed for markup): "denies smoking" alone ⇒ UNKNOWN unless explicit past context (⇒ FORMER, verified i2b2 escape analog); "quit <1 yr" ⇒ CURRENT if one-year rule adopted
- Provenance per row: label_source (gold/teacher/hybrid), annotator_ids, evidence offsets, unknown_reason, conflict_flag, review_date.
- 9-4.3 Cross-dataset harmonization register (added 2026-10-28; every corpus's label file emitted separately, never mixed):
  | Corpus | Native status taxonomy | Mapping → our 4-way | Conflict / notes |
  |---|---|---|---|
  | i2b2-2006 | Past / Current / Smoker / Non-Smoker / Unknown | §9-4.2 mapping | Ratified one-year rule = i2b2's own rule ✓ full continuity |
  | SHAC (tobacco events, per section) | Status ∈ {none, current, past}; absent event = no mention | none→NEVER (explicit denial/none only); current→CURRENT; past→FORMER; absent→UNKNOWN_reason=no-mention | SHAC "none" includes denials; recent-quitter cases unverified — audit at accession |
  | Wright LCS synthetic | per-history fields + eligibility decisions; never = <100 lifetime cigarettes; any documented quit ⇒ FORMER (even e-cig/NRT switch) | map to their task format for comparability only | **Conflicts with ratified one-year rule**: "quit 4 months ago" ⇒ FORMER here vs CURRENT in our v1. Use as E2 pipeline-comparability set + label-policy sensitivity study, never as gold. Never definition (<100 cigs) matches CDC convention — useful anchor |
  | SDOH-NLI | entail / contradict / neutral per hypothesis | not a label source (format/smoke only) | — |

### Stage 5 — Model arms (configs locked for implementation wave 1)
- C (primary): BioClinical-ModernBERT-base + 4-way head. lr {1e-5, 2e-5, 5e-5}, eff batch 32, ≤5 epochs w/ early stopping on dev macro-F1, max_len 512, BF16, 3 seeds; truncation refusal; no token_type_ids (verified ModernBERT architecture).
- D: plain ModernBERT-base, identical — isolates clinical pretraining value.
- E: Ettin-32M, same protocol, max_len 256 fits crops.
- A: TF-IDF word+char (1-2,3-5)-grams, LogisticRegression + LinearSVM grid.
- B: MiniLM frozen 384-d + LR head; sentence-level input only (128-token training verified), one crop per evidence-span not merged.
- G: Qwen3-0.6B SFT LoRA: TRL-era defaults verified (completion_only_loss only-left implicit default; explicit in config for clarity), adapter r16 α32 dropout .05, LR 1e-4 (adapter guidance verified), batch-2 grad-acc 8, grad ckpt default-on (verified SFTConfig default), packing off, max_length 1024, eval-on-macro-F via generation of label string and exact-match on locked eval set.
- Environment pins: resolve at smoke (transformers 4.57.3 vs current 5.x+TRL1.16-evolution: the verified SFT doc page reflects ModernBERT/TRL current; our starter scripts were written for 4.57 API set (processing_class/eval_strategy) — smoke test resolves; report in research-log).

### Stage 6 — Aggregation (crops → note)
- Policy (deterministic, auditable):
  1. Any patient-specific affirmative CURRENT evidence ⇒ CURRENT
  2. Else any FORMER-supported evidence w/ cessation ⇒ FORMER
  3. Else explicit NEVER (lifetime) ⇒ NEVER
  4. Else UNKNOWN (+reason among: no-mention, retrieval-negative, timing-unresolved, family-only, conflict)
- Non-rules adopted tactical: no current-always-wins-forever (contradiction w/ later cessation documented requires explicit flag → conflict/review); unrelated UNKNOWN does not cancel affirmative (adopted rule, guide#2).
- Every aggregation decision logged w/ contributing crop ids (audit).

### Stage 7 — Abstention & uncertainty lane
- Calibrated max-softmax threshold θ on dev; per-class thresholds (CURRENT higher bar—fewer false-CURRENT); routing:
  confidence ≥ θ ⇒ answer; < θ ⇒ expanded-context pass (1024-token example); still < θ ⇒ review-queue w/ evidence dump.
- Calibration artifact: reliability curves + risk-coverage; re-done after quantization.
- Note-vs-model distinction enforced via schema (label UNKNOWN vs abstention scheduled as separate field — never overwritten).

### Stage 8 — Evaluation contract
- Splits: patient-disjoint, hash-deduped, locked test untouched by ALL optimization (teacher/prompt/arms/windows) — one set per source corpus; challenge set 200–500 hard cases (B-matrix rows); template-cluster cross-check.
- Metrics: macro-F1 (primary), per-class P/R/F1, CURRENT-recall & CURRENT-precision separately (safety), family-history→patient misattribution rate, calibration (ECE + risk-coverage), unknown-subtype breakdown, retrieval-layer metrics (fn rates above), and full-pipeline metrics alongside crop-classifier metrics (reported separately — never blended).
- CIs: patient-cluster bootstrap; paired model differences; significance α=0.1 randomization (verified i2b2-2006 procedure for comparability).
- Deployment gate pre-registered: smallest model within 1 macro-F1 (95% CI overlap accepted) w/o CURRENT-recall drop >2pp vs best; report the $/M alongside.

### Stage 9 — Serving & cost measurement
- Mechanism: HF Jobs (verified pricing table) A10G/L4 benchmark; median+P95 latency, R (tokens or snippets/sec) warm; $/M = 1e6·H/(3600·R) (guide#2 formula).
- Quantization: dynamic INT8 eval slot after winner chosen; re-eval B11.
- Milieu notes: MPS local = prototype only (verified OOM lesson); HF Jobs billing per-minute — set timeout explicitly (default 30 min verified).

## 10. Comparison-matrix report template (brief)
Every result row includes: arm, data slice (regex-positive/all), split, seeds, macro-F1 ±CI, per-class table, CURRENT metrics, family-history error, $/M, GPU-hrs. One table = the paper's Table 1.

## 11. Data acquisition priority queue
1. **SDOH-NLI** (zero-DUA, CC-BY-4.0 verified): smoke corpus for lengths/formatging/section stats; NOT a smoking label source (not smoking domain — verified).
2. **LCS synthetic benchmark** (Wright medRxiv 2026-09; "shareable" claimed — locate link via preprint PMC/medRxiv before relying)
3. **SHAC** (DUA + PhysioNet; verified form URL): first real tobacco-annotated clinical text, section-level.
4. **i2b2-2006** (user manual DUA once portal up; portal TLS expired verified): 398 train.
5. **MIMIC-IV-Note** (later; credentialing+DUA; large-scale unlabeled+teacher/annotation).
6. **Local gold** (institution; 100–200 pilot per guide#2's allocation).

## 12. Study queue — everything to ingest before implementation (the user's "go through every doc" list)
Status legend: [x] ingested+verified into design; [ ] queued.

| # | Resource | Stage(s) it decides | Purpose & verified-relevant fact | Status |
|---|---|---|---|---|
| S-1 | Uzuner et al. JAMIA 2008 (PMC2274873) | 4,6,8 | label rules verbatim; κ/agreement methods; metric conventions (micro vs macro), significance α=0.1 randomization | [x] |
| S-2 | guide#2 (research/external-report2…) | all | ranking, cost model, split sizes, caution on MiniLM truncation, scripts | [x] |
| S-3 | guide#1 (research/external-report…) | 4,5 | TRL skeleton; label policy philosophy; deployment checklist | [x] |
| S-4 | ModernBERT blog (Dec 2024) | 5,3 | no token_type_ids (collator!), FA2, alternating attn 128-local + global-every-3, unpadding | [x] |
| S-5 | BioClinical-ModernBERT HF card (thomas-sounack) | 5 | MIT, 150M, corpus incl. glossary, Social-History bench 58.5 | [x] |
| S-6 | TRL SFTTrainer docs (current) | 5(G) | verified completion_only_loss/packing/model_init_kwargs/eos_token defaults; Qwen3-0.6B quickstart; DFT optional | [x] |
| S-7 | MSEP paper (JMIR 2026, PMC13044345) | 2,5,6 | FULL TEXT ingested 2026-10-28. Added: LLM-prompt arm (Mixtral-8×7B) intermediate & heavy — 12–24 s/sentence, 35GB GPU vs CamemBERT 0.09–0.12 s, 2GB; rules WIN sparse classes (family-hx macro-F 0.94±7.9% vs CamemBERT 0.73±17%) but CamemBERT wins smoking (F 95%); annotation 1.2–2.9 s/sentence (vs NER 7.82–16.48 s); active learning REJECTED (updates distorted by majority samples) — note tension w/ SHAC where AL succeeded; **verified leakage instance: dedupe pre-CV but no patient stratification → 3–8% same-patient sentences across folds** (our B7/Stage-8 forbids exactly this); doc-type/section shifts meaning ("hypertension" chronic in Past Med Hx vs acute in daily note) → section-aware data selection justified; preannotation bias monitored via 10% non-preannotated sentences; low-prevalence statuses effectively binary | [x] full |
| S-8 | Dávila-García JAMIA Open 2026 (PMC12803785) | 5(G), 9 | gpt-oss-20B 0.99 vs cTAKES 0.97; 2.5–14.5 s/note; 3-class labels caveat | [x] (abstract verified; full table of per-model scores queued) |
| S-9 | KOTI paper (Alleva E et al., BMC MIDM 2025, PMC12224782 / PMID 40611214; code github.com/eugenial/koti) | 3 | FULL TEXT ingested 2026-10-28. **Its Smk task = N2C2 smoking challenge: 502 notes, smoker+current merged (3 test-only examples) → 4 classes, challenge split 398/104 — direct pre-comparator for our exact task/labels.** Zero-shot macro-F1 0.13–0.18 across all 4 models; few-shot best 0.346 (ClinicalBERT KOTI, N=50) / 0.471 (GatorTron KOTI, N=10 random) → **few-shot prompting is weak for 4-class smoking status vs full-FT 0.85–0.90 (Palmer, 2006 winners) → few-shot arms stay fallbacks (F7), never primary**. Smoking-specific placement facts: KOTI vs STI-k (template near evidence, keyword-truncated input) NOT significant (+3.4/+2.9%, p=0.50/0.53); KOTI vs STI-s (naive tail-truncation) significant for Smk: +21.6% GatorTron (p=0.0005), +15.8% ClinicalBERT (p=0.0008) → **the smoking gain comes from keeping evidence in-window, not template insertion per se — our trigger-anchored crops already deliver it**; template-distance decay verified (F1 peaks at keyword sentence, declines at ±1–12 sentences). Scale facts: 98% of N2C2 obesity notes >512 tokens → chunk-predict-then-max-pool ≈3 passes/note → bounded crops remain the million-snippet answer; GatorTron (~350M) beat ClinicalT5 (~740M) → domain pre-training > parameters. Decoder asymmetry: BioGPT +19% avg but **Smk KOTI significant DROP −10.1% (p=0.018); ClinicalT5 −7.8% (p=0.051)** → do NOT reorder decoder input around triggers. Related same-group dedup paper (B7-adjacent, deferred): Landi/Alleva arXiv 2312.09469 | [x] full |
| S-10 | "Alleva follow-up": few-shot curves | 5 | **CLOSED 2026-10-28 — no separate follow-up exists**: the Alleva paper IS KOTI (PMID 40611214 = PMC12224782 = S-9); zero/few-shot curves (k=1,4,10 per class + N=50/100 random; balanced vs random) are inside it. PubMed author search screened 102 hits — only clinical-NLP Alleva = KOTI; other matches are homonymous authors (Maastricht psychology, MPI polymer chemistry, D'Alleva cardiology, D'Alleva sports science). The 8/class-few-shot question answered empirically in S-9: smoking few-shot macro-F1 0.35–0.47 ≪ full-FT 0.85+ → balanced few-shot helps but not to deployment gate | [x] closed (resolved in S-9) |
| S-11 | Wright LCS preprint (medRxiv 2026-09.24) | 3,8,9 | FULL TEXT ingested; **benchmark LOCATED: github.com/vclic/smokingbenchmark (3,000 notes + reference labels, CC-BY-4.0) + github.com/vclic/smokingeval (prompts, schemas, frozen hashes, all 15 runs' predictions)**. Costs/1k: Luna $0.12–0.21, Jev $0.61–0.64, Sol $2.44–4.22, Haiku $3.37–4.43, Sonnet $3.76–6.61; prompt caching −47% (Sonnet), Haiku uncacheable (<4,096-token prefix); Jev ≈15k input tokens/note (answer-options dominate; input $0.042/M, output free, latency 0.45–1.21 s); quit-date errors → 12/14 Jev FNs under USPSTF; sensitivity 0.957→0.993 under ACS ("evaluate against the rules you actually feed"); label rules = quit-however-recently ⇒ FORMER (even e-cig/NRT) + never = <100 lifetime cigarettes → **conflicts with ratified one-year rule, logged in §9-4.3**; template-shorthand notes HARDER than LLM-prose (evals need both formats); reference adjudication performed by an LLM (their stated limitation); benchmark built for $62.34; n2c2 portal datasets "temporarily unavailable" since Sep-2024 (consistent w/ our expired-TLS finding) | [x] full |
| S-12 | SHAC: Lybarger JBI 2021 (PMC7856628) + n2c2/UW 2022 shared task (JAMIA ocad012, PMC10354764) | 4,11 | BOTH FULL TEXTS ingested 2026-10-28. Substance-use event schema: Status (required; **{none, current, past}**) + Duration/History/Type/Amount/Frequency span args; 4,480 sections (2021) vs **4,405 (2022 harmonized release = current PhysioNet number)** — discrepancy resolved; trigger κ 0.94–0.97 incl. Tobacco; active learning (surrogate classifier + diversity term, α=0.1) lifted current-Tobacco-Status +0.07 ΔF1 and risk-factor frequencies +83–151%; baseline extractor: Tobacco trigger 0.97, Status 0.90 F1; ocad012 top = Microsoft T5-large seq2seq 0.901/0.774/0.889 (A/B/C; paired bootstrap n=10k, note-level) → **Subtask-B cross-domain drop −13pp = pre-registered risk R7 for our cross-corpus eval**; named FN/FP causes: unpopulated templates ("Tobacco use: _") → FPs; vocabulary mismatch ("toxic habits", "nicotine"-as-tobacco) → FNs; multi-event notes degrade performance (density >1 harder); **conflicting intra-note statuses annotated as SEPARATE events** (supports our conflict-flag/precedence Stage-6 design); <2% events span sentences but the verified "quit" cue in an EARLIER sentence failing alone = direct precedent for sentence-merged crops; access = PhysioNet credentialing + UW DUA (two-gated, confirmed); scoring code github.com/Lybarger/brat_scoring. Appendix tables read (2026-10-28): corpus split 3,136/448/896 train/dev/test over 12 annotation rounds (MIMIC+UW mixed; AL rounds 5/9/10/11); AL query-function tuning best "sum/maximum, α=0.1" = 0.794 vs random baseline 0.752 (+0.042 macro-F1, p<0.05) → the +0.07 headline was the current-Tobacco-Status-specific lift; the overall AL gain is modest — reinforces "AL = optional accelerator, never substitute for balanced accrual"; surrogate/extractor hyperparameters recorded (LSTM hidden 100; AL query batch 20, eval batch 100) | [x] full (appendix tables incl.) |
| S-13 | SDOH-NLI paper (arXiv 2310.18431, EMNLP 2023 Findings) | 11 | abstract verified 2026-10-28: NLI formulation — premises (clinical-note sentence segments) ↔ hypotheses (normalized status statements), independently drawn; confirms NOT a smoking-label corpus → E0-format exercising only, never a floor-arm data source | [x] (full schema wording verified at download time) |
| S-14 | Ettin paper/blog (2025) | 5 | paired encoder-decoder suite findings, 32M capacity claims | [x] headline (400M-encoder>1B-decoder classification claim); [ ] not blocking implementation |
| S-15 | HHS de-identification & cloud guidance pages | 0 | Direct GET = 403 (bot-protected). Wayback capture verified 2026-10-28: summary page live, content last reviewed **2025-02-03**, verbatim anchor captured: "the two methods that can be used to satisfy the Privacy Rule's de-identification standard: Expert Determination and Safe Harbor" (OCR guidance). Full-guidance PDF = the Stage-0 signer-trail document; 18-identifier Safe Harbor list is its core. T5 language (cropping ≠ de-identification; cloud egress needs BAA or de-id) stands on guide#2 citations + this anchor | [x] (signed-trail copy to fetch manually in browser for the institution checklist) |
| S-16 | SetFit docs (SBERT) | 5(B) | Verified 2026-10-28: prompt-free few-shot fine-tuning of Sentence Transformers — 8 labeled examples/class competitive with full fine-tuning RoBERTa-large (CR sentiment); works with any sentence-transformer backbone → arm-B few-shot variant (8/class) is definable — but S-9 KOTI full-text evidence places smoking-task few-shot well below the deployment gate (macro-F1 0.35–0.47 at N=10–50 vs 0.85+ full-FT) → fallback-only, priority dropped per F7 | [x] (usage conditional on arm-B outcome) |
| S-17 | Collator/tokenizer offsets doc (HF tokenizers) | 3 | offset mapping for crop construction correctness | [x] standard; [ ] verify in code only |
| S-18 | i2b2-2014 risk-factor dataset docs | later | longitudinal extension viability (heart disease; smoking risk factor labels) | [ ] (Phase 3+) |
| S-19 | UMLS NegEx/ConText (negation scope tooling) | 1,4 | negation handling depth decision | [ ] (only if Stage-2 audit shows negation confusions) |
| S-20 | vLLM ModernBERT support page (comment-thread claim) | 9 | serving engine availability for encoder | [ ] (opt. serving optimization) |
| S-21 | Lybarger et al., JAMIA 2023 ocad073 (PMC10354760) — NLP-augmenting structured SDOH data | 2,8 | ingested via PubMed abstract 2026-10-28. SDOH extractor 0.86 F1; UW case study 225,089 patients / 430,406 social-hx notes: 32% homeless, **19% of current tobacco users**, 10% drug users have the risk factor ONLY in narrative → quantified structured-vs-text information gain (deployment motivation); frame precisely: motivates text extraction, NOT a regex-failure rate | [x] |
| S-22 | Prado et al., BMJ Open 2023 (PMC10124310) — lung-cancer symptoms case-control | peripheral | reviewed via abstract 2026-10-28: smoking-status-matched controls; no extraction methodology our design needs | [x] (peripheral; no follow-up) |
| S-23 | SmokeBERT (Xue et al., JCO Clin Cancer Inform 2025; e2500223, PMC13532985; preceded by medRxiv 2025-06 preprint) | 5 | Verified 2026-10-28 via abstract: fine-tuned **ClinicalBERT** (selected over BERT/BioBERT/MedBERT on CV) for six quantitative smoking-history variables (pack-years, years-since-quit, …); 3,261 sentences CV + external validation 3,191 sentences; holdout F1 0.97 vs rule-based NLP 0.88–0.90 and **gpt-oss-20B 0.72–0.79**; external validation 0.86 vs 0.72–0.79 → (a) zero-shot gpt-oss-20B is weak on quantitative fields ⇒ **teacher scope stays 4-class status; pack-years/quit-dates = sideways scope**; (b) external drop 0.97→0.86 = second R7 datapoint; LCS-eligibility ID: 97% vs 59–97% (≥20 pack-years), 100% vs 60–84% (≤15 yr since quit); companion editorial Tan & Osterman (e2500350) | [x] |
| S-24 | Palmer et al., BMC MIDM 2019 (PMC6657102) — tobacco user registry, multiple behaviors | 2,6 | Verified 2026-10-28 via abstract: ML status classifier **F1 0.90 on BOTH i2b2-2006 (502 notes) and local 758 visit notes**; regex pack-year extraction 91.7% sens / 95.2% spec but incomplete in 25% of cases (packs/day or years-smoked only); **regex cessation-date only 63.2% sens / 94.6% spec** → quit-date is the hard field (corroborates Wright S-11 + SmokeBERT); direct precedent for status-classifier + add-on regex-field architecture on the same i2b2 corpus; informs sideways scope (pack-years) and Stage-6 aggregation | [x] |

## 13. Risk register & kill criteria
| Risk | Signal | Mitigation | Kill criterion |
|---|---|---|---|
| R1 labels too scarce (all DUA slow) after smoke wave | >6 wks no labeled tobacco corpus | SHAC form + synthetic benchmark pipeline | pivot to paper scoped on synthetic+SDOH alternative? (research-only pivot; not planned) |
| R2 crop-label policy too conservative → UNKNOWN-dominant training | UNKNOWN >60% of row labels | re-examine escape rules w/ reviewer; consider context-rich example format | if classifier useless on i2b2-harmonized macro-F1 (<0.75 after correction) — revisit label policy not model size |
| R3 encoder can't hold CURRENT-recall at cost | Current→Never safety error >2pp vs ground truth on challenge set | abstention lane thresholding; expanded-context | revisit 4-way head → per-class heads (T6 ablation) |
| R4 budget blow-out | cumulative spend > $120 before first fine-tune wave | stick to zero-DUA + local MPS first wave; only then paid wave | pause, re-plan |
| R5 TRL/transformers version drift regresses starter | smoke fails on API | pin working env; fork guide scripts | n/a (measured failure → fix pins) |
| R6 teacher route adopted by accident into eval | teacher score used as truth anywhere | locked human test only; eval gates in code | any occurrence = restart eval round |
| R7 cross-corpus degradation | trained-arm macro-F1 drops >8pp between public-corpus evals and local/synthetic evals (verified precedent: n2c2/UW Subtask B — 0.901 in-domain → 0.774 cross-domain, −13pp MIMIC→UW) | per-corpus locked tests (Stage 8); doc-type/section-aware data selection (S-7 lesson); E2 dual-format check | if local in-domain data can't be brought above the deployment gate by teacher-assisted bootstrapping (H4), escalate to local annotation before any scope expansion |

## 14. Zero-DUA smoke experiments (E0–E4) — pre-implementation validation
- E0 (corpus format + retrieval sanity): SDOH-NLI premises → run extract_windows.py → trigger-scan stats (does regex over-fire? merge correct? offsets roundtrip?) Metrics: crops/note, token distributions vs predicted smoke budget, merge ratio.
- E1 (encoding stack smoke): load BioClinical-ModernBERT-base on MPS small batch, forward+backard on toy 512-token batch; record MPS R; settle version pin.
- E2 (end-to-end dry run): **the Wright synthetic benchmark (3,000 notes, CC-BY-4.0, github.com/vclic/smokingbenchmark — located 2026-10-28)** → full pipeline (triggers→crops→C-config 3-epochs toy→aggregate→report queue) to validate the code path + metrics plumbing before touching any real labeled corpus. Cautions: LLM-adjudicated references (their limitation) → treat as pipeline-comparability set, never gold; include BOTH template-shorthand and LLM-prose notes (verified difficulty difference); label mapping per §9-4.3, and run the one-year-rule sensitivity slice on it.
- E3 (arm A discipline): TF-IDF on same synthetic slice — confirm floor arm not embarrassing (validates report template).
- E4 (cost baseline): benchmark local R and simple call rate ⇒ provisional $/M table at H=T4/L4/A10G rates (verified pricing) — sanity-check H0-ARCH magnitude before real spend.
Gate: all pass → implementation wave 1 begins on real tobacco data (SHAC→i2b2 when DUA lands).

## 15. RATIFIED decisions (2026-10-28, user)
1. **Crop granularity ✍ RATIFIED ★**: sentence-aligned merged windows w/ ±100 default radius + evidence-density ordering.
2. **One-year rule ✍ RATIFIED ★**: adopt i2b2's (quit <1 yr ⇒ CURRENT) for v1; consequence documented ("quit 4 months ago" = CURRENT).
3. **Teacher route ✍ RATIFIED (user selection, modified from ★)**: **cloud-on-public-data first** — cloud APIs (cheapest experiments) restricted to public/de-identified corpora; privately-hosted gpt-oss-20B becomes the second-stage route when teacher labeling approaches real-corpus or cost scales (re-evaluated at Phase 3 gate). ePHI wall unchanged: no raw ePHI to any cloud API regardless.
4. Challenge-split location: per-corpus locked tests + eventual local-locked test (pending institutional access) — unchanged.

## 16. Traceability (evidence anchors)
- i2b2 rules/counts/κ/α: S-1 (PMC2274873, verified word-for-word).
- 2026 cost + accuracy datapoints: S-8 (JAMIA Open), S-11 (medRxiv), S-7 (MSEP), KOTI (S-9), NPJ teacher (S-13-adjacent), arXiv 2411.03004.
- ModernBERT/TRL/MiniLM: S-4/S-5/S-6 + MiniLM card (verified 256wp/128-train, Apache-2.0).
- HF Jobs pricing table verified 2026-10-28 (T4 $.40 / L4 $.80 / A10G $1 / L40S $1.80 / A100-80 $2.50 / H200 $5 / RTX-PRO6000 $2.75).
- SHAC release form + SDOH-NLI license/ungated: tasksource card + Lybarger README (verified).
- SHAC full texts: JBI 2021 (PMC7856628) event schema/AL/extraction tables + n2c2/UW ocad012 (PMC10354764) subtasks/tables — verified 2026-10-28.
- Wright benchmark URL: Data Availability section of the preprint (github.com/vclic/smokingbenchmark, CC-BY-4.0).
- Information-gain motivation: Lybarger ocad073 (PMC10354760) — 19% current-tobacco narrative-only.
- Label-policy conflicts + per-corpus mappings: §9-4.3 register.
