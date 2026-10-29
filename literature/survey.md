# Literature & Verification Survey
Compiled 2026-10-28 (outer-loop research pass #1, per user request "do one more research").
All claims below verified against primary sources by this agent unless marked otherwise.

---

## 1. i2b2-2006 smoking challenge — VERIFIED from the primary paper

Source: Özlem Uzuner, Ira Goldstein, Yuan Luo, Isaac Kohane.
*Identifying Patient Smoking Status from Medical Discharge Records.*
J Am Med Inform Assoc. 2008 Jan-Feb; 15(1):14–24. doi:10.1197/jamia.M2408.
PMCID: PMC2274873, PMID: 17947624 (fetch of full PMC text, 2026-10-28).

### Corpus facts (exact)
- 502 de-identified Partners HealthCare discharge summaries = **398 train + 104 test** (both external reports were correct).
- Underlying annotation: **928 records**, two pulmonologists, *textual judgments only* (explicitly stated facts; "intuitive" judgments were abandoned after κ=0.45 vs 0.84).
- 63 records dropped for unresolvable annotator disagreement; **363 Unknown-class records dropped** to rebalance — i.e., the natural corpus has a huge Unknown majority.
- Test distribution: Past 11 / Current 11 / Smoker 3 / Non-Smoker 16 / Unknown 63.
- Train distribution: Past 36 / Current 35 / Smoker 9 / Non-Smoker 66 / Unknown 252.

### Label definitions (verbatim rules — these are the canonical source for our harmonization)
| Category | Rule |
|---|---|
| Past Smoker | explicitly smoked ≥1 yr ago AND not smoked for ≥1 yr. "Past smoker" with no temporal qualifier → Past *unless* text says quit <1 yr ago. |
| Current Smoker | explicitly smoked within past year (the "one-year rule"). "Current smoker" with no qualifier → Current *unless* text says quit >1 yr ago. |
| Smoker | ever used, timing unresolvable (Current or Past, insufficient info). |
| Non-Smoker | never smoked. "No tobacco" → Non-Smoker. **Secondhand exposure → Non-Smoker.** **Marijuana smoking → does not count.** |
| Unknown | no smoking mention at all. Indecision between Current/Past does NOT belong here. |

Annotated example confirming the one-year rule quirk: *"She quit smoking four months ago."* → **Current Smoker** (quit <1 yr ago counts as Current).

### Baseline-era results to demystify the benchmark
- Best system (Clark_3, SVMs over phrase-level smoking references + rule aggregation): **micro-F1 0.90, macro-F1 0.76**.
- 23 runs from 11 teams; 12 runs >0.84 micro.
- Macro<<micro gap is structural: Unknown (61% of test, mostly trivially empty) inflates micro; Smoker class has **3 test records**.
- Success hinged on exactly our design: "discharge summaries express smoking status using a limited number of textual features (…'smok', 'tobac', 'cigar', Social History…)."
- 2006 winning architecture = **sentence/phrase-level extraction, then classification, then document-level rules** → direct precedent for our snippet→note aggregation pipeline.
- Data location per the paper: "obtained from i2b2.org" (now the n2c2/DBMI portal).

### Access process — BLOCKED, not resolved
- `portal.dbmi.hms.harvard.edu` **TLS certificate has expired** (confirmed via raw fetch: `certificate has expired`); the legacy `www.i2b2.org/NLP/` site returns HTTP 500. The Wayback API rate-limited (429).
- ⇒ The user's DUA sign-up can only be done manually (browser) once their site is back, or by emailing n2c2 support. Not a research blocker: we proceed on SDOH-NLI (below) meanwhile.
- UNVERIFIED remaining: current DUA cost/fee and turnaround time. Do not rely on memory — user must confirm on the portal directly.

## 2. Datasets cross-check (fresh verification)

| Dataset | Status (verified 2026-10-28) |
|---|---|
| SDOH-NLI | ✅ HF `tasksource/SDOH-NLI`, **license cc-by-4.0, ungated**; ~30k pairs; original arXiv 2310.18431 (Lelkes et al. 2023, Google). Third mirror `presencesw/sdoh-nli` has explicit splits **21,090 + 4,033 + 4,212 = 29,335** → resolves guide#2's "29,635 vs 29,335" mismatch (mirror sums to 29,335; treat original card's ~30k as rounded). Lowest-friction public source = confirmed. |
| SHAC | ✅ **Released and accessible** (README updated 2025-07-07): "...publicly available through a data use agreement and PhysioNet credentialing" via a Google data-access form (forms.gle/baupGvG2L8Dc6mZc6). Two-gated as guide#2 predicted; README contradiction resolved. Email-based access also possible via Lybarger (GMU) / Yetisgen (UW). |
| i2b2-2006 | ✅ corpus facts fully verified from the 2008 paper; ⚠️ portal access blocked by expired TLS cert (see §1). |
| MIMIC-III/IV-Note | (guide#2 scale claims consistent with evidence reviewed; not re-counted — not on critical path) |

## 3. Fresh literature sweep 2019–2026 (via arXiv API + PubMed E-utilities; web search unavailable, these routes worked)

### Directly on-topic (must-reads)
| ID | What it shows | Value to us |
|---|---|---|
| **PMID 41542317** — Dávila-García et al., *JAMIA Open* 2026;9:ooaf182. Lightweight open-source LLMs vs cTAKES on tobacco smoking status | 250 adult discharge summaries (BJC HealthCare), Smoker/Never/Unknown, reader κ=0.91. gpt-oss-20B **weighted-F1 0.99** vs cTAKES **0.97** (non-inferior, p<0.021); 6 LLMs tested (Llama-3 1B–70B, gpt-oss-20B, MedGemma-27B); 2.5–14.5 s/note, weights 2.3–47.3 GB. **Zero-shot, no fine-tuning.** | This is the study guide#2 paraphrased ("κ=0.91, 250 summaries") — now verified. Nuances found: labels were 3-class (current+former merged — confirms that gap); hospital-grade classifier cTAKES is already at 0.97, so our fine-tuned *encoder* must at least beat ~0.97-class systems to justify replacement **on accuracy alone** — but neither cTAKES quality nor 2.5–14 s/note LLM latency says anything about $/million, which is our deciding metric. Also: gpt-oss-20B → strong candidate for **teacher** (H4) if open-weights: yes, gpt-oss-20B is Apache-2.0. |
| **PMID 41923446** — Dong et al., *JMIR Med Inform* 2026;14:e77409. MSEP sentence-classification pipeline (Rennes UH) | 12,119 hand-annotated sentences, 6 statuses incl. **smoking**; classes presence/absence/**unknown**; fine-tuned CamemBERT **macro-F > 0.94 on 5/6**; a *rule-based* extractor beat the learned one on the sparsest class (0.94 vs 0.73 — family history); annotation cost 1.2–2.9 s/sentence; full pipeline reproducible in 8 h. | **Strongest external validation of our exact architecture** (sentence/snippet-level classifier + local institutional adaptation). Also directly supports: (a) our rule/TF-IDF floor arm may genuinely win on rare classes; (b) separate per-class assessors (presence/absence/unknown) is a viable framing of our 4-way task; (c) annotation budget reality: 12k sentences was feasible at one institution. |
| **PMID 42818476** — Wright, Liu, Wright. medRxiv preprint 2026-09-24 (not peer-reviewed). Smoking-history extraction for lung-cancer-screening CDS | Synthetic **shareable** 3,000-note benchmark (prog. ground truth), 3 difficulty conditions. Structured-judgment model (TypeSafe Jev) vs Claude Haiku 4.5/Sonnet 5, GPT-6 Luna/Sol, identical schema. Eligibility-correct: Jev 99.1/99.9/94.4 vs LLMs 97.8–100/98.4–99.9/98.1–99.8. **Cost per 1,000 notes: Luna $0.12–0.21; Jev $0.61–0.64; Sol $2.44–4.22; Haiku $3.37–4.43; Sonnet $3.76–6.61.** Latency: Jev 0.45–1.21 s median vs LLMs 1.71–3.50 s. | **Best current cost datapoint for a comparable task.** Per-million extrapolation: cloud LLM full-note = **$120–6,610/M notes**; a local structured approach ~$610/M. Context for our encoder target: <100M model on a single small GPU should land at ~$1–10/M snippets — 100×–1000× cheaper per unit, IF accuracy survives the crop. Synthetic benchmark is publicly shareable → usable as a clean smoke-test for our retrieval-vs-classifier eval harness. Caveat: single-author-affiliation preprint + synthetic data. |
| **PMID 40611214** — Alleva et al., *BMC Med Inform Decis Mak* 2025;25:247 (Mount Sinai). KOTI: keyword-optimized template insertion | Few/zero-shot clinical note classification incl. smoking status; placing the prompt template **next to the keyword-bearing sentence** instead of tail-truncating input: encoder gains +8% (ClinicalBERT), **+24% (GatorTron)** F1; balanced sampling helps few-shot; decoder results inconsistent (BioGPT +19%, ClinicalT5 −18%). | Directly actionable for our crop-prompt construction and for the retrieval study: **evidence-position in the input matters**, especially near the context limit — reinforces guide#2's "keep boundaries visible + don't tail-truncate". Adopt as a candidate improvement to `src/train_encoder.py` data prep (rank snippets so the prompt/instruction sits adjacent to the highest-evidence sentence). |
| **PMID 42174984** — Sorayaie Azar et al., *Stud Health Technol Inform* 2026;336:904-908. Multitask transformers on Danish EHR | BioBERT (smoking status acc 97.9%, F1 86.3%), XLM-R (pack-year F1 92.0). Multitask (status+pack-year+symptoms) framing; LIME-explained. | External-language evidence that small clinical multitask encoders carry smoking status well; supports encoder-arm + optional multitask heads as a later extension. Conference-scale paper (short) → weight low. |
| **PMID 42175026** — Dong et al., *MINE platform*, *Stud Health Technol Inform* 2026;336:1064-1065. | Expert-guided extraction platform; early smoking-status/diabetes/hypertension F > 0.95. | Two-paragraph abstract; trend confirmation only. |
| **PMID 42251136** — Zou et al., *NPJ Precis Oncol* 2026. On-prem LLMs for 30 prognostic variables incl. smoking history (H&N oncology) | 1,360 reports/882 patients; Llama-3.3-70B zero-shot **98.6% F1** (vs 3-oncologist majority); best smaller: gpt-oss-20.9B 89.4%, MedGemma-27B 88.5%; 53 s/report → Cox models validated w/ 1000-bootstrap. | Supports **H4 teacher design**: private-hosted open-weight teacher (Llama-3.3-70B class, or gpt-oss-20B) extracts smoking among many variables at high F1 without any ePHI leaving campus. |
| **arXiv 2411.03004** — Lee & Wood-Doughty (NeurIPS-AIM-FM 2024). LLM-classified smoking status + measurement-error correction | LLM predicts smoking status as an unobserved confounder from MIMIC notes; categorical-error correction of classifier output recovers unbiased causal effect estimates (TTE-on-mortality case study). | Downstream look-ahead: if our labels feed causal/observational analyses, classifier mismeasurement can be corrected by a onetime validation sample — OUR pipeline should produce the per-class confusion matrices that enable exactly this. Adds a reason to never silently merge unknown into other classes. |
| **arXiv 1910.13664** — Mulyar et al. (2019). BERT doc phenotyping | Early 2019-era: BERT pass over full notes, competitive on phenotyping incl. smoking status. | Historical only. Predates ModernBERT-era truncation-efficiency work we need. |

### Not adopted (off-topic after abstract review)
Periodontitis LLM-vs-NN (42356005), perinatal surrogacy (42331008), Framingham social networks (42310578), ICI-ILD (42262534), WTC exposure LLM (42168810), plus 5 lifestyle/epidemiology papers using smoking status as an *input variable* only — not method-relevant.

## 4. Compute pricing — verified snapshot 2026-10-28

HF Jobs (billing per-minute; charged while Starting/Running; default timeout 30 min → always set longer):
T4-16GB $0.40 · T4-medium $0.60 · L4-24GB $0.80 · L40S-48GB $1.80 · A10G-24GB $1.00 · A100-80GB $2.50 · H200-141GB $5.00 · RTX PRO 6000-96GB $2.75 · CPU XL $1.00. Exposed ports +$0.01/h. Matches guide#2's snapshot; new items beyond it: L40S, H200, RTX PRO 6000 tiers.
(λ/Modal/RunPod/Azure deferred — re-verify at purchase time in Phase 3.)

## 5. Implications synthesis → hypothesis impact

1. **H0-ARCH gains external support & a sharper bar.** MSEP (CamemBERT sentence-level, macro-F 0.94 deployed) + i2b2-2006's winner (sentence-level→rules) both validate encoder/snippet-first; but JAMIA-Open-2026 sets an accuracy bar of ≈0.97 F1 (cTAKES) that our deployed encoder must match for full-note tasks — our contribution is doing that on **crops at million-crop economy**, which no verified study measures.
2. **H1-DATA unaffected; canonical i2b2 rules now available verbatim** (§1) for harmonization: marijuana-excluded, secondhand→never, one-year rule, Smoker→UNKNOWN-with-reason. These map cleanly onto our policy — including that i2b2's "Past/current with unqualified assertion + contradicting quit-date" exceptions match guide#2's ruling table, which can now be cited to a clinical reviewer.
3. **H2-CONTEXT gains a tool**: KOTI's +8–24% F1 from *placing the instruction beside the evidence* is a cheap ablation to add to the retrieval study.
4. **H4-TEACHER candidates widened**: gpt-oss-20B (Apache-2.0, 0.99 F1 tobacco zero-shot, 20B) is now the leading private-hosted teacher candidate; Llama-3.3-70B the heavy option.
5. **Cost ceiling set**: strongest cloud LLM full-note baseline ≈ $120–6,610/M notes (2026 numbers). Our encoder target: ≥10× cheaper per snippet at no accuracy loss — articulable, falsifiable, and consistent with H0.
6. **New falsifiable check added**: against the shareable synthetic LCS benchmark (Wright medRxiv), our crop-based system can be evaluated head-to-head with cloud LLM full-note outputs on the same notes — cheap, zero-DUA, useful early signal.

## 6. Open/remaining
- n2c2 portal DUA page: retry when their TLS cert is fixed (user can also sign manually).
- λ/Modal/RunPod/Azure live prices at purchase time.
- ModernBERT-base (non-clinical) card skim — pending, not critical.
- LM/perf verification of starter scripts still pending data acquisition.
