# Research Findings

## Research Question

What is the smallest, most accurate, most economical open-source model and training strategy for classifying smoking status (CURRENT / FORMER / NEVER / UNKNOWN) from regex-extracted ~±100-token clinical note snippets, at million-snippet scale, with clinical reliability?

## Current Understanding

(borrowed from two external expert reports; see research/ for verbatim sources — our own evidence accumulates below)

- **Task definition** (binding): note-level smoking status at the note's reference time. NOT longitudinal patient history. "Note says unknown" is a label; "model is uncertain" is a routing signal for abstention/review/expanded context.
- **Adaptation direction**: encoder-first. Fully fine-tuned clinical encoder (~150M BioClinical-ModernBERT) + 4-way head is the production candidate; small generative LLMs (Qwen3-0.6B LoRA first, then Qwen3.5-0.8B) are challenger/annotation-assistant/fallback arms. TRL `completion_only_loss` only for the decoder arm.
- **Label policy core** (to be ratified with a clinical reviewer): CURRENT = explicit ongoing combustible-tobacco use; FORMER = explicit prior use + documented cessation (decide the recent-quitter rule deliberately); NEVER = explicit lifetime non-use; UNKNOWN = missing/ambiguous/third-party-only/out-of-scope/conflicting. "Denies smoking" alone stays UNKNOWN (cannot separate FORMER vs NEVER) unless paired with explicit past-smoking evidence (→ FORMER).
- **Dual targets**: snippet label (what the crop supports → may be context-insufficient/UNKNOWN) + full-note label (for pipeline evaluation). Never labels a small window with the full-note label when supporting evidence lies outside it.
- **Datasets reality**: public 4-class clinical smoking gold data is thin and DUA-gated (i2b2-2006: 502 docs, 398/104 splits, minority classes ~11/11 in test; i2b2-2014: 1,304 longitudinal notes; SHAC 4,405 sections, access murky; SDOH-NLI CC-BY-4.0 low-friction; MIMIC pools for annotation). Local gold set is the deployment priority (pilot 200–300 → dev 2k → frozen test 1k at natural prevalence → challenge 200–500).
- **Known gap we fill**: no controlled comparison exists for this exact setting (4 labels + ±100 crops + patient-disjoint eval + cost-per-million). That benchmark is the project's main novel deliverable.
- **Cost model**: $/million ≈ 1e6·H/(3600·R) with all-in rate H and measured throughput R; decode arms count prompt tokens + bounded label output; pipeline calls/million ≈ 1e6·q·k.

## Key Results

(no training runs yet — verification-phase results below, details in literature/survey.md)

1. **i2b2-2006 verified from the primary paper** (Uzuner et al., JAMIA 2008, PMC2274873). Both external reports were accurate (502 = 398+104), and we recovered the *exact* rules both reports only summarized: one-year rule (quit <1 yr ago → Current Smoker), Smoker → timing-unresolvable, marijuana excluded, secondhand → Non-Smoker, "No tobacco" → Non-Smoker. Test is tiny and Unknown-majority (63/104); Smoker has 3 test records → macro-F1 on this benchmark is structurally harsh. Best 2006 system = sentence-level extraction + rules (0.90 micro / 0.76 macro) — direct precedent for our snippet→aggregate architecture.
2. **Fresh 2024-2026 survey found directly on-topic work** (all verified via abstracts):
   - JAMIA Open 2026 (Dávila-García): gpt-oss-20B zero-shot tobacco extraction F1 0.99 vs cTAKES 0.97 (250 notes, κ=0.91, Smoker/Never/Unknown) — this is the study guide#2 cited; now first-hand. Raises the accuracy bar to ≈0.97-class for any deployed replacement, and makes **gpt-oss-20B the leading open-weights teacher candidate**.
   - JMIR Med Inform 2026 (MSEP, Rennes): sentence-classification pipeline w/ presence/absence/unknown classes across 6 conditions incl. smoking; CamemBERT macro-F >0.94 on 5/6; **rule-based beat learned extractor on the sparsest class** — validates our sentence-level pipeline design + mandatory rule/TF-IDF floor arm, in real institutional deployment conditions (12,119 annotated sentences).
   - medRxiv 2026-09 (Wright): synthetic shareable 3,000-note smoking-extraction benchmark; **cost per 1,000 notes: $0.12–0.21 (cheapest cloud LLM) to $3.76–6.61 (Claude Sonnet 5)** → per-million $120–6,610 sets the 2026 cost ceiling our encoder must beat 10×+.
   - BMC MIDM 2025 (KOTI, Mount Sinai): placing the template **beside the keyword-bearing evidence** yields +8–24% encoder F1 in zero/few-shot — cheap ablation to add to our retrieval study.
   - NPJ Precis Oncol 2026: on-prem Llama-3.3-70B 98.6% F1 extracting smoking among 30 variables — supports private-hosted teacher route.
3. **Datasets confirmed**: SDOH-NLI (tasksource, CC-BY-4.0, ungated, splits sum 29,335) is the lowest-friction public corpus; SHAC is now openly accessible (DUA + PhysioNet form, README 2025-07-07). n2c2 portal blocked by its own expired TLS certificate — user must sign DUA manually.

## Patterns and Insights

- **The field already converged on our architecture**: the 2006 i2b2 winner, MSEP-2026 deployment, and the 2026 cTAKES-comparison all classify *at sentence/snippet level* then aggregate — the open question is not "does this work" but, per our hypotheses, "how small and how cheap can the classifier be while keeping current-smoker recall."
- **Cost economics sharpen H0-ARCH**: verified cloud-LLM full-note costs ($120–6,610/M notes) vs our encoder target (~$1–10/M snippets) formalizes the ≥10× claim H0-ARCH must demonstrate.
- **i2b2-2006 harmonization can now be cited, not asserted**: our label policy's escape rules (quit<1yr → CURRENT etc.) match the challenge's original pulmonologist rules verbatim — useful for the clinical-reviewer ratification step.

## Lessons and Constraints

- **Local dev (this machine)**: Apple M4 16GB MPS. Past verified OOM lesson: fp32 LoRA at bs=8/512 tokens OOMs; use bf16 + gradient checkpointing + small batches, or run on rented GPU. `peft`/`accelerate`/`datasets` must be reinstalled when prototyping locally (were uninstalled in an earlier cleanup; transformers pre-existed).
- Starter scripts from guide #2 in `src/` are syntax-checked but NOT executed (no models/CUDA runtime check yet) — smoke-test on public data before trusting.
- HIPAA: cropping ≠ de-identification; BAA + approved environment for ePHI; no raw notes in logs/trackers; no auto Hub uploads; DUAs reviewed even for de-identified data.
- Provider/GPU prices are snapshots — re-verify at purchase time.

## Open Questions

- Exact i2b2-2006 access process/cost/time (user has signed nothing yet).
- Can i2b2-2014 longitudinal smoking risk-factor labels adapt to our note-time definition?
- Retrieval recall on their regex-negative notes (needs a small audited sample from THEIR corpus, inside their environment).
- Teacher choice (cloud on de-identified public only vs private-hosted open model) — decide after seeing public-data results.
- Whether BioClinical-ModernBERT weights license permits their institutional deployment policy review (MIT-claimed; verify card).

## Optimization Trajectory

(no training runs yet — research/verification phase)
