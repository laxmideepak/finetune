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

(none yet — no data; Phase 1 verification + retrieval/type audits precede training)

## Patterns and Insights

(none yet)

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
