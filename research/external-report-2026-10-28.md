# External Researcher Report — verbatim source (received 2026-10-28)

> Source: report from another AI researcher as pasted by the user. Claims marked "verify" in
> MASTER_CONTEXT.md section 9 were not independently verified by us. Design guidance and the
> TRL training skeleton are treated as recommendations, not evidence.

---

For your task, start with a small supervised classifier—not a large medical chatbot. My recommended first LLM experiment is Qwen/Qwen3-0.6B with LoRA, benchmarked against a much smaller MiniLM/SetFit classifier; also test the newer Qwen3.5-0.8B before choosing your final model. These are candidates, not proven winners on your notes. (huggingface.co/Qwen/Qwen3.5-0.8B)

Your regex → short context windows → four-label prediction architecture is sensible. Similar smoking-status systems have successfully used smoking-related passages rather than entire notes, but retrieval quality, label definitions, and local evaluation matter more than choosing the newest model. (academic.oup.com/jamia JAMIA 15(1):14 smoking articles)

## Recommended starting stack

| Component | Recommended starting choice | Why |
|---|---|---|
| Prediction target | Patient's smoking status as documented in this note, at the note's reference time | An explicit task definition avoids mixing note classification with longitudinal patient-history reconciliation |
| Retrieval | Regex triggers, merged token windows, section headers, original order | A refinement of your existing approach; keyword-based passage selection has precedent in smoking-status research. |
| Lightweight baseline | MiniLM + SetFit | MiniLM ~22.7M params; SetFit supports few-shot fine-tuning of sentence-transformer classifiers. |
| First generative model | Qwen3-0.6B + LoRA | Small causal LM, Apache-2.0 license, conventional Transformers training path. |
| Newer challenger | Qwen3.5-0.8B | Its official card explicitly lists task-specific fine-tuning among intended uses. |
| Training libraries | Transformers + Datasets + PEFT + TRL | TRL supports prompt–completion training, completion-only loss, and LoRA integration. |
| Clinical deployment | Institution-approved private environment | Processing ePHI in cloud requires appropriate agreements and safeguards. (hhs.gov HIPAA cloud guidance) |

Do not commit to an LLM until it beats your encoder baseline on the same held-out patients. That is an experimental recommendation, not a claim that either architecture will necessarily win.

## Define the labels first

The biggest hidden problem is that "not currently smoking" does not mean "never smoked." The original smoking benchmark separates current smokers, past smokers, and smokers whose timing is unclear. (JAMIA)

Proposed annotation policy (for this project, not a universal clinical standard):

| Output | Proposed definition | Important distinction |
|---|---|---|
| `current` | Patient documented as currently smoking at the reference time | Occasional smoking still counts if in scope |
| `former` | Prior smoking established, and cessation explicitly documented | Decide how to handle someone who quit very recently |
| `never` | Explicit lifetime non-smoking status | Do not automatically assign from "denies current smoking" |
| `unknown` | Evidence does not establish one of the other three states | Includes missing info, unresolved timing, unresolved contradictions |

Choose your cessation rule deliberately. The original i2b2 smoking challenge considered smoking within the preceding year "current," whereas its "past smoker" category generally required at least a year without smoking; a model trained on those labels inherits that convention unless you relabel. (JAMIA)

Settle these scope questions in the annotation guide:
- Whether the target is cigarette smoking, combustible tobacco, or all tobacco use
- Whether cigars and pipes count
- Whether vaping and smokeless tobacco are separate variables
- How to exclude marijuana smoking and secondhand exposure
- How to distinguish the patient from relatives
- How to handle conflicting statements and copied historical sections

For auditing, keep a richer internal output while exposing only the four labels:

```
status: current | former | never | unknown
evidence_offsets: positions in the original note
unknown_reason: optional reason
conflict_flag: true | false
```

Keep "the note says unknown" separate from "the model is uncertain." The former is a label; the latter should trigger abstention, review, or expanded-context processing.

## Datasets

### i2b2/n2c2 smoking challenge
Most directly relevant established dataset. Harvard n2c2 2006 project page (portal.dbmi.hms.harvard.edu/projects/n2c2-2006/) lists a smoking-annotated collection of 502 de-identified discharge summaries; the associated publication explains annotation policy and classification task. (PMC2274873)

Not a clean interchangeable four-label dataset:

| Original category | Mapping to your task |
|---|---|
| Current smoker | Map only after reconciling the original one-year rule |
| Past smoker | Usually maps to `former`, subject to your cessation definition |
| Smoker | Current vs former unspecified; do NOT auto-map to current |
| Non-smoker | Review annotation definition before equating with lifetime `never` |
| Unknown | Usually maps to `unknown` |

Recommended use: obtain through official access process, reproduce its benchmark separately, then harmonize labels. Do not silently mix original labels with locally annotated labels using different definitions.

### SHAC / n2c2 SDOH
Social History Annotated Corpus contains clinical social-history annotations including tobacco events; underpinned the 2022 n2c2/UW SDOH challenge (github.com/Lybarger/sdoh_extraction). Useful for event-level tobacco attributes; converting to note-level four-class needs an explicit aggregation policy. Access caveat: the project page contains a dated announcement promising a future release; a completed generally accessible release could not be verified — do not plan around immediate SHAC availability.

### MIMIC-IV-Note
Provides real clinical notes incl. 331,794 discharge summaries, but is a source corpus, not a ready-made four-class smoking benchmark (physionet.org/content/mimic-iv-note/2.2/). Access requires credentialing, training, and a signed DUA. Annotate a subset or create reviewed teacher labels within data-use restrictions.

### Local notes
For deployment in your institution, make a local gold-standard dataset the priority. Proposed annotation plan (a proposal, not an evidenced minimum):

| Stage | Proposed allocation | Purpose |
|---|---:|---|
| Guideline pilot | 100–200 notes | Find disagreements, fix label definitions |
| Initial dev corpus | ~1,000–2,000 notes if feasible | Baselines + learning curves |
| Locked test | patient-disjoint sample | Final evaluation |
| Challenge set | deliberately hard cases | Negation, cessation, relatives, contradictions, missing triggers |

Use a naturally sampled test set for real-world performance estimate; keep enriched challenge set separate.

Hugging Face caveat: no verified unrestricted, well-documented, four-label clinical smoking dataset there. Search returned unrelated datasets; a matching name is not evidence of suitable provenance or labels.

## Improve context retrieval

±100 tokens is a starting hyperparameter, not an established optimal context length.

Pipeline:
```
Original note → find triggers → tokenizer-based windows → merge overlaps
(preserve source order) → retain section headers and dates → classify all
retained evidence for the note → return status + evidence offsets
```

Keyword-based passage identification has precedent in smoking-status research (pubmed.ncbi.nlm.nih.gov/17947624/).

Candidate trigger pattern (for testing, not a validated final regex):

```python
pattern = r"""
(?ix)
\b(
    smok\w*
  | tobac\w*
  | cigar\w*
  | nicotine
  | pack[\s-]?years?
  | packs?\s*(?:/|per)\s*day
  | ppd
)\b
"""
```

Implementation recommendations:
- Use tokenizer offsets, not whitespace-word counts
- Collect all relevant windows, not just the first match
- Merge overlaps
- Keep boundaries visible when concatenating distant sections
- Preserve negation and cessation clauses; expand to sentence boundaries where needed
- Treat broad triggers as retrieval hints, not classification rules
- Audit regex-negative notes to measure misses

Crucial training rule: do not give a tiny window the full-note label when the evidence supporting that label lies outside the window. Either label visible evidence, expand context, or train one example containing all retained windows.

Retrieval-compression validation experiment (run before fine-tuning):

| Candidate | What to compare |
|---|---|
| ±50 tokens | more aggressive compression |
| ±100 tokens | proposed baseline |
| ±200 tokens | additional context |
| Sentence/section-aware extraction | linguistic/section boundaries |
| Union of windows + social history | broader evidence coverage |

Measure evidence retention, contradiction retention, retained tokens, final classification quality.

## Which model to fine-tune?

| Candidate | Role | Main consideration |
|---|---|---|
| MiniLM-L6-v2 + SetFit | smallest practical baseline | ~22.7M params; check context-length handling for merged windows |
| Bio_ClinicalBERT + classification head | clinical encoder baseline | encoder checkpoint, not a ready smoking classifier |
| Qwen3-0.6B | recommended first LLM experiment | Apache-2.0; straightforward causal-LM implementation |
| Gemma 3 270M IT | very-small generative challenger | ~268M params; gated access, Gemma-specific license |
| Qwen3.5-0.8B | newer small-model challenger | positioned for task-specific fine-tuning; includes a vision encoder, despite text-only task |
| Qwen3.5-2B | optional larger challenger | benchmark only if smaller models miss quality target |

Order of experiments: MiniLM/SetFit → Qwen3-0.6B LoRA → Qwen3.5-0.8B → larger model only if needed.

No evidence from your corpus yet that a medical-pretrained model or the newest Qwen wins. Choose based on held-out macro-F1, current-smoker recall, calibration, latency, and memory — not model name.

## Fine-tuning methods

| Method | Plain language | Recommended role |
|---|---|---|
| SFT | train on input + correct output | main training objective for the 4-label task |
| LoRA | train adapter params | recommended first generative-model training method; TRL/PEFT supported |
| QLoRA | adapters + quantized base | consider if memory limited; adds config/compat considerations |
| SetFit | fine-tune sentence-transformer classifier with limited labels | strong lightweight baseline |
| DSPy | program/optimize LM prompts (and weights in some workflows) | improving teacher instructions/examples using dev data; not required for first student |
| Distillation | train small student from teacher outputs | after validating a stronger teacher |
| Active learning | prioritize uncertain/disagreement examples | direct annotation effort |
| Preference/RL | optimize preferences/rewards | not a sensible first step |

### Where distillation fits
```
Human-labeled dev data → validate stronger teacher → teacher labels additional
permitted local notes → review disagreements/difficulties → train small student
on gold + reviewed teacher labels → evaluate on untouched human-labeled test patients
```
Start with hard-label distillation. Do not use teacher agreement as the definition of correctness — both can share annotation-policy errors.

### Where DSPy fits
Initial role: improving teacher's instructions and examples using development data. Keep locked test set outside prompt optimization, teacher selection, and student training. Prefer a simple well-evaluated pipeline to added orchestration frameworks before trustworthy labels exist.

## Beginner training recipe

LoRA SFT on prompt–completion examples, loss only on the answer (TRL completion-only loss). Suggested initial settings (not optimized): Qwen3-0.6B; one lowercase label output; max sequence 1024 tokens; LoRA r=16 alpha=32 dropout=0.05; LR 1e-4; 2 epochs; effective batch 16; packing off; select on validation classification metrics, not training loss alone.

Dataset format: patient-disjoint train.jsonl / validation.jsonl rows with `text` (merged evidence) and `label` (current|former|unknown|never). Maintain patient/note IDs, source offsets, provenance in secure preprocessing records. Split by patient BEFORE creating windows or teacher-labeled derivatives.

Training skeleton (unexecuted starting skeleton, TRL SFT + PEFT LoRA; see source chat for full code):

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

model_id = "Qwen/Qwen3-0.6B"
labels = {"current", "former", "unknown", "never"}
max_length = 1024

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

instruction = (
    "Classify the patient's smoking status at the note's reference time. "
    "Use only the supplied evidence. Return exactly one label: "
    "current, former, unknown, never. "
    "Former requires documented prior smoking and cessation. "
    "Never requires explicit lifetime non-smoking. "
    "Use unknown for insufficient or unresolved conflicting evidence."
)

def prepare(row):
    assert row["label"] in labels
    prompt = tokenizer.apply_chat_template(
        [{"role": "user",
          "content": instruction + "\n\nEvidence:\n" + row["text"]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    completion = row["label"] + tokenizer.eos_token
    n_tokens = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
    if n_tokens > max_length:
        raise ValueError("Example exceeds context budget; revise retrieval, rather than silently truncating evidence or labels.")
    return {"prompt": prompt, "completion": completion}

data = load_dataset("json", data_files={"train": "train.jsonl", "validation": "validation.jsonl"})
data = data.map(prepare, remove_columns=data["train"].column_names)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16 if use_bf16 else torch.float16)
model.config.use_cache = False

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    peft_config=LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear",
                           r=16, lora_alpha=32, lora_dropout=0.05, bias="none"),
    args=SFTConfig(
        output_dir="./smoking_adapter", max_length=max_length,
        completion_only_loss=True, packing=False, learning_rate=1e-4,
        num_train_epochs=2, per_device_train_batch_size=2,
        gradient_accumulation_steps=8, gradient_checkpointing=True,
        bf16=use_bf16, fp16=not use_bf16, eval_strategy="epoch",
        save_strategy="epoch", report_to="none", push_to_hub=False,
    ),
)
trainer.train()
trainer.save_model("./smoking_adapter")
tokenizer.save_pretrained("./smoking_adapter")
```

Important: single-GPU CUDA-oriented starter, not a production application. Does not implement patient splitting, annotation review, classification-metric evaluation, or deployment security. Do not assume changing model ID to Qwen3.5 suffices: architecture includes a vision component; Unsloth provides a model-specific fine-tuning route.

## Cloud and sandbox options

For a first experiment, request one modest GPU and run a short memory/throughput pilot. Avoid A100/H100 time before measuring whether the small model needs it.

| Environment | Useful for | Important limitation |
|---|---|---|
| Approved institutional GPU or private Azure VM | local clinical dev/training | follow institution's data/security approval |
| Azure Machine Learning | managed jobs, controlled networking | private inbound / restricted outbound requires explicit config |
| Hugging Face Jobs | reproducible remote training/batch | data permission must be checked separately |
| HF dedicated Sandbox | interactive GPU experimentation | experimental API |
| HF shared SandboxPool | parallel lightweight CPU work | CPU-only, weaker isolation |
| Unsloth Studio/Desktop | UI-assisted experimentation/training | validate compatibility and security before clinical use |

A sandbox is an execution boundary, not proof of HIPAA compliance. HHS requires an appropriate BAA and applicable safeguards when a cloud provider handles ePHI. Cropping to 200 tokens does not de-identify a note; HIPAA de-identification uses specified methods, not context-length thresholds.

Deployment checklist:
- approved storage, compute, identity, network configuration
- no raw notes in experiment trackers, console examples, exception messages
- controlled access to adapters/checkpoints
- no automatic public Hub uploads
- review teacher-model endpoints and external telemetry
- separate review of dataset DUAs, even for de-identified research data

De-identified research data can still carry contractual restrictions (e.g., PhysioNet credentialed license restricts sharing).

## Evaluate the entire system

Evaluate retrieval + classification, not just classification on preselected successful snippets.

| Layer | Measurements |
|---|---|
| Retrieval | evidence retention, missed smoking mentions, missed contradictory evidence, retained-token count |
| Classification | macro-F1, per-class P/R/F1, confusion matrix |
| Safety-relevant errors | current→never, former→never, family history→patient smoking |
| Uncertainty | calibration and performance vs review/abstention rate |
| Operations | end-to-end latency, peak memory, throughput, processing cost |

Also: split by patient, not random windows; keep copied/near-duplicate notes from leaking across splits; report regex-positive and whole-pipeline performance separately; evaluate unknown subtypes; re-evaluate after quantization; if predicting longitudinal status, build a separate dated reconciliation layer.

Do NOT use "current always overrides former" across a patient's record — that policy can ignore later cessation; note-level extraction and patient-level timeline resolution must remain distinct.

## Concrete first milestones

1. Write the label guide (recent cessation, "non-smoker," unspecified history, product scope).
2. Annotate a small local pilot with evidence spans.
3. Compare retrieval strategies before assuming ±100 tokens suffice.
4. Train MiniLM/SetFit and Qwen3-0.6B LoRA on identical patient-disjoint splits.
5. Add Qwen3.5-0.8B as newer challenger with its supported recipe.
6. Introduce teacher-label distillation only if needed; keep human gold evaluation untouched.
7. Deploy the smallest model that meets the predefined quality target, with expanded-context review for difficult cases.

The project's main deliverable should be a validated smoking-status extraction system, not merely a fine-tuned model checkpoint.
