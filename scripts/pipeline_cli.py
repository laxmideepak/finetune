#!/usr/bin/env python3
"""Train, run and benchmark the end-to-end smoking-status pipeline (src/pipeline.py).

    # train the evidence-sentence TF-IDF classifier on Wright train + SDOH-NLI train
    python scripts/pipeline_cli.py train --wright-dir ../smokingbenchmark --sdoh-dir ../SDOH-NLI \
        --out runs/pipeline-tfidf.pkl

    # recommended model: fine-tuned Flan-T5-base (CPU, no GPU needed), or the TF-IDF -> Flan cascade
    python scripts/pipeline_cli.py train --classifier flan --flan-base ../models/flan-t5-base \
        --wright-dir ../smokingbenchmark --sdoh-dir ../SDOH-NLI --out runs/pipeline-flan.pkl
    python scripts/pipeline_cli.py train --classifier cascade --share 0.4 --flan-base ../models/flan-t5-base \
        --wright-dir ../smokingbenchmark --sdoh-dir ../SDOH-NLI --out runs/pipeline-cascade.pkl

    # label one note
    python scripts/pipeline_cli.py predict --model runs/pipeline-tfidf.pkl --note note.txt \
        --note-date 2025-06-01 --age 64

    # warm end-to-end CPU throughput and $/M on the Wright dev notes
    python scripts/pipeline_cli.py bench --model runs/pipeline-tfidf.pkl --wright-dir ../smokingbenchmark \
        --hourly-price 0.17
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.pipeline import SmokingStatusPipeline, as_dict, build_tfidf_classifier  # noqa: E402


def cmd_train(args) -> int:
    import pandas as pd

    from run_h7_ood import add_evidence, load_sdoh
    from run_wright_smoke import assign_splits, load

    frames = []
    wright = load(Path(args.wright_dir))
    wright["split"] = assign_splits(wright)
    frames.append(add_evidence(wright[wright["split"] == "train"])[["evid0", "label_native"]])
    if args.sdoh_dir:
        frames.append(add_evidence(load_sdoh(Path(args.sdoh_dir), "train"))[["evid0", "label_native"]])
    data = pd.concat(frames, ignore_index=True)
    data = data[data["evid0"].str.len() > 0]
    source = "wright+sdoh" if args.sdoh_dir else "wright"
    if args.classifier == "tfidf":
        classifier = build_tfidf_classifier().fit(data["evid0"], data["label_native"])
    else:
        from src.flan_status import CascadeClassifier, FlanStatusClassifier

        if not args.flan_base:
            raise SystemExit("--flan-base (a Flan-T5 folder, e.g. from scripts/fetch_flan_t5.py) is required")
        flan_dir = args.flan_out or str(Path(args.out).with_suffix("")) + "-flan"
        classifier = FlanStatusClassifier.train(args.flan_base, data["evid0"], data["label_native"], flan_dir)
        if args.classifier == "cascade":
            threshold = CascadeClassifier.fit_threshold(build_tfidf_classifier, data["evid0"], data["label_native"], share=args.share)
            classifier = CascadeClassifier(build_tfidf_classifier().fit(data["evid0"], data["label_native"]), classifier, threshold)
    pipeline = SmokingStatusPipeline(classifier, name=f"{args.classifier}-evidence[{source}]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pipeline.save(args.out)
    print(json.dumps({"saved": args.out, "train_rows": int(len(data)), "versions": pipeline.versions}, indent=1))
    return 0


def cmd_predict(args) -> int:
    pipeline = SmokingStatusPipeline.load(args.model)
    note = Path(args.note).read_text(encoding="utf-8")
    prediction = pipeline.predict([note], [date.fromisoformat(args.note_date)], [args.age])[0]
    print(json.dumps(as_dict(prediction), indent=1))
    return 0


def cmd_bench(args) -> int:
    from run_wright_smoke import assign_splits, load

    pipeline = SmokingStatusPipeline.load(args.model)
    wright = load(Path(args.wright_dir))
    wright["split"] = assign_splits(wright)
    dev = wright[wright["split"] == "dev"]
    notes, dates, ages = dev["text"].tolist(), [date.fromisoformat(d) for d in dev["note_date"]], dev["age"].astype(int).tolist()
    pipeline.predict(notes[:50], dates[:50], ages[:50])  # warm-up
    timings = []
    for _ in range(args.passes):
        started = time.perf_counter()
        predictions = pipeline.predict(notes, dates, ages)
        timings.append(time.perf_counter() - started)
    rate = len(notes) / min(timings)
    labels = [p.label for p in predictions]
    accuracy = sum(a == b for a, b in zip(labels, dev["label_one_year"])) / len(labels)
    report = {
        "notes": len(notes),
        "notes_per_second_single_process": round(rate, 1),
        "scope": "full pipeline on whole notes: retrieval + evidence + classifier + quit resolver, one CPU process",
        "one_year_v1_accuracy_on_wright_dev": round(accuracy, 4),
    }
    if args.hourly_price:
        report["hourly_price"] = args.hourly_price
        report["cpu_processes"] = args.processes
        report["usd_per_million_notes"] = round(1e6 * args.hourly_price / (3600 * rate * args.processes), 3)
    print(json.dumps(report, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--wright-dir", required=True)
    t.add_argument("--sdoh-dir", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--classifier", choices=("tfidf", "flan", "cascade"), default="tfidf")
    t.add_argument("--flan-base", default=None, help="Flan-T5 folder to fine-tune (flan / cascade)")
    t.add_argument("--flan-out", default=None, help="where to save the fine-tuned Flan-T5 (default: next to --out)")
    t.add_argument("--share", type=float, default=0.4, help="cascade: share of sentences routed to Flan (exploratory)")
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True)
    p.add_argument("--note", required=True)
    p.add_argument("--note-date", required=True)
    p.add_argument("--age", type=int, default=None)
    b = sub.add_parser("bench")
    b.add_argument("--model", required=True)
    b.add_argument("--wright-dir", required=True)
    b.add_argument("--passes", type=int, default=2)
    b.add_argument("--hourly-price", type=float, default=None)
    b.add_argument("--processes", type=int, default=4, help="parallel single-core processes on the priced machine")
    args = parser.parse_args(argv)
    return {"train": cmd_train, "predict": cmd_predict, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
