#!/usr/bin/env python3
"""Train, run and benchmark the end-to-end smoking-status pipeline (src/pipeline.py).

    # train the evidence-sentence TF-IDF classifier on Wright train + SDOH-NLI train
    python scripts/pipeline_cli.py train --wright-dir ../smokingbenchmark --sdoh-dir ../SDOH-NLI \
        --out runs/pipeline-tfidf.pkl

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
    classifier = build_tfidf_classifier().fit(data["evid0"], data["label_native"])
    pipeline = SmokingStatusPipeline(classifier, name=f"tfidf-svm-evidence[{'wright+sdoh' if args.sdoh_dir else 'wright'}]")
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
