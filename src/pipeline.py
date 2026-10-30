"""End-to-end smoking-status pipeline: note -> label + reason + evidence.

Stages (design contract Stages 2-7):
  1. retrieval     ``extract_windows.extract`` (retrieval-v2 lexicon, sentence-aligned)
  2. evidence      ``evidence.evidence_text`` (trigger sentences only; H6)
  3. status        a native-status classifier (CURRENT / FORMER / NEVER / UNKNOWN)
  4. one-year rule ``quit_resolver`` reads the quit time and applies the rule using the
                   note date (H5); runs only when status is FORMER
Notes without any trigger are UNKNOWN(retrieval-negative) without calling a model.

The classifier is pluggable: anything with ``predict(list[str]) -> list[str]``. The
default is the TF-IDF word+char LinearSVM (arm A) trained on evidence sentences.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

try:
    from src.evidence import evidence_text
    from src.extract_windows import RETRIEVAL_VERSION, WhitespaceOffsetTokenizer, extract
    from src.label_policy import POLICY_VERSION
    from src.quit_resolver import RESOLVER_VERSION, apply_one_year_rule, resolve_quit
except ImportError:  # running from inside src/
    from evidence import evidence_text
    from extract_windows import RETRIEVAL_VERSION, WhitespaceOffsetTokenizer, extract
    from label_policy import POLICY_VERSION
    from quit_resolver import RESOLVER_VERSION, apply_one_year_rule, resolve_quit


class StatusClassifier(Protocol):
    def predict(self, texts: list[str]) -> list[str]: ...


@dataclass
class Prediction:
    label: str
    unknown_reason: str | None
    native_status: str
    evidence: str
    quit_interval: tuple[str, str] | None
    quit_text: str | None
    versions: dict


def build_tfidf_classifier():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, make_pipeline
    from sklearn.svm import LinearSVC

    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ]
    )
    return make_pipeline(features, LinearSVC(C=1.0, class_weight="balanced"))


class SmokingStatusPipeline:
    def __init__(self, classifier: StatusClassifier, *, name: str = "tfidf-svm-evidence"):
        self.classifier = classifier
        self.name = name
        self._tokenizer = WhitespaceOffsetTokenizer()

    @property
    def versions(self) -> dict:
        return {"policy": POLICY_VERSION, "retrieval": RETRIEVAL_VERSION, "resolver": RESOLVER_VERSION, "classifier": self.name}

    def evidence(self, note: str) -> tuple[str, str]:
        result = extract(note, self._tokenizer)
        crops = "\n\n".join(c["text"] for c in result["crops"])
        return evidence_text(note, result, context=0), crops

    def predict(self, notes: list[str], note_dates: list[date], ages: list[int | None]) -> list[Prediction]:
        views = [self.evidence(n) for n in notes]
        todo = [i for i, (ev, _) in enumerate(views) if ev.strip()]
        native = ["UNKNOWN"] * len(notes)
        if todo:
            for i, label in zip(todo, self.classifier.predict([views[i][0] for i in todo])):
                native[i] = str(label)
        out = []
        for i, note in enumerate(notes):
            ev, crops = views[i]
            if not ev.strip():
                out.append(Prediction("UNKNOWN", "retrieval-negative", "UNKNOWN", "", None, None, self.versions))
                continue
            resolution = resolve_quit(crops, note_date=note_dates[i], age=ages[i]) if native[i] == "FORMER" else None
            if resolution is not None:
                label, reason = apply_one_year_rule(native[i], resolution, note_date=note_dates[i])
                interval = (str(resolution.earliest), str(resolution.latest)) if resolution.earliest else None
                quit_text = resolution.matched_text
            else:
                label, reason = native[i], ("unclear-or-denial" if native[i] == "UNKNOWN" else None)
                interval, quit_text = None, None
            out.append(Prediction(label, reason, native[i], ev, interval, quit_text, self.versions))
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as handle:
            pickle.dump({"name": self.name, "classifier": self.classifier, "versions": self.versions}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SmokingStatusPipeline":
        with open(path, "rb") as handle:
            blob = pickle.load(handle)
        return cls(blob["classifier"], name=blob["name"])


def as_dict(prediction: Prediction) -> dict:
    return asdict(prediction)
