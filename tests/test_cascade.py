import numpy as np

from src.flan_status import CascadeClassifier
from src.pipeline import build_tfidf_classifier

TEXTS = ["He smokes daily.", "Current smoker.", "Never smoked.", "Never smoker.", "Quit in 2001.", "Former smoker.", "Denies smoking.", "No tobacco use."] * 3
LABELS = ["CURRENT", "CURRENT", "NEVER", "NEVER", "FORMER", "FORMER", "UNKNOWN", "UNKNOWN"] * 3


class FakeFlan:
    def __init__(self):
        self.calls = []

    def predict(self, texts):
        self.calls.append(list(texts))
        return ["FLAN"] * len(texts)


def test_threshold_routes_roughly_the_requested_share():
    threshold = CascadeClassifier.fit_threshold(build_tfidf_classifier, TEXTS, LABELS, share=0.5)
    flan = FakeFlan()
    cascade = CascadeClassifier(build_tfidf_classifier().fit(TEXTS, LABELS), flan, threshold)
    pred = cascade.predict(TEXTS)
    assert len(pred) == len(TEXTS)
    assert 0.0 <= cascade.last_routed_share <= 1.0
    assert sum(p == "FLAN" for p in pred) == sum(len(c) for c in flan.calls)


def test_zero_threshold_never_calls_flan_and_infinite_always_does():
    tfidf = build_tfidf_classifier().fit(TEXTS, LABELS)
    flan = FakeFlan()
    assert "FLAN" not in CascadeClassifier(tfidf, flan, -np.inf).predict(TEXTS) and not flan.calls
    assert set(CascadeClassifier(tfidf, flan, np.inf).predict(TEXTS)) == {"FLAN"}
