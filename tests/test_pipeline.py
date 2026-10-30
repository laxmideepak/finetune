from datetime import date

from src.pipeline import SmokingStatusPipeline, build_tfidf_classifier

TRAIN = [
    ("He smokes a pack a day.", "CURRENT"), ("Current smoker, 1 ppd.", "CURRENT"), ("Tob: current, 40 PY.", "CURRENT"),
    ("Former smoker, quit 2001.", "FORMER"), ("She quit smoking years ago.", "FORMER"), ("Ex-smoker, quit 2010.", "FORMER"),
    ("Never smoker.", "NEVER"), ("She has never smoked.", "NEVER"), ("Tob: never.", "NEVER"),
    ("Denies smoking.", "UNKNOWN"), ("Smoking status unknown.", "UNKNOWN"), ("Declined to discuss tobacco.", "UNKNOWN"),
] * 2


def _pipeline():
    texts, labels = zip(*TRAIN)
    return SmokingStatusPipeline(build_tfidf_classifier().fit(list(texts), list(labels)), name="test")


def test_no_trigger_is_retrieval_negative():
    p = _pipeline().predict(["Knee pain for two weeks."], [date(2025, 6, 1)], [60])[0]
    assert (p.label, p.unknown_reason) == ("UNKNOWN", "retrieval-negative")


def test_recent_quitter_becomes_current_and_old_quitter_former():
    pipe = _pipeline()
    notes = ["SH: Former smoker, quit smoking in March 2025.", "SH: Former smoker, quit smoking in 2001."]
    recent, old = pipe.predict(notes, [date(2025, 10, 1)] * 2, [66, 66])
    assert recent.native_status == "FORMER" and recent.label == "CURRENT"
    assert old.label == "FORMER"


def test_save_and_load_roundtrip(tmp_path):
    pipe = _pipeline()
    pipe.save(tmp_path / "p.pkl")
    again = SmokingStatusPipeline.load(tmp_path / "p.pkl")
    note = ["Never smoker."]
    assert again.predict(note, [date(2025, 1, 1)], [50])[0].label == pipe.predict(note, [date(2025, 1, 1)], [50])[0].label
