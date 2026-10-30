"""Evidence-sentence views of an extraction result (H6-EVIDENCE).

``extract`` returns merged, sentence-aligned crops. In short template notes a crop can
be a whole section, mixing the smoking sentence with distractors ("Retired, former
nurse", family history, medication lists). These helpers rebuild a smaller view: only
the sentences that contain a *used* retrieval trigger, optionally with neighbours from
the same section. Text is taken verbatim from the note, in source order.
"""

from __future__ import annotations

from typing import Any


def evidence_sentence_ids(result: dict[str, Any], *, context: int = 0) -> list[int]:
    sentences = {s["sent_id"]: s for s in result["sentences"]}
    anchors = sorted({t["anchor_sent_id"] for t in result["triggers"] if t.get("status") == "used"})
    keep: set[int] = set()
    for sid in anchors:
        section = sentences[sid]["section_id"]
        for offset in range(-context, context + 1):
            neighbour = sentences.get(sid + offset)
            if neighbour is None or neighbour["section_id"] != section or neighbour.get("excluded_reason"):
                continue
            keep.add(neighbour["sent_id"])
    return sorted(keep)


def evidence_text(note: str, result: dict[str, Any], *, context: int = 0, separator: str = "\n") -> str:
    sentences = {s["sent_id"]: s for s in result["sentences"]}
    return separator.join(
        note[sentences[sid]["start"] : sentences[sid]["end"]].strip()
        for sid in evidence_sentence_ids(result, context=context)
    )
