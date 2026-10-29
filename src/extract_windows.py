"""Sentence-aligned, section-aware crop extraction contract.

Candidate extraction only: this module never assigns a smoking label.

Design contract (research/pipeline-design.md Stage 3, v0.9.1):
  * one pinned retrieval tokenizer produces fixed character spans;
  * each trigger expands to sentence boundaries inside its section;
  * overlapping spans merge in source order;
  * crops are capped at ``budget`` tokens; evidence overflow creates wider
    note-context examples capped at ``wider_budget``;
  * nothing is dropped silently: truncated crops and multi-chunk wider context
    are flagged for explicit routing;
  * within-section duplicate sentences and boilerplate zones never seed crops;
  * every trigger, sentence, and crop keeps offsets and section metadata so the
    exact retained text can be re-tokenized by each model arm.

Run the offline demo:

    python src/extract_windows.py --demo

Run on a note with a real fast tokenizer:

    python src/extract_windows.py --text-file note.txt \
        --tokenizer thomas-sounack/BioClinical-ModernBERT-base --full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

RETRIEVAL_VERSION = "retrieval-v1"

# Recall-first lexicon from Stage 2. "toxic habits" is the verified UW synonym.
CANDIDATES = re.compile(
    r"\b(?:smok\w*|tobacc\w*|cigar\w*|cigaret\w*|cig\w*|nicotin\w*|"
    r"pack[\s-]*years?|ppd|pack[s]?\s*(?:a|per)?\s*day|cigs?\w*|"
    r"hookah|shisha|snuff|chew\w*|vap(?:e|es|ed|ing)\w*|e-?cig\w*|"
    r"toxic\s+habits?)\b",
    re.IGNORECASE,
)

_SECTION_NAMES = (
    "social history",
    "family history",
    "past medical history",
    "medical history",
    "surgical history",
    "hospital course",
    "history of present illness",
    "chief complaint",
    "medication list",
    "medications",
    "allergies",
    "substance use",
    "tobacco use",
    "smoking history",
    "smoking status",
    "social habits",
    "habits",
    "health[- ]related behaviors?",
    "review of systems",
    "assessment and plan",
    "assessment",
    "impression",
    "objective",
    "subjective",
    "plan",
)
_SECTION_ALT = "|".join(_SECTION_NAMES)
HEADER_LINE_RE = re.compile(
    r"^\s*(?P<header>" + _SECTION_ALT + r")\s*(?::\s*(?P<rest>.*))?\s*$",
    re.IGNORECASE,
)

# 10 BN-NR boilerplate topics verified in the dedup study (arXiv 2312.09469).
BOILERPLATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("attestation", re.compile(r"\b(?:i\s+attest|attests?\s+that|electronically\s+signed|signed\s+by)\b", re.I)),
    ("see-flowsheet", re.compile(r"\b(?:see|refer\s+to)\s+(?:the\s+)?flowsheet\b|\bflowsheet\b", re.I)),
    ("legal-disclaimer", re.compile(r"\b(?:disclaimer|confidential|unauthorized\s+use|not\s+intended\s+as\s+medical\s+advice)\b", re.I)),
    ("contact", re.compile(r"\b(?:if\s+you\s+have\s+any\s+questions|please\s+(?:call|contact)|contact\s+information)\b", re.I)),
    ("hypothetical-instruction", re.compile(r"\bif\s+(?:you|the\s+patient)\s+(?:develop|develops|experience|experiences|notice|notices)\b", re.I)),
    ("consent", re.compile(r"\b(?:informed\s+consent|consent(?:ed|s)?\s+(?:to|for)|authorized\s+to)\b", re.I)),
    ("billing-coding", re.compile(r"\b(?:billing|cpt\s*code|icd-?10|revenue\s+code)\b", re.I)),
    ("administrative", re.compile(r"\b(?:dictated\s+by|transcribed\s+by|copy\s+to|cc\s*:)\b", re.I)),
    ("medication-reconciliation", re.compile(r"\b(?:medication\s+reconciliation|medications?\s+reconciled)\b", re.I)),
    ("template-placeholder", re.compile(r"_{3,}|\[\s*\]|\bplaceholder\b", re.I)),
)
UNPOPULATED_TEMPLATE_RE = re.compile(
    r"^\s*(?:tobacco|smoking|alcohol|substance)\s+(?:use|status|history)\s*:\s*[_\-\s]*$",
    re.IGNORECASE,
)
FAMILY_RE = re.compile(
    r"\b(?:mother|father|parent|husband|wife|spouse|partner|son|daughter|child|children|"
    r"sibling|brother|sister|family|household|secondhand|second-hand|passive)\b",
    re.IGNORECASE,
)

_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc", "e.g", "i.e", "approx",
    "pt", "pts", "y.o", "y/o", "h/o", "s/p", "w/o", "b/l", "q.d", "b.i.d",
    "t.i.d", "q.i.d", "p.r.n", "s/l", "neg", "pos", "fig", "al", "no", "nos",
}


@dataclass
class Section:
    section_id: int
    header: str | None
    start: int
    content_start: int
    end: int


@dataclass
class Sentence:
    sent_id: int
    section_id: int
    start: int
    end: int
    text: str
    token_start: int = -1
    token_end: int = -1
    block_id: int | None = None
    excluded_reason: str | None = None
    duplicate_of: int | None = None


@dataclass
class Trigger:
    trigger_id: str
    char_start: int
    char_end: int
    text: str
    sent_id: int
    anchor_sent_id: int
    section_id: int
    family_margin: bool = False
    aliased_from_sent_id: int | None = None
    status: str = "used"
    ignored_reason: str | None = None


@dataclass
class Piece:
    sent_id: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    complete: bool = True
    truncated: bool = False
    omitted_tokens: int = 0


class WhitespaceOffsetTokenizer:
    """Offline word tokenizer used only by the demo and unit tests.

    It is deliberately not a clinical tokenizer. Real runs must pass a pinned
    Hugging Face fast tokenizer; the retrieval version and tokenizer revision
    are recorded in every result.
    """

    is_fast = True
    name_or_path = "builtin-whitespace-v1"
    _commit_hash = None

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {
            "offset_mapping": offsets,
            "input_ids": list(range(len(offsets))),
            "attention_mask": [1] * len(offsets),
        }


def split_sections(note: str) -> list[Section]:
    """Line-anchored section splitter. Headers are kept as metadata only."""
    headers: list[tuple[int, int, int, str]] = []
    offset = 0
    tobacco_template_names = {"tobacco use", "smoking status", "smoking history"}
    for line in note.splitlines(keepends=True):
        match = HEADER_LINE_RE.match(line)
        if match:
            header = match.group("header").strip().upper()
            rest = match.group("rest")
            # Tobacco/smoking template lines are evidence lines, not section
            # headers: "Tobacco use: ___" must stay visible as an unpopulated
            # template, and "Tobacco use: Current" must keep its trigger.
            if match.group("header").strip().lower() in tobacco_template_names:
                offset += len(line)
                continue
            if rest is None or rest == "":
                content_start = offset + len(line)
            else:
                content_start = offset + match.start("rest")
            headers.append((offset, content_start, offset + len(line), header))
        offset += len(line)

    if not headers:
        return [Section(0, None, 0, 0, len(note))]

    sections: list[Section] = []
    first_header_start = headers[0][0]
    if note[:first_header_start].strip():
        sections.append(Section(0, None, 0, 0, first_header_start))
    for index, (start, content_start, _line_end, header) in enumerate(headers):
        end = headers[index + 1][0] if index + 1 < len(headers) else len(note)
        sections.append(Section(len(sections), header, start, content_start, end))
    return sections


def _is_abbreviation(text: str, sentence_start: int, period_index: int) -> bool:
    if period_index > 0 and period_index + 1 < len(text):
        if text[period_index - 1].isdigit() and text[period_index + 1].isdigit():
            return True  # decimal number, not a sentence boundary
    prefix = text[sentence_start:period_index]
    match = re.search(r"([A-Za-z][A-Za-z./-]*)$", prefix)
    if not match:
        return False
    word = match.group(1).lower()
    return len(word) == 1 or word in _ABBREVIATIONS


def _sentence_spans(text: str, base: int) -> list[tuple[int, int]]:
    """Split text into sentence spans, preserving absolute character offsets."""
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if char == "\n":
            spans.append((start, i))
            start = i + 1
        elif char in ".!?":
            j = i + 1
            while j < len(text) and text[j] in "\"')]”’":
                j += 1
            k = j
            while k < len(text) and text[k] in " \t":
                k += 1
            boundary = k == len(text) or text[k].isupper() or text[k] in "\"'([“"
            if boundary and not _is_abbreviation(text, start, i):
                spans.append((start, j))
                start = j
                i = j
                continue
        i += 1
    spans.append((start, len(text)))

    cleaned: list[tuple[int, int]] = []
    for span_start, span_end in spans:
        while span_start < span_end and text[span_start].isspace():
            span_start += 1
        while span_end > span_start and text[span_end - 1].isspace():
            span_end -= 1
        if span_end > span_start and re.search(r"[A-Za-z0-9]", text[span_start:span_end]):
            cleaned.append((base + span_start, base + span_end))
    return cleaned


def tokenize_offsets(note: str, tokenizer: Any) -> list[tuple[int, int]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer with offset mappings is required for retrieval")
    encoded = tokenizer(
        note,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"] if e > s]
    if any(b[0] < a[0] for a, b in zip(offsets, offsets[1:])):
        raise ValueError("Tokenizer offsets are not monotonic")
    return offsets


def _map_range(starts: list[int], ends: list[int], char_start: int, char_end: int) -> tuple[int, int] | None:
    first = bisect_right(ends, char_start)
    last = bisect_left(starts, char_end)
    if first >= last:
        return None
    return first, last


def _tokenizer_id(tokenizer: Any) -> str:
    return str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__))


def _tokenizer_revision(tokenizer: Any) -> str | None:
    revision = getattr(tokenizer, "_commit_hash", None)
    if revision is None:
        init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
        revision = init_kwargs.get("_commit_hash")
    return str(revision) if revision else None


def _boilerplate_topic(text: str) -> str | None:
    if UNPOPULATED_TEMPLATE_RE.match(text):
        return "unpopulated-template"
    for topic, pattern in BOILERPLATE_PATTERNS:
        if pattern.search(text):
            return topic
    return None


def _family_margin(sentence_text: str, match_start: int, match_end: int) -> bool:
    lo = max(0, match_start - 80)
    hi = min(len(sentence_text), match_end + 80)
    return bool(FAMILY_RE.search(sentence_text[lo:hi]))


def _normalize_sentence(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _piece_from_sentence(sentence: Sentence) -> Piece:
    return Piece(
        sent_id=sentence.sent_id,
        char_start=sentence.start,
        char_end=sentence.end,
        token_start=sentence.token_start,
        token_end=sentence.token_end,
    )


def _build_sentences(note: str, sections: list[Section], offsets: list[tuple[int, int]]) -> list[Sentence]:
    starts = [x[0] for x in offsets]
    ends = [x[1] for x in offsets]
    sentences: list[Sentence] = []
    for section in sections:
        content = note[section.content_start:section.end]
        for char_start, char_end in _sentence_spans(content, section.content_start):
            token_range = _map_range(starts, ends, char_start, char_end)
            if token_range is None:
                continue
            sentences.append(
                Sentence(
                    sent_id=len(sentences),
                    section_id=section.section_id,
                    start=char_start,
                    end=char_end,
                    text=note[char_start:char_end],
                    token_start=token_range[0],
                    token_end=token_range[1],
                )
            )

    # Section-scoped duplicate detection: identical text in a *different*
    # section is retained because section context (social vs family history)
    # changes its meaning.
    seen: dict[tuple[int, str], int] = {}
    for sentence in sentences:
        topic = _boilerplate_topic(sentence.text)
        if topic:
            sentence.excluded_reason = topic
            continue
        normalized = _normalize_sentence(sentence.text)
        if len(normalized.split()) < 3:
            continue
        key = (sentence.section_id, normalized)
        if key in seen:
            sentence.duplicate_of = seen[key]
            sentence.excluded_reason = "within-section-duplicate"
        else:
            seen[key] = sentence.sent_id

    block = -1
    last_section: int | None = None
    previous_excluded = True
    for sentence in sentences:
        if sentence.excluded_reason:
            sentence.block_id = None
            previous_excluded = True
            continue
        if block < 0 or sentence.section_id != last_section or previous_excluded:
            block += 1
        sentence.block_id = block
        last_section = sentence.section_id
        previous_excluded = False
    return sentences


def _merge_ranges(
    ranges: list[tuple[int, int, int, str]], block_positions: dict[int, list[int]]
) -> list[dict[str, Any]]:
    """Merge overlapping sentence ranges within a block, preserving source order."""
    by_block: dict[int, list[tuple[int, int, str]]] = {}
    for lo_sent, hi_sent, block_id, trigger_id in ranges:
        positions = block_positions[block_id]
        lo = positions.index(lo_sent)
        hi = positions.index(hi_sent)
        by_block.setdefault(block_id, []).append((lo, hi, trigger_id))

    merged: list[dict[str, Any]] = []
    for block_id, items in by_block.items():
        items.sort(key=lambda item: (item[0], item[1]))
        for lo, hi, trigger_id in items:
            if merged and merged[-1]["block_id"] == block_id and lo <= merged[-1]["hi"]:
                merged[-1]["hi"] = max(merged[-1]["hi"], hi)
                merged[-1]["trigger_ids"].append(trigger_id)
            else:
                merged.append({"block_id": block_id, "lo": lo, "hi": hi, "trigger_ids": [trigger_id]})
    merged.sort(key=lambda item: (item["block_id"], item["lo"]))
    return merged


def _pack_crop_pieces(sentences: list[Sentence], budget: int) -> list[list[Piece]]:
    chunks: list[list[Piece]] = []
    current: list[Piece] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = sentence.token_end - sentence.token_start
        if sentence_tokens > budget:
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            chunks.append([
                Piece(
                    sent_id=sentence.sent_id,
                    char_start=sentence.start,
                    char_end=sentence.end,
                    token_start=sentence.token_start,
                    token_end=sentence.token_start + budget,
                    complete=False,
                    truncated=True,
                    omitted_tokens=sentence_tokens - budget,
                )
            ])
            continue
        if current and current_tokens + sentence_tokens > budget:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(_piece_from_sentence(sentence))
        current_tokens += sentence_tokens
    if current:
        chunks.append(current)
    return chunks


def _pack_wider_pieces(sentences: list[Sentence], wider_budget: int) -> list[list[Piece]]:
    chunks: list[list[Piece]] = []
    current: list[Piece] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = sentence.token_end - sentence.token_start
        if sentence_tokens > wider_budget:
            if current:
                chunks.append(current)
                current = []
                current_tokens = 0
            lo = sentence.token_start
            while lo < sentence.token_end:
                hi = min(sentence.token_end, lo + wider_budget)
                chunks.append([
                    Piece(
                        sent_id=sentence.sent_id,
                        char_start=sentence.start,
                        char_end=sentence.end,
                        token_start=lo,
                        token_end=hi,
                        complete=False,
                    )
                ])
                lo = hi
            continue
        if current and current_tokens + sentence_tokens > wider_budget:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(_piece_from_sentence(sentence))
        current_tokens += sentence_tokens
    if current:
        chunks.append(current)
    return chunks


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def extract(
    note: str,
    tokenizer: Any,
    *,
    radius: int = 100,
    budget: int = 512,
    wider_budget: int = 1024,
    dedupe_within_section: bool = True,
    drop_boilerplate: bool = True,
) -> dict[str, Any]:
    """Extract sentence-aligned crops and wider-context overflow examples.

    ``tokenizer`` must be the single pinned retrieval tokenizer. The returned
    character spans are the fixed evidence unit: every model arm re-tokenizes
    exactly these spans with its own tokenizer, which keeps comparisons fair.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if budget <= 0:
        raise ValueError("budget must be > 0")
    if wider_budget < budget:
        raise ValueError("wider_budget must be >= budget")
    if not isinstance(note, str):
        raise TypeError("note must be a string")
    if not note.strip():
        return _empty_result(note, tokenizer, status="empty-note")

    offsets = tokenize_offsets(note, tokenizer)
    if not offsets:
        return _empty_result(note, tokenizer, status="empty-note")

    starts = [x[0] for x in offsets]
    ends = [x[1] for x in offsets]
    sections = split_sections(note)
    sentences = _build_sentences(note, sections, offsets)
    if not dedupe_within_section:
        for sentence in sentences:
            if sentence.excluded_reason == "within-section-duplicate":
                sentence.excluded_reason = None
                sentence.duplicate_of = None
    if not drop_boilerplate:
        for sentence in sentences:
            if sentence.excluded_reason and sentence.excluded_reason != "within-section-duplicate":
                sentence.excluded_reason = None

    if not dedupe_within_section or not drop_boilerplate:
        block = -1
        last_section: int | None = None
        previous_excluded = True
        for sentence in sentences:
            if sentence.excluded_reason:
                sentence.block_id = None
                previous_excluded = True
                continue
            if block < 0 or sentence.section_id != last_section or previous_excluded:
                block += 1
            sentence.block_id = block
            last_section = sentence.section_id
            previous_excluded = False

    sentences_by_id = {sentence.sent_id: sentence for sentence in sentences}
    eligible = [sentence for sentence in sentences if sentence.block_id is not None]
    block_sentences: dict[int, list[Sentence]] = {}
    for sentence in eligible:
        block_sentences.setdefault(sentence.block_id, []).append(sentence)
    for block_id in block_sentences:
        block_sentences[block_id].sort(key=lambda sentence: sentence.sent_id)
        block_sentences[block_id].sort(key=lambda sentence: sentence.token_start)
    block_positions = {
        block_id: [sentence.sent_id for sentence in items]
        for block_id, items in block_sentences.items()
    }

    triggers: list[Trigger] = []
    ignored_triggers: list[dict[str, Any]] = []
    for index, match in enumerate(CANDIDATES.finditer(note)):
        trigger_id = f"t{index}"
        containing = [
            sentence for sentence in sentences
            if sentence.start <= match.start() and match.end() <= sentence.end
        ]
        if not containing:
            continue
        original = containing[0]
        anchor = original
        aliased_from: int | None = None
        if original.duplicate_of is not None:
            anchor = sentences_by_id[original.duplicate_of]
            aliased_from = original.sent_id
        if anchor.block_id is None:
            ignored_triggers.append({
                "trigger_id": trigger_id,
                "text": match.group(0),
                "char_start": match.start(),
                "char_end": match.end(),
                "sent_id": anchor.sent_id,
                "reason": anchor.excluded_reason or "excluded-sentence",
            })
            continue
        trigger = Trigger(
            trigger_id=trigger_id,
            char_start=match.start(),
            char_end=match.end(),
            text=match.group(0),
            sent_id=original.sent_id,
            anchor_sent_id=anchor.sent_id,
            section_id=anchor.section_id,
            family_margin=_family_margin(original.text, match.start() - original.start, match.end() - original.start),
            aliased_from_sent_id=aliased_from,
        )
        triggers.append(trigger)

    unpopulated_templates = [
        {
            "sent_id": sentence.sent_id,
            "section_id": sentence.section_id,
            "text": sentence.text,
        }
        for sentence in sentences
        if sentence.excluded_reason == "unpopulated-template"
    ]

    ranges: list[tuple[int, int, int, str]] = []
    for trigger in triggers:
        token_range = _map_range(starts, ends, trigger.char_start, trigger.char_end)
        if token_range is None:
            trigger.status = "ignored"
            trigger.ignored_reason = "token-mapping-failed"
            continue
        anchor = sentences_by_id[trigger.anchor_sent_id]
        block_items = block_sentences[anchor.block_id]
        raw_start = max(0, token_range[0] - radius)
        raw_end = min(len(offsets), token_range[1] + radius)
        overlapping = [
            sentence for sentence in block_items
            if sentence.token_end > raw_start and sentence.token_start < raw_end
        ]
        if not overlapping:
            overlapping = [anchor]
        lo_sent = min(sentence.sent_id for sentence in overlapping)
        hi_sent = max(sentence.sent_id for sentence in overlapping)
        ranges.append((lo_sent, hi_sent, anchor.block_id, trigger.trigger_id))

    clusters = _merge_ranges(ranges, block_positions)

    crops: list[dict[str, Any]] = []
    wider_context: list[dict[str, Any]] = []
    crop_counter = 0
    wider_counter = 0
    omitted_tokens_total = 0
    multi_chunk_wider_clusters = 0

    for cluster_index, cluster in enumerate(clusters):
        cluster_id = f"cluster{cluster_index}"
        block_id = cluster["block_id"]
        block_items = block_sentences[block_id]
        selected = block_items[cluster["lo"]:cluster["hi"] + 1]
        total_tokens = sum(sentence.token_end - sentence.token_start for sentence in selected)
        overflow = total_tokens > budget
        trigger_ids = cluster["trigger_ids"]

        piece_chunks = _pack_crop_pieces(selected, budget)
        part_count = len(piece_chunks)
        for part_index, pieces in enumerate(piece_chunks):
            char_start = min(piece.char_start for piece in pieces)
            char_end = max(piece.char_end for piece in pieces)
            text = note[char_start:char_end]
            n_tokens = sum(piece.token_end - piece.token_start for piece in pieces)
            truncated = any(piece.truncated for piece in pieces)
            omitted_tokens = sum(piece.omitted_tokens for piece in pieces)
            omitted_tokens_total += omitted_tokens
            crops.append({
                "crop_id": f"c{crop_counter}",
                "parent_cluster": cluster_id,
                "part_index": part_index,
                "part_count": part_count,
                "char_start": char_start,
                "char_end": char_end,
                "text": text,
                "marked_text": "«" + text + "»",
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "n_tokens": n_tokens,
                "sentence_ids": sorted({piece.sent_id for piece in pieces}),
                "section_ids": sorted({sentences_by_id[piece.sent_id].section_id for piece in pieces}),
                "section_headers": sorted({
                    sections[sentences_by_id[piece.sent_id].section_id].header
                    for piece in pieces
                    if sections[sentences_by_id[piece.sent_id].section_id].header
                }),
                "trigger_ids": trigger_ids,
                "block_id": block_id,
                "sentence_aligned": all(piece.complete for piece in pieces),
                "sentence_complete": all(piece.complete for piece in pieces),
                "truncated": truncated,
                "omitted_tokens": omitted_tokens,
                "needs_manual_route": truncated,
                "overflow_of_budget": overflow,
            })
            crop_counter += 1

        if overflow:
            wider_chunks = _pack_wider_pieces(selected, wider_budget)
            wider_part_count = len(wider_chunks)
            if wider_part_count > 1:
                multi_chunk_wider_clusters += 1
            for part_index, pieces in enumerate(wider_chunks):
                char_start = min(piece.char_start for piece in pieces)
                char_end = max(piece.char_end for piece in pieces)
                text = note[char_start:char_end]
                mid_sentence_split = any(not piece.complete for piece in pieces)
                wider_context.append({
                    "wider_id": f"w{wider_counter}",
                    "parent_cluster": cluster_id,
                    "part_index": part_index,
                    "part_count": wider_part_count,
                    "char_start": char_start,
                    "char_end": char_end,
                    "text": text,
                    "marked_text": "«" + text + "»",
                    "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "n_tokens": sum(piece.token_end - piece.token_start for piece in pieces),
                    "sentence_ids": sorted({piece.sent_id for piece in pieces}),
                    "section_ids": sorted({sentences_by_id[piece.sent_id].section_id for piece in pieces}),
                    "trigger_ids": trigger_ids,
                    "note_context": True,
                    "mid_sentence_split": mid_sentence_split,
                    "requires_multi_chunk_handling": wider_part_count > 1,
                    "truncated": False,
                    "omitted_tokens": 0,
                })
                wider_counter += 1

    crop_tokens = [crop["n_tokens"] for crop in crops]
    wider_tokens = [item["n_tokens"] for item in wider_context]
    if crops:
        status = "ok"
    elif triggers or unpopulated_templates:
        status = "all-triggers-excluded"
    else:
        status = "no-triggers"

    trigger_dicts = [asdict(trigger) for trigger in triggers]
    ignored_ids = {item["trigger_id"] for item in ignored_triggers}
    for trigger in trigger_dicts:
        if trigger["status"] == "ignored":
            ignored_ids.add(trigger["trigger_id"])
    used_ids = {trigger_id for crop in crops for trigger_id in crop["trigger_ids"]}

    stats = {
        "retrieval_version": RETRIEVAL_VERSION,
        "retrieval_tokenizer": _tokenizer_id(tokenizer),
        "retrieval_tokenizer_revision": _tokenizer_revision(tokenizer),
        "status": status,
        "note_chars": len(note),
        "note_tokens": len(offsets),
        "n_sections": len(sections),
        "n_sentences": len(sentences),
        "n_eligible_sentences": len(eligible),
        "n_boilerplate_sentences": sum(
            1 for sentence in sentences
            if sentence.excluded_reason and sentence.excluded_reason not in ("within-section-duplicate", "unpopulated-template")
        ),
        "n_within_section_duplicates": sum(
            1 for sentence in sentences if sentence.excluded_reason == "within-section-duplicate"
        ),
        "n_unpopulated_templates": len(unpopulated_templates),
        "n_triggers": len(triggers) + len(ignored_triggers),
        "n_triggers_used": len(used_ids),
        "n_triggers_ignored": len(ignored_ids),
        "n_clusters": len(clusters),
        "n_crops": len(crops),
        "n_wider_context": len(wider_context),
        "crop_tokens_max": max(crop_tokens) if crop_tokens else 0,
        "crop_tokens_p50": _percentile(crop_tokens, 0.50),
        "crop_tokens_p95": _percentile(crop_tokens, 0.95),
        "crops_over_256": sum(1 for value in crop_tokens if value > 256),
        "crops_over_256_rate": (sum(1 for value in crop_tokens if value > 256) / len(crop_tokens)) if crop_tokens else 0.0,
        "crops_over_512": sum(1 for value in crop_tokens if value > 512),
        "crops_over_1024": sum(1 for value in crop_tokens if value > 1024),
        "truncated_crops": sum(1 for crop in crops if crop["truncated"]),
        "tokens_omitted_from_crops": omitted_tokens_total,
        "wider_tokens_max": max(wider_tokens) if wider_tokens else 0,
        "wider_over_1024": sum(1 for value in wider_tokens if value > 1024),
        "multi_chunk_wider_clusters": multi_chunk_wider_clusters,
        "token_count_note": "counts use the pinned retrieval tokenizer; each model must recount the same character spans with its own tokenizer",
    }

    return {
        "retrieval_version": RETRIEVAL_VERSION,
        "retrieval_tokenizer": _tokenizer_id(tokenizer),
        "retrieval_tokenizer_revision": _tokenizer_revision(tokenizer),
        "note_chars": len(note),
        "status": status,
        "sections": [asdict(section) for section in sections],
        "sentences": [asdict(sentence) for sentence in sentences],
        "triggers": trigger_dicts,
        "ignored_triggers": ignored_triggers,
        "unpopulated_templates": unpopulated_templates,
        "crops": crops,
        "wider_context": wider_context,
        "stats": stats,
        "retrieval_notes": {
            "fixed_spans": "crops and wider_context carry character offsets; re-tokenize exactly these spans per model",
            "overflow": "evidence beyond budget is routed to wider_context, never silently discarded",
            "truncated_crops": "training prep must refuse truncated crops unless the wider-context route is used",
            "section_boundary": "crops never cross a section boundary",
        },
    }


def _empty_result(note: str, tokenizer: Any, *, status: str) -> dict[str, Any]:
    return {
        "retrieval_version": RETRIEVAL_VERSION,
        "retrieval_tokenizer": _tokenizer_id(tokenizer),
        "retrieval_tokenizer_revision": _tokenizer_revision(tokenizer),
        "note_chars": len(note),
        "status": status,
        "sections": [],
        "sentences": [],
        "triggers": [],
        "ignored_triggers": [],
        "unpopulated_templates": [],
        "crops": [],
        "wider_context": [],
        "stats": {
            "retrieval_version": RETRIEVAL_VERSION,
            "retrieval_tokenizer": _tokenizer_id(tokenizer),
            "retrieval_tokenizer_revision": _tokenizer_revision(tokenizer),
            "status": status,
            "note_chars": len(note),
            "note_tokens": 0,
            "n_sections": 0,
            "n_sentences": 0,
            "n_eligible_sentences": 0,
            "n_boilerplate_sentences": 0,
            "n_within_section_duplicates": 0,
            "n_unpopulated_templates": 0,
            "n_triggers": 0,
            "n_triggers_used": 0,
            "n_triggers_ignored": 0,
            "n_clusters": 0,
            "n_crops": 0,
            "n_wider_context": 0,
            "crop_tokens_max": 0,
            "crop_tokens_p50": None,
            "crop_tokens_p95": None,
            "crops_over_256": 0,
            "crops_over_256_rate": 0.0,
            "crops_over_512": 0,
            "crops_over_1024": 0,
            "truncated_crops": 0,
            "tokens_omitted_from_crops": 0,
            "wider_tokens_max": 0,
            "wider_over_1024": 0,
            "multi_chunk_wider_clusters": 0,
            "token_count_note": "counts use the pinned retrieval tokenizer; each model must recount the same character spans with its own tokenizer",
        },
        "retrieval_notes": {
            "fixed_spans": "crops and wider_context carry character offsets; re-tokenize exactly these spans per model",
            "overflow": "evidence beyond budget is routed to wider_context, never silently discarded",
            "truncated_crops": "training prep must refuse truncated crops unless the wider-context route is used",
            "section_boundary": "crops never cross a section boundary",
        },
    }


# Backward-compatible name: the full contract is the entry point now.
extract_windows = extract


DEMO_NOTE = """Social History:
Patient is a 58-year-old man. He quit smoking four months ago after 30 pack-years. Denies current smoking.
Family History:
Mother smoked for 30 years. Father never smoked.
Past Medical History:
Former smoker, cessation date unspecified.
"""


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "retrieval_version": result["retrieval_version"],
        "retrieval_tokenizer": result["retrieval_tokenizer"],
        "sections": [section["header"] for section in result["sections"]],
        "n_triggers": result["stats"]["n_triggers"],
        "n_triggers_used": result["stats"]["n_triggers_used"],
        "n_crops": len(result["crops"]),
        "n_wider_context": len(result["wider_context"]),
        "crops": [
            {
                "crop_id": crop["crop_id"],
                "n_tokens": crop["n_tokens"],
                "section_headers": crop["section_headers"],
                "truncated": crop["truncated"],
                "text": crop["text"],
            }
            for crop in result["crops"]
        ],
        "wider_context": [
            {
                "wider_id": item["wider_id"],
                "n_tokens": item["n_tokens"],
                "parts": item["part_count"],
                "mid_sentence_split": item["mid_sentence_split"],
            }
            for item in result["wider_context"]
        ],
        "stats": result["stats"],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="run the offline built-in example")
    parser.add_argument("--text-file", help="note text file (requires --tokenizer)")
    parser.add_argument("--tokenizer", help="pinned Hugging Face fast tokenizer id")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--radius", type=int, default=100)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--wider-budget", type=int, default=1024)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--keep-boilerplate", action="store_true")
    parser.add_argument("--full", action="store_true", help="print the full JSON result")
    parser.add_argument("--json-out", help="write the full JSON result to this path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.demo:
        note = DEMO_NOTE
        tokenizer: Any = WhitespaceOffsetTokenizer()
    elif args.text_file and args.tokenizer:
        from transformers import AutoTokenizer

        note = open(args.text_file, encoding="utf-8").read()
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.tokenizer_revision, use_fast=True
        )
    else:
        parser.error("use --demo, or provide both --text-file and --tokenizer")

    result = extract(
        note,
        tokenizer,
        radius=args.radius,
        budget=args.budget,
        wider_budget=args.wider_budget,
        dedupe_within_section=not args.no_dedupe,
        drop_boilerplate=not args.keep_boilerplate,
    )
    print(json.dumps(result if args.full else _summary(result), indent=2, ensure_ascii=False))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
