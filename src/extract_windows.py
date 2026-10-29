"""Candidate extraction only: this does NOT assign smoking labels.
Preserves character offsets. Add section/sentence boundaries in production.
"""
import re
from bisect import bisect_left, bisect_right

CANDIDATES = re.compile(
    r"\b(?:smok\w*|tobacc\w*|cig\w*|nicotin\w*|vaping|vape\w*|"
    r"pack[\s-]*years?|ppd|hookah|shisha|cigar\w*|snuff|chew\w*)\b",
    re.IGNORECASE,
)

def extract_windows(note, tokenizer, radius=100):
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offset mappings is required")
    encoded = tokenizer(note, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True, return_attention_mask=False)
    offsets = [tuple(x) for x in encoded["offset_mapping"] if x[1] > x[0]]
    if not offsets:
        return []
    starts, ends = zip(*offsets)
    windows = []
    for match in CANDIDATES.finditer(note):
        first = bisect_right(ends, match.start())
        stop = bisect_left(starts, match.end())
        if first >= stop:
            continue
        lo, hi = max(0, first - radius), min(len(offsets), stop + radius)
        windows.append((lo, hi))
    merged = []
    for lo, hi in sorted(windows):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return [{"start": offsets[lo][0], "end": offsets[hi-1][1],
             "text": note[offsets[lo][0]:offsets[hi-1][1]],
             "n_tokens": hi-lo} for lo, hi in merged]
