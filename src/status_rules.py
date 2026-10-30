"""Keyword rules for native smoking status (a no-training baseline arm).

Applied to evidence sentences. Precedence: explicit never > former/quit > current >
denial-only. Anything else with a trigger is UNKNOWN. Built from the label policy's
canonical cases, not tuned on any evaluation split.
"""

from __future__ import annotations

import re

NEVER = re.compile(
    r"\b(?:never\s+(?:a\s+)?smok\w*|never[\s-]+smoker|non-?smoker|lifelong\s+non|never\s+(?:used\s+)?(?:tobacco|tob|cig\w*)|"
    r"no\s+(?:prior\s+|past\s+)?(?:history|hx)\s+of\s+(?:tobacco|smoking|cig\w*)|no\s+tob\w*\s+hx|"
    r"has\s+never\s+smoked|denies\s+(?:ever|any\s+history\s+of|history\s+of)\b[^.]{0,30}?(?:smok\w*|tob\w*|cig\w*)|tob\w*\s*:\s*never|smoking\s+status\s*:\s*never)",
    re.I,
)
FORMER = re.compile(
    r"\b(?:former|ex-?\s?smoker|ex-?tob\w*|quit|stopped|gave\s+up|used\s+to\s+smoke|previously\s+smoked|"
    r"past\s+smoker|smoke[\s-]?free|tob\w*\s*:\s*former|smoking\s+status\s*:\s*former|remote\s+(?:history|hx))",
    re.I,
)
CURRENT = re.compile(
    r"\b(?:current(?:ly)?\s+(?:smok\w*|tob\w*|uses?)|active\s+smoker|smokes|is\s+smoking|still\s+smok\w*|continues\s+to\s+smoke|"
    r"(?:packs?|cig\w*)\s+(?:of\s+cig\w*\s+)?(?:a|per)\s+day|(?:packs?|cig\w*)\s+daily|"
    r"tob\w*\s*:\s*current|smoking\s+status\s*:\s*(?:current|every\s+day|some\s+days)|\+\s*tob\w*|"
    r"\d+(?:\.\d+)?\s*(?:ppd|packs?\s+(?:a|per)\s+day|cig\w*\s+(?:a|per)\s+day))",
    re.I,
)
DENIAL = re.compile(r"\b(?:denies|does\s+not|doesn't|no)\s+(?:\w+\s+){0,2}(?:smok\w*|tob\w*|cig\w*)", re.I)


def rule_status(evidence: str) -> str:
    if not evidence or not evidence.strip():
        return "UNKNOWN"
    if NEVER.search(evidence):
        return "NEVER"
    if FORMER.search(evidence):
        return "FORMER"
    if CURRENT.search(evidence):
        return "CURRENT"
    if DENIAL.search(evidence):
        return "UNKNOWN"
    return "UNKNOWN"
