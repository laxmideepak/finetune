"""Rule-based cigarette quit-time resolver for the one-year rule (H5-DATE).

Reads note text (normally the retrieved crops), finds statements of WHEN the patient
stopped smoking cigarettes, and turns each into a cessation-date interval
[earliest, latest]. ``apply_one_year_rule`` then combines a status (from a classifier
or gold) with that interval through ``label_policy.status_from_cessation``.

Inputs besides text are EHR metadata, not note text: the note's reference date and the
patient's age at that date.

Precedence when several quit statements exist: the most recent cessation interval wins
(handles "quit 2007, relapsed 2013, quit again 2019"). Statements dated after the note
(planned quit dates) and statements about alcohol, cannabis or vaping are ignored.
Built from TRAIN-split phrasings only (experiments/h5-date-resolver/protocol.md).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta

try:
    from src.label_policy import status_from_cessation
except ImportError:  # running as a script from src/
    from label_policy import status_from_cessation

RESOLVER_VERSION = "quit-resolver-v1"

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "a couple of": 2, "a couple": 2, "a few": 3, "several": 3, "half a": 0.5,
}
_NUM = r"(?P<num>\d+(?:\.\d+)?|" + "|".join(sorted((re.escape(k) for k in _NUMBER_WORDS), key=len, reverse=True)) + r")"
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTHS["sept"] = 9
_MONTH = r"(?P<month>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_YEAR = r"(?P<year>(?:19|20)\d{2}|'\d{2})"

# Cessation cues. "ceased" needs a word boundary so "deceased" never matches.
QUIT_CUE = re.compile(
    r"\b(?:quit(?:s|ting)?(?:\s+again)?|quite\s+smoking|qiut|stopped(?:\s+smoking)?|stop(?:ped)?\s+date|gave\s+up|"
    r"ceased|cessation|kicked\s+the\s+habit|smoke[\s-]?free|tob(?:acco)?[\s-]?free|"
    r"cigarette[\s-]?free|nicotine[\s-]?free|no\s+cig(?:arette)?s|"
    r"(?:has\s+not|hasn't|not)\s+(?:smoked|had\s+(?:a\s+)?(?:cigarette|cig))|"
    r"last\s+(?:cigarette|cig)|former\s+smoker|ex-?smoker)\b",
    re.IGNORECASE,
)
# Clause-level exclusions: other substances, plans, counselling boilerplate, relapse.
OTHER_SUBSTANCE = re.compile(r"\b(?:drink\w*|alcohol|etoh|beer|wine|liquor|sober\w*|marijuana|cannabis|thc|vap\w*|e-?cig\w*|opioid\w*|heroin|cocaine)\b", re.I)
PLAN = re.compile(r"\b(?:plan\w*|will|wants?\s+to|set|target|advis\w*|motivated|ready|contemplat\w*|ambivalent|attempt\w*|trying|try|thinking|consider\w*|quitline|quit-now|counsel\w*|encourag\w*|discussed|interested)\b", re.I)
RELAPSE = re.compile(r"\b(?:relaps\w*|restart\w*|resum\w*|started\s+again|back\s+to\s+smoking|began\s+again)\b", re.I)


@dataclass
class Resolution:
    earliest: date | None = None
    latest: date | None = None
    matched_text: str | None = None
    rule: str | None = None
    n_candidates: int = 0

    def as_dict(self) -> dict:
        return {"earliest": self.earliest, "latest": self.latest, "matched_text": self.matched_text, "rule": self.rule, "n_candidates": self.n_candidates}


def _num(token: str) -> float:
    token = token.lower().strip()
    return float(token) if re.match(r"^\d", token) else float(_NUMBER_WORDS[token])


def _year(token: str, note_date: date) -> int:
    if token.startswith("'"):
        yy = int(token[1:])
        return 2000 + yy if 2000 + yy <= note_date.year else 1900 + yy
    return int(token)


def _shift_years(d: date, years: float) -> date:
    return d - timedelta(days=round(years * 365.25))


def _shift_months(d: date, months: float) -> date:
    return d - timedelta(days=round(months * 30.44))


def _clauses(text: str):
    """Yield (start, clause) split on sentence/semicolon/bullet boundaries, keeping commas."""
    start = 0
    for m in re.finditer(r"[;\n•]|\.\s+(?=[A-Z])|\.$", text):
        clause = text[start : m.start()]
        if clause.strip():
            yield start, clause
        start = m.end()
    if text[start:].strip():
        yield start, text[start:]


def _decade(word: str, decade: int) -> tuple[int, int]:
    word = (word or "").lower()
    if word == "early":
        return decade, decade + 3
    if word in ("mid", "middle"):
        return decade + 3, decade + 6
    if word == "late":
        return decade + 6, decade + 9
    return decade, decade + 9


_HOLIDAYS = {"new year's": (1, 1), "new years": (1, 1), "new year": (1, 1), "july 4th": (7, 4), "fourth of july": (7, 4), "july 4": (7, 4), "christmas": (12, 25), "valentine's day": (2, 14)}

# Ordered from most to least precise. Each pattern is searched in the text that FOLLOWS a
# quit cue inside one clause (window), except the "ago"/"in N months" families which may
# also precede it ("15 months ago she quit").
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("full-date-mdy", re.compile(r"(?P<m>\d{1,2})/(?P<d>\d{1,2})/(?P<y>(?:19|20)\d{2})")),
    ("full-date-iso", re.compile(r"(?P<y>(?:19|20)\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})")),
    ("full-date-text", re.compile(_MONTH + r"\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>(?:19|20)\d{2})", re.I)),
    ("holiday", re.compile(r"(?P<h>" + "|".join(re.escape(h) for h in _HOLIDAYS) + r")(?:\s+day)?,?\s+(?:of\s+)?(?P<y>(?:19|20)\d{2})", re.I)),
    ("years-since-quitting", re.compile(r"years?\s+since\s+quitting\s*:?\s*(?P<num>\d+(?:\.\d+)?)", re.I)),
    ("month-year-slash", re.compile(r"\b(?P<m>\d{1,2})/(?P<y>(?:19|20)\d{2})\b")),
    ("month-year", re.compile(_MONTH + r"\s+(?:of\s+)?" + _YEAR, re.I)),
    ("decade", re.compile(r"(?:the\s+)?(?P<w>early|mid|middle|late)?[\s-]*(?P<dec>(?:19|20)\d0)'?s\b", re.I)),
    ("months-ago", re.compile(r"(?:about|approximately|~|roughly|around)?\s*" + _NUM + r"\s*(?:months?|mos?|mths?)\b(?:\s+ago|\s+prior)?", re.I)),
    ("weeks-ago", re.compile(r"" + _NUM + r"\s*(?:weeks?|wks?)\b(?:\s+ago)?", re.I)),
    ("days-ago", re.compile(r"" + _NUM + r"\s*days?\b(?:\s+ago)?", re.I)),
    ("years-ago", re.compile(r"(?:about|approximately|~|roughly|around|x)?\s*" + _NUM + r"\s*(?:\+\s*)?(?:years?|yrs?|y)\b(?:\s+ago|\s+prior)?", re.I)),
    ("last-year", re.compile(r"\blast\s+year\b", re.I)),
    ("birthday", re.compile(r"(?P<bday>\d{1,2})(?:st|nd|rd|th)\s+birthday", re.I)),
    ("event-offset", re.compile(r"(?P<num>\d+|" + "|".join(k for k in _NUMBER_WORDS if " " not in k) + r")\s+(?:years?|yrs?)\s+(?P<dir>after|before)\b[^;]{0,80}?\b(?P<year>(?:19|20)\d{2})\b", re.I)),
    ("child-age", re.compile(r"(?:born|birth)\b[^;]{0,60}?\b(?:now|is)\s+(?:age\s+)?" + _NUM + r"(?:\s*(?:years?|yrs?|y)(?:\s+old)?)?", re.I)),
    ("event-ref", re.compile(r"\b(?P<event>heart\s+attack|myocardial\s+infarction|mi|n?stemi|bypass|cabg|stent|pci|des|stroke|cva|copd|lung\s+cancer|cancer|surgery|diagnos\w+|wife|husband|spouse)\b", re.I)),
    ("at-age", re.compile(r"(?:at|by)\s+(?:the\s+)?age\s+(?:of\s+)?(?P<age>\d{1,2})|age\s*:?\s*(?P<age2>\d{1,2})\b|when\s+(?:he|she|they)\s+(?:was|were)\s+(?P<age3>\d{1,2})", re.I)),
    ("year", re.compile(r"(?:in|since|of|during|around|circa|ca\.?|~)?\s*" + _YEAR + r"\b", re.I)),
]

_WINDOW = 80


def _interval(kind: str, m: re.Match[str], note_date: date, age: int | None) -> tuple[date, date] | None:
    g = m.groupdict()
    try:
        if kind in ("full-date-mdy", "full-date-iso"):
            d = date(int(g["y"]), int(g["m"]), int(g["d"]))
            return d, d
        if kind == "full-date-text":
            d = date(int(g["y"]), _MONTHS[g["month"].lower()], int(g["d"]))
            return d, d
        if kind == "holiday":
            month, day = _HOLIDAYS[g["h"].lower()]
            d = date(int(g["y"]), month, day)
            return d, d
        if kind == "years-since-quitting":
            n = float(g["num"])
            return _shift_years(note_date, n + 0.1), _shift_years(note_date, max(n - 0.1, 0))
        if kind in ("month-year-slash", "month-year"):
            month = int(g["m"]) if kind == "month-year-slash" else _MONTHS[g["month"].lower()]
            if not 1 <= month <= 12:
                return None
            year = _year(g["y"] if kind == "month-year-slash" else g["year"], note_date)
            return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
        if kind == "decade":
            lo, hi = _decade(g["w"], int(g["dec"]))
            return date(lo, 1, 1), date(hi, 12, 31)
        if kind == "months-ago":
            n = _num(g["num"])
            if n > 240:
                return None
            return _shift_months(note_date, n + 0.5), _shift_months(note_date, max(n - 0.5, 0))
        if kind == "weeks-ago":
            n = _num(g["num"])
            return note_date - timedelta(weeks=n + 0.5), note_date - timedelta(weeks=max(n - 0.5, 0))
        if kind == "days-ago":
            n = _num(g["num"])
            return note_date - timedelta(days=n + 1), note_date - timedelta(days=max(n - 1, 0))
        if kind == "years-ago":
            n = _num(g["num"])
            if n > 80:
                return None
            # "quit 2 years ago": completed years, so cessation lies in (N+1, N] years before.
            return _shift_years(note_date, n + 1), _shift_years(note_date, n)
        if kind == "last-year":
            y = note_date.year - 1
            return date(y, 1, 1), date(y, 12, 31)
        if kind == "at-age":
            quit_age = int(g["age"] or g["age2"] or g["age3"])
            if age is None or quit_age > age or quit_age < 8:
                return None
            # "quit at age Q" in a patient aged A: anywhere from (A-Q-1) to (A-Q+1) years ago.
            gap = age - quit_age
            return _shift_years(note_date, gap + 1), _shift_years(note_date, max(gap - 1, 0))
        if kind == "birthday":
            bday = int(g["bday"])
            if age is None or bday > age:
                return None
            gap = age - bday
            return _shift_years(note_date, gap + 1), _shift_years(note_date, gap)
        if kind == "event-offset":
            n = _num(g["num"])
            y = int(g["year"]) + (int(n) if g["dir"].lower() == "after" else -int(n))
            return date(y, 1, 1), date(y, 12, 31)
        if kind == "child-age":
            n = _num(g["num"])
            if n > 60:
                return None
            return _shift_years(note_date, n + 1), _shift_years(note_date, n)
        if kind == "year":
            y = _year(g["year"], note_date)
            return date(y, 1, 1), date(y, 12, 31)
    except (ValueError, KeyError):
        return None
    return None


# Time expressions after these words describe something else (duration of smoking,
# relapse year, smoking start), never the cessation time.
_NOT_CESSATION_BEFORE = re.compile(
    r"(?:\bfor|\bx|×|\bsmoked|\bstarted|\bbegan|\bstarting|\bfrom|\buntil\s+\d|\bfirst|\bpreviously|\bprior\s+quit|\bhistory\s+of|\bpack[\s-]*y(?:ea)?rs?|\bpk[\s-]*y(?:ea)?rs?|\bppd|\bper\s+day|\ba\s+day|\bdaily|\bcigs?/day|\bcig(?:arette)?s?\s+(?:a|per)\s+day|\bage\s+\d{1,2}\s*-)\s*[^,;()]{0,12}$",
    re.I,
)


_EVENT_SYNONYMS = {
    "heart attack": r"heart\s+attack|myocardial\s+infarction|\bn?stemi\b|\bmi\b",
    "myocardial infarction": r"heart\s+attack|myocardial\s+infarction|\bn?stemi\b|\bmi\b",
    "mi": r"heart\s+attack|myocardial\s+infarction|\bn?stemi\b|\bmi\b",
    "stemi": r"heart\s+attack|myocardial\s+infarction|\bn?stemi\b|\bmi\b",
    "nstemi": r"heart\s+attack|myocardial\s+infarction|\bn?stemi\b|\bmi\b",
    "bypass": r"bypass|\bcabg\b",
    "cabg": r"bypass|\bcabg\b",
    "stent": r"stent|\bpci\b|\bdes\b",
    "pci": r"stent|\bpci\b|\bdes\b",
    "des": r"stent|\bpci\b|\bdes\b",
    "stroke": r"stroke|\bcva\b",
    "cva": r"stroke|\bcva\b",
    "copd": r"\bcopd\b",
    "lung cancer": r"lung\s+cancer|adenocarcinoma|nsclc",
    "cancer": r"cancer|carcinoma",
}


def _event_year(text: str, event: str) -> tuple[date, date] | None:
    """Year of a medical event named in a quit statement, looked up anywhere in the note
    ("quit after his CABG" ... "CAD s/p CABG x3 (1994)")."""
    synonyms = _EVENT_SYNONYMS.get(re.sub(r"\s+", " ", event.lower()))
    if not synonyms:
        return None
    years = set()
    for m in re.finditer(r"(?:" + synonyms + r")[^\n;.]{0,40}?\(?\b(?:dx\s+|diagnosed\s+|in\s+)?((?:19|20)\d{2})\b", text, re.I):
        years.add(int(m.group(1)))
    if len(years) != 1:
        return None
    y = years.pop()
    return date(y, 1, 1), date(y, 12, 31)


def _child_age(text: str, start: int, note_date: date) -> tuple[date, date, str] | None:
    """'quit ... when his granddaughter was born; she is now 3 years old' (crosses ';')."""
    span = text[start : start + 180]
    m = re.search(r"\b(?:born|birth)\b[^.]{0,80}?\b(?:now|is)\s+(?:age\s+)?" + _NUM + r"\b", span, re.I)
    if not m:
        return None
    n = _num(m.group("num"))
    if n > 60:
        return None
    return _shift_years(note_date, n + 1), _shift_years(note_date, n), span[: m.end()]


_OFFSET_BACK = re.compile(
    r"\b(?P<year>(?:19|20)\d{2})\b[^\n]{0,220}?\b(?P<num>\d+|" + "|".join(k for k in _NUMBER_WORDS if " " not in k) + r")\s+(?:years?|yrs?)\s+(?P<dir>after|before)\b",
    re.I,
)


def _offset_back(text: str, cue: re.Match[str]) -> tuple[date, date, str] | None:
    """'husband passed away in 2017 ... quitting approximately 5 years after his death'."""
    lo = max(0, cue.start() - 250)
    span = text[lo : cue.end() + 120]
    best = None
    for m in _OFFSET_BACK.finditer(span):
        if OTHER_SUBSTANCE.search(span[m.start() : m.end()]):
            continue
        n = int(_num(m.group("num")))
        y = int(m.group("year")) + (n if m.group("dir").lower() == "after" else -n)
        best = (date(y, 1, 1), date(y, 12, 31), m.group(0))
    return best


def _candidates(text: str, note_date: date, age: int | None):
    for cue in QUIT_CUE.finditer(text):
        back = _offset_back(text, cue)
        if back and back[0] <= note_date:
            yield "event-offset-back", back[0], min(back[1], note_date), back[2]
        child = _child_age(text, cue.end(), note_date)
        if child:
            yield "child-age", child[0], child[1], cue.group(0) + child[2]
    for clause_start, clause in _clauses(text):
        for cue in QUIT_CUE.finditer(clause):
            before = clause[max(0, cue.start() - 40) : cue.start()]
            after = clause[cue.end() : cue.end() + _WINDOW]
            if OTHER_SUBSTANCE.search(after[:25]) or OTHER_SUBSTANCE.search(before[-25:]):
                continue
            if PLAN.search(before[-30:]) or re.match(r"\s*(?:date\s+)?(?:set|planned|target)", after, re.I):
                continue
            # Cut the window at a relapse word: anything after it is relapse timing.
            relapse = RELAPSE.search(after)
            window = after[: relapse.start()] if relapse else after
            # Also allow a preceding relative expression ("15 months ago she quit").
            pre = clause[max(0, cue.start() - 35) : cue.start()]
            for kind, pattern in _PATTERNS:
                for source, offset_text in (("after", window), ("before", pre)):
                    if source == "before" and kind not in ("months-ago", "years-ago", "weeks-ago"):
                        continue
                    for m in pattern.finditer(offset_text):
                        if not m.group(0).strip():
                            continue
                        if kind in ("months-ago", "years-ago", "weeks-ago", "days-ago") and "ago" not in m.group(0).lower() and source == "before":
                            continue
                        lead = offset_text[: m.start()]
                        if re.search(r"\b(?:today|date\s+of\s+service|dos|seen|presenting|presents|visit|encounter|signed|dictated)\b[^;]{0,15}$", lead, re.I):
                            continue
                        cue_l = cue.group(0).lower()
                        quit_x = kind in ("years-ago", "months-ago") and source == "after" and "quit" in cue_l and (re.fullmatch(r"\s*(?:x|×)\s*", lead) or (not lead.strip() and re.match(r"\s*x\s*\d", m.group(0), re.I)))
                        if kind in ("years-ago", "months-ago") and re.search(r"\bfor\s*$", lead, re.I) and any(k in cue_l for k in ("free", "not", "no cig", "hasn't")):
                            pass
                        elif quit_x:
                            pass
                        elif kind in ("years-ago", "months-ago", "weeks-ago", "days-ago", "year", "at-age") and _NOT_CESSATION_BEFORE.search(lead):
                            continue
                        if kind in ("years-ago", "months-ago", "weeks-ago", "days-ago"):
                            # Bare durations ("x 41 yrs") are smoking duration unless the cue
                            # itself is a smoke-free/not-smoked cue or the phrase says "ago".
                            cue_text = cue.group(0).lower()
                            durational_cue = any(k in cue_text for k in ("free", "not", "no cig", "hasn't"))
                            if "ago" not in m.group(0).lower() and "prior" not in m.group(0).lower() and not durational_cue and not quit_x:
                                continue
                        if kind == "event-ref":
                            interval = _event_year(text, m.group("event"))
                        else:
                            interval = _interval(kind, m, note_date, age)
                        if interval is None:
                            continue
                        earliest, latest = interval
                        if earliest > note_date:  # a planned/future quit date
                            continue
                        latest = min(latest, note_date)
                        span = (clause_start + cue.start(), clause_start + cue.end() + m.end())
                        yield kind, earliest, latest, text[span[0] : span[1]]
                if kind in ("full-date-mdy", "full-date-iso", "full-date-text", "holiday", "month-year-slash", "month-year"):
                    # A precise date in this window outranks a bare year inside it.
                    if pattern.search(window):
                        break


def resolve_quit(text: str, *, note_date: date, age: int | None = None) -> Resolution:
    """Return the most recent cigarette-cessation interval found in ``text``."""
    best: Resolution | None = None
    n = 0
    seen_precise: set[tuple] = set()
    for kind, earliest, latest, matched in _candidates(text or "", note_date, age):
        n += 1
        key = (earliest, latest)
        if key in seen_precise:
            continue
        seen_precise.add(key)
        if best is None or latest > best.latest or (latest == best.latest and earliest > best.earliest):
            best = Resolution(earliest, latest, matched.strip(), kind)
    if best is None:
        return Resolution(n_candidates=n)
    best.n_candidates = n
    return best


def apply_one_year_rule(status: str, resolution: Resolution, *, note_date: date) -> tuple[str, str | None]:
    """Combine a native-style status with a resolved cessation interval.

    Only FORMER is re-examined: CURRENT / NEVER / UNKNOWN pass through unchanged.
    """
    if status != "FORMER":
        return status, ("no-mention" if status == "UNKNOWN" else None)
    if resolution.earliest is None:
        return status_from_cessation(None)
    shortest = max((note_date - resolution.latest).days, 0)
    longest = max((note_date - resolution.earliest).days, 0)
    lo, _ = status_from_cessation(shortest)
    hi, _ = status_from_cessation(longest)
    if lo == hi:
        return lo, None
    return status_from_cessation(None)
