# H5-DATE — protocol (locked before running)

**Date locked:** 2026-10-30 · **Type:** CONFIRMATORY · **Data:** Wright synthetic benchmark (comparability set, never gold)

## Hypothesis

E-wave-1 showed the TF-IDF floor drops from 0.979 (Wright native labels) to 0.888
macro-F1 under our one-year rule, and the errors are date arithmetic: 30/36
`UNKNOWN(timing-unresolved)` and 6/6 recent quitters were predicted FORMER.

**H5-DATE:** the one-year rule is a *deterministic* function of (status, quit time,
note date). A classifier only needs to learn the native status
(CURRENT / FORMER / NEVER / UNKNOWN-no-mention); a rule-based quit-time resolver plus
the note's reference date (EHR metadata, not note text) can then apply the rule.
Mechanism: the hard part moves from "learn date arithmetic from bag-of-words" (impossible)
to "find and parse a quit expression" (regex-tractable for most phrasings).

## Arms (all on the same patient-disjoint split as E-wave-1, seed 20261029)

| Arm | Description |
|---|---|
| A0 | Baseline from E-wave-1: TF-IDF crops → LinearSVM trained directly on one-year labels |
| A-native+R | TF-IDF crops → LinearSVM trained on **native** labels; if it predicts FORMER, the resolver reads quit time from the crop text and maps through `label_policy.status_from_cessation` using the note date; no quit time found ⇒ `UNKNOWN(timing-unresolved)` |
| Oracle-status+R | Gold native status + resolver (upper bound for the resolver alone) |
| Oracle-status+Oracle-quit | Gold native status + answer-key quit fields (sanity check: must reproduce the one-year labels exactly) |

Resolver precedence (date-interval logic, same as the label derivation):
explicit day > month > year > relative ("N years/months ago", "last year") > stop age with
current age. Each expression yields an [earliest, latest] cessation interval; if the whole
interval falls on one side of 365 days the rule decides, otherwise `UNKNOWN(timing-unresolved)`.
If several quit expressions disagree, take the most recent one that is not negated or about
alcohol/other substances.

## Development discipline

- The resolver is written and debugged **on train-split notes only**. Dev notes are scored
  exactly once per arm after the resolver is frozen (frozen = committed).
- Test split stays locked.

## Predictions (stated before running)

1. Oracle-status+Oracle-quit reproduces one-year labels at 100% (pipeline sanity).
2. Oracle-status+R reaches ≥ 0.95 one-year macro-F1 on dev overall; lowest on `complex`
   (multi-step date reasoning such as "three years after her husband died in 2010").
3. A-native+R beats A0 by ≥ 4 macro-F1 points (0.888 → ≥ 0.93) and raises CURRENT recall.
4. Recent-quitter recall rises from 0/6 to ≥ 4/6.

## What would refute it

A-native+R ≤ A0 + 1 point, or Oracle-status+R < 0.90 — meaning quit expressions are too
varied for rules and the date step needs a learned model (arm G or a dedicated extractor).

## Metrics

macro-F1 (primary), CURRENT recall/precision, per-class F1, per-condition macro-F1,
`UNKNOWN(timing-unresolved)` recall, resolver coverage (share of FORMER-status notes where
any quit time is found) and resolver accuracy against the answer key's quit-year range.
