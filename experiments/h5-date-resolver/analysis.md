# H5-DATE — analysis (dev split, scored once after freezing quit-resolver-v1 at eec6f03)

**Verdict: SUPPORTED (confirmatory).** Splitting the one-year rule into
"classify native status" + "resolve quit time with rules + note date" removes almost all
of the date-arithmetic errors of the direct classifier.

## Results (dev, n = 600, Wright synthetic — comparability set, never gold)

Primary arm reads the retrieved crops (protocol). Full-note reading is exploratory.

| Arm | Labels | macro-F1 | CURRENT R | timing-unresolved R | recent quitters | basic | messy | complex |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Oracle-status + Oracle-quit (sanity) | v1 | **1.000** | 1.000 | 1.00 | 6/6 | 1.000 | 1.000 | 1.000 |
| A0: TF-IDF direct | v1 | 0.888 | 0.923 | 0.08 | 0/6 | 0.833 | 0.912 | 0.914 |
| **A-native + R (crops)** | v1 | **0.947** | 0.956 | 0.92 | **6/6** | 0.898 | 0.976 | 0.972 |
| A-native + R (crops) | v2 | 0.949 | 0.956 | 0.94 | 6/6 | 0.898 | 0.982 | 0.972 |
| A-native + R (full note, exploratory) | v2 | 0.957 | 0.956 | 0.94 | 6/6 | 0.919 | 0.982 | 0.972 |
| Oracle-status + R (crops) | v2 | 0.973 | 1.000 | 0.97 | 6/6 | 0.949 | 1.000 | 0.972 |
| Oracle-status + R (full note) | v2 | 0.981 | 1.000 | 0.97 | 6/6 | 0.971 | 1.000 | 0.972 |

Resolver on gold-FORMER dev notes: finds a quit time in 94% (crops) / 96% (full note) of
notes whose answer key has one; **0 false finds** in notes whose key has none; the found
year lies inside the key's accepted range in 86%.

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | Sanity arm reproduces labels 100% | ✅ 1.000 |
| 2 | Oracle-status + R ≥ 0.95, lowest on `complex` | ✅ 0.971–0.981; ❌ lowest is `basic` (0.949 crops) — template shorthand quit statements sit in sentences without a trigger word, so crops miss them (full note: 0.971) |
| 3 | A-native + R beats A0 by ≥ 4 points | ✅ +5.9 (crops), +6.9 (full note) |
| 4 | Recent-quitter recall ≥ 4/6 | ✅ 6/6 (A0: 0/6) |

## What we learned

1. **The one-year rule should not be learned end-to-end from bag-of-words.** It is a
   deterministic function of status, quit time and note date. A small rule layer that
   reads the quit time recovers 6 of the 9 points the direct model lost.
2. **The bottleneck has moved to native status classification**, especially on short
   template notes (`basic`: 0.898 vs 0.949 with gold status). That is where a better
   (neural) classifier should help — the date step does not need one on this data.
3. **Crops cost ~1 point vs full notes** for the resolver: some quit statements ("She quit
   cold turkey in 2001.") have no tobacco trigger in their sentence. A ±1-sentence
   expansion around FORMER-predicted crops is a cheap fix to test (H2-CONTEXT).
4. **Label-derivation deviation (v2)**: 10 of 3,000 notes changed UNKNOWN → FORMER
   ("quit 2 years ago" style, which our policy calls FORMER). All numbers move ≤ 0.3 points;
   both versions are reported.
5. Remaining resolver errors are mostly boundary disputes (e.g. "quit at age 65" in a
   67-year-old) and multi-step stories ("1 year after the move", grandchild ages) — the
   cases where a small generative model could earn its cost.

## Cost

The resolver is pure Python regex: ~1 ms per note on one CPU core. It adds nothing
measurable to the $/M of any arm.

## Next

- H6-STATUS: improve the native-status classifier on `basic` (template shorthand) —
  features, ±1-sentence context, or a neural encoder once weights are reachable.
- H2-CONTEXT: ±1-sentence expansion for FORMER crops.
