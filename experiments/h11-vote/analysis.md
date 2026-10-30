# H11-VOTE — analysis (exploratory)

**Verdict: REFUTED.** Majority vote (TF-IDF, Flan-T5-base FT, Flan-T5-large ZS + guard;
ties → Flan-base) pooled macro-F1 **0.840** (NEVER 0.58), below its best voter Flan-T5-base FT
(0.868). Δ vs TF-IDF +0.061 [+0.013, +0.112]. The weaker voters drag the strong one down more
often than they correct it; the errors are not complementary enough. Use the single best
model.
