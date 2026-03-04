# Recommended Evaluations & Analyses

Summary of recommendations for maximizing insight from existing data and prioritizing new compute.

---

## Part 1: Analyses with Existing Data (15 trials, temperature 1)

These require no additional compute — just new analysis code on data already collected.

### 1.1 Paired P1–P2 comparison

Because the 15 question seeds are shared across Protocols 1 and 2, every P2 trial has a matched P1 trial with identical questions. For each (condition, N, hint, question-seed) tuple, compute:

- Δ = P2(output=T) − P1(output=T)

Test whether Δ is consistently positive or negative across the 15 pairs using a paired sign test or Wilcoxon signed-rank test. This is the most statistically powerful way to test whether the act of predicting changes behavior, given the existing sample size. It controls for question-set effects automatically.

### 1.2 Cross-condition question-set correlation

For each of the 15 question seeds, compute the average P(output=T) across multiple conditions and N values. Rank the 15 seeds by this average. If the ranking is consistent across conditions (seed #3 always produces more T than seed #7), that's a genuine question-content effect — not sampling noise. Test with Kendall's W (concordance across conditions) or pairwise Spearman correlations between condition-specific seed rankings.

This reveals whether certain question content anchors the model in instruction-following mode. If significant, inspect the high-T and low-T question sets to identify what contextual features drive the effect.

### 1.3 Variance as transition indicator

For each (condition, hint) combination, compute the standard deviation of P(output=T) across the 15 samples at each N value. Plot SD(N). The peak should coincide roughly with the behavioral transition point (where the mean crosses 0.5). If the variance peak occurs at a different N than the mean crossover, the transition is asymmetric.

Compare variance profiles across conditions. Conditions where the pattern is value-misaligned vs. value-aligned may show different variance signatures, revealing whether value alignment provides a more robust anchor against question-content perturbation.

### 1.4 P2 calibration analysis (coarse)

For each (condition, N) cell, compute the 2×2 joint distribution of (prediction, behavior) from 15 trials. The per-cell counts will be small, but you can aggregate across N values by regime:

- **Low N** (1–5): pool all trials, compute calibration
- **Transition** (N values where 4–11 out of 15 are T): pool all trials
- **High N** (25+): pool all trials

This gives three coarse calibration estimates with ~45–75 observations each (15 trials × 3–5 N values per regime), which is enough to detect large calibration failures.

### 1.5 Endorsement consistency check (Protocol 3)

For each (condition, N, prefill_type) cell, check whether endorsement responses cluster or scatter across the 15 trials. If all 15 endorse (or all 15 disavow), the finding is robust despite the small sample. If it's 8/7, you can't say much. Report the cells where the result is clear and flag the ambiguous ones as needing more data.

---

## Part 2: New Evaluations (prioritized by insight per compute)

### Priority 1: Additional P2 trials near the transition (2 focal models)

**Why**: Protocol 2's key question — does predicting change behavior? — requires comparing P1 and P2 rates, and the interesting signal is in the off-diagonal cells (predict T, output P and vice versa) which are sparsely populated at 15 trials.

**What to run**: Select 2 models (one early-transition, one late-transition, identified from existing P1 data). For each:

- Identify 3 transition-adjacent N values from existing data (where P(output=T) is between 0.25 and 0.75)
- Run 35 additional P2 trials at those N values (new question seeds), bringing total to 50
- Run 35 additional matched P1 trials at the same N values with the same new seeds

**Conditions**: Start with `neutral`, `value_pattern`, `value_target` (3 conditions)

**Compute**: 2 models × 3 N values × 3 conditions × 2 protocols × 35 trials = 1,260 samples

**What this unlocks**:
- Reliable 2×2 calibration tables at the transition
- Paired P1–P2 behavioral comparison with enough power to detect a 15-point shift
- Per-question-set analysis of whether specific questions drive prediction failures

### Priority 2: Additional P1 trials near the transition (same 2 focal models)

**Why**: The behavioral baseline is the foundation everything else builds on. Precise transition curves for two models give you clean figures for the paper.

**What to run**: Same 2 models, same 3 transition-adjacent N values, same 3 conditions. 35 additional P1 trials (beyond the 35 already added in Priority 1 as matched controls), giving 50 additional P1 trials total per cell (65 total with original 15).

**Compute**: 2 models × 3 N × 3 conditions × 35 trials = 630 samples

**What this unlocks**:
- Tight confidence intervals on P(output=T) at the transition
- Reliable transition point estimates (sigmoid fit with narrow CI)
- Enough data to test whether variance peaks at the transition (Section 1.3) with real statistical power

### Priority 3: Additional P3 (endorsement) trials near the transition

**Why**: Endorsement is the most novel protocol. The key claims — does the model endorse induction-driven responses, does endorsement track instruction or behavior — need to be well-supported.

**What to run**: Same 2 models, same 3 transition-adjacent N values, 3 static conditions, both prefill types. 35 additional trials per cell.

**Compute**: 2 models × 3 N × 3 conditions × 2 prefill × 35 trials = 1,260 samples

**What this unlocks**:
- Reliable endorsement rates for T-prefill vs P-prefill at the transition
- Enough data to detect whether endorsement shifts with N or stays constant
- Credible claims about the "conflict awareness" (ambiguous) category, which is likely a minority response that needs more samples to characterize

### Priority 4: Breadth across models at 15 trials

**Why**: Generalization across model families is important for the paper but doesn't need high precision per model.

**What to run**: Additional models (beyond the 2 focal models) at the existing 15 trials per cell, covering the full N grid and static conditions only.

**Compute**: Depends on number of models. Per model: 5 conditions × 12 N × 2 hint × 15 trials = 1,800 samples for P1; same for P2; double for P3.

**What this unlocks**:
- A table of approximate transition points across model families
- Qualitative claims about which architectures / training approaches are more resistant to induction pressure
- Evidence that the phenomenon is not model-specific

### Priority 5 (stretch): Variation conditions

**Why**: Interesting extensions but introduce scorer noise on top of sampling noise.

**What to run**: 15 trials across the full grid for variation conditions, on the 2 focal models only.

**What this unlocks**: Supplementary material showing the effect generalizes beyond static token pairs.

---

## Part 3: Analyses Unlocked by New Data

These become possible once Priority 1–3 evaluations are complete.

### 3.1 Precise P1–P2 behavioral comparison (from Priority 1)

With 50 paired trials at transition N values, run the paired analysis from Section 1.1 with real statistical power. Report the estimated effect size of prediction on behavior, with confidence intervals. If the prediction intervention shifts P(output=T) by even 10 points, that's a meaningful "introspection changes behavior" finding.

### 3.2 Full calibration analysis (from Priority 1)

With 50 trials, the 2×2 joint distribution of (prediction, behavior) has enough observations to analyze per N value rather than pooling across regimes. For each N near the transition:

- Compute P(output=T | predicted T) vs P(output=T | predicted P). If prediction is informative, these should differ.
- Compute P(predicted correctly). Plot calibration as a function of N.
- Test whether the model is *over-confident* (predicts T but outputs P, i.e., doesn't recognize its own susceptibility to induction) or *under-confident* (predicts P but outputs T, i.e., underestimates its instruction-following ability).

The direction of miscalibration is theoretically interesting: over-confidence in instruction-following suggests the model's self-model doesn't account for induction pressure.

### 3.3 Endorsement × behavioral regime (from Priority 3)

With 50 trials per (N, prefill) cell, split trials by what the model *would have done* (using Protocol 1 base rates at that N) and ask whether endorsement tracks the prefill or the likely counterfactual behavior. Specifically:

- At an N where P1 shows 70% T-output, prefill P and ask for endorsement. The model "would have" output T most of the time. Does it disavow P (tracking its likely behavior) or endorse P (suggestible to prefill)?
- At an N where P1 shows 30% T-output, prefill T. The model "would have" output P. Does it endorse T (tracking instruction) or disavow T (tracking its shifted disposition)?

### 3.4 Question-set moderation of prediction accuracy (from Priority 1)

With 50 trials across multiple question sets, test whether the question sets that produce more T in P1 also produce more accurate self-prediction in P2. If yes: question content that anchors instruction-following also supports self-knowledge. If no: the model's behavioral robustness and metacognitive accuracy are dissociable, which is a subtle finding about the separability of instruction-following and self-modeling.

### 3.5 Cross-protocol variance decomposition (from Priorities 1–3)

With enough data, decompose total variance at each N into:

- **Between-question-set variance**: does the choice of questions matter?
- **Within-question-set, between-sample variance**: pure sampling noise at temperature 1
- **Between-protocol variance**: does P1 vs P2 vs P3 change outcomes for the same question set?

This requires the nested design from Priority 1 (multiple samples per question set). Even a rough decomposition tells you whether your results are primarily driven by question content, sampling noise, or protocol structure — which informs both interpretation and future experimental design.

---

## Summary Table

| What | Data needed | Compute | Key insight |
|---|---|---|---|
| Paired P1–P2 test | Existing | None | Does predicting change behavior? |
| Question-set correlations | Existing | None | Is question content a real moderator? |
| Variance-as-transition | Existing | None | Independent transition point estimate |
| Coarse P2 calibration | Existing | None | Gross calibration failures by regime |
| More P2 at transition | New (Priority 1) | ~1,260 samples | Precise calibration, prediction effect size |
| More P1 at transition | New (Priority 2) | ~630 samples | Clean behavioral curves |
| More P3 at transition | New (Priority 3) | ~1,260 samples | Reliable endorsement findings |
| More models at 15 trials | New (Priority 4) | ~1,800/model | Generalization claims |
| Variation conditions | New (Priority 5) | ~1,800/model | Supplementary generalization |
