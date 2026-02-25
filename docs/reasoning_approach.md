# Research Strategy: Reasoning Models in Induction vs. Instruction

## Context

The experiments measure instruction-following vs. induction pressure across models. Reasoning (chain-of-thought tokens) is a confound: it may change behavior independently of the model's base capacity. This matters because:

1. Some models have **mandatory reasoning** (can't be disabled even via OpenRouter)
2. Some models have **optional reasoning** — and the current sweeps ran them *with* reasoning on by default
3. Some models have **no reasoning at all**

The current sweep data is therefore heterogeneous: comparing, say, kimi-k2.5 (1.2M reasoning tokens/proto) with deepseek-v3.2 (0 reasoning tokens) conflates two variables.

---

## Reasoning Status of Current Models

| Model | Reasoning in sweep | Can disable? |
|---|---|---|
| deepseek/deepseek-v3.2 | None | N/A |
| google/gemini-2.5-flash | None | N/A |
| google/gemini-3-flash-preview | ~0 (965 total) | N/A |
| meta-llama/llama-3.3-70b | None | N/A |
| meta-llama/llama-4-maverick | None | N/A |
| meta-llama/llama-4-scout | None | N/A |
| qwen/qwen3.5-plus-02-15 | None | N/A |
| openai/gpt-5.2 | Yes (~135K/proto1) | Yes |
| moonshotai/kimi-k2.5 | Yes (~1.2M/proto1) | Yes |
| anthropic/claude-sonnet-4.6 | Yes (~128K/proto1) | Yes |
| anthropic/claude-opus-4.6 | Yes (~187K/proto1) | Yes |
| google/gemini-2.5-pro | Yes (~944K/proto1) | **No** |
| google/gemini-3-pro-preview | Yes (~905K/proto1) | **No** |
| minimax/minimax-m2.5 | Yes (~734K/proto1) | **No** |

---

## Options

### Option A: "Exclude and Stratify" — most conservative

**Design:** Two clearly separated cohorts:
- **Primary cohort**: models with no reasoning capability + reasoning-optional models run with reasoning disabled (7 pure + 4 re-run = 11 total)
- **Supplementary appendix**: mandatory-reasoning models (gemini-2.5-pro, gemini-3-pro-preview, minimax-m2.5) analyzed separately, explicitly flagged as confounded

**What needs to be run:**
Re-run gpt-5.2, kimi-k2.5, claude-sonnet-4.6, claude-opus-4.6 with `reasoning_enabled: false` for both protocols.

**Pro:** Completely clean apples-to-apples comparison. Easy to defend to reviewers — the main claims are based on a confound-free sample. The supplementary models are acknowledged as different regime.
**Con:** Discards existing data for 4 models (or requires re-running them). Mandatory-reasoning models (some of the most interesting) are marginalized.

---

### Option B: "Reasoning as Treatment" — turns confound into finding

**Design:** For the 4 reasoning-optional models (claude-opus, claude-sonnet, gpt-5.2, kimi-k2.5), run a **paired comparison**: same model, same tasks, with and without reasoning. This creates a within-model experiment:

- Does reasoning help a model resist induction pressure?
- Does reasoning make the model more self-aware of its behavior (Protocol 2 prediction accuracy)?

The non-reasoning models serve as the baseline cohort. The paired data becomes a focused secondary analysis or sub-paper.

**What needs to be run:**
The "reasoning disabled" versions of the 4 optional-reasoning models (both protocols). Cost: ~4× the per-model cost of the current sweep for those models.

**Pro:** Transforms the confound into a publishable result. Directly relevant to current discourse on reasoning models. Covers both behavioral and self-prediction protocols, so the reasoning effect is visible on both.
**Con:** Adds experimental cost. The mandatory-reasoning models (gemini-2.5-pro, minimax) still can't be cleanly compared.

---

### Option C: "Accept Heterogeneity, Annotate" — fastest path to publication

**Design:** Keep all models as-is. In analysis, annotate models by reasoning tier (none / optional-disabled / optional-enabled / mandatory). Make claims at the model-family level rather than individual model level. Present reasoning token count as a covariate in the analysis.

**What needs to be run:** Nothing additional.

**Pro:** Uses all existing data immediately. Rich model coverage (14 models). Can still report meaningful within-tier patterns (e.g., "among non-reasoning models, instruction-following drops at N=X").
**Con:** Primary comparison is confounded. Reviewers will notice that the most heavy-reasoning models (kimi-k2.5, gemini-2.5-pro) look different — and can attribute it to reasoning, not induction dynamics. Weakest scientific design.

---

### Option D: "Non-Reasoning Primary + Mandatory-Reasoning Replication"

**Design:**
- Primary analysis: 7 no-reasoning models only (clean cohort, no re-runs needed)
- For models with optional reasoning: exclude from primary (don't re-run)
- Add mandatory-reasoning models as a separate "replication in reasoning regime" section — not compared to the main cohort, but checked for qualitative consistency (do the same N-value transition patterns hold?)

**What needs to be run:** Nothing additional.

**Pro:** Uses existing data immediately. The primary claim is maximally clean (7 models, all confirmed 0 reasoning tokens). The mandatory-reasoning models are neither dropped nor conflated.
**Con:** Smaller primary cohort (7 vs 14). Claude and GPT-5 family excluded from primary analysis entirely — may concern reviewers about coverage of frontier models.

---

### Option E: "Tiered Model Selection" — principled redesign

**Design:** Define three explicit tiers and be transparent about them:

1. **Tier 1 (No reasoning)**: deepseek-v3.2, gemini-2.5-flash, llama-3.3-70b, llama-4-maverick, llama-4-scout, qwen3.5-plus — primary analysis, clean comparison
2. **Tier 2 (Reasoning disabled)**: claude-sonnet-4.6, gpt-5.2 + optionally kimi-k2.5 — re-run with reasoning off, join primary cohort
3. **Tier 3 (Reasoning enabled)**: gemini-2.5-pro, minimax-m2.5, kimi-k2.5-reasoning-on — analyzed in a dedicated "reasoning model" section

For Tier 2: the existing "reasoning on" data can serve as one arm of a within-model comparison.

**What needs to be run:** Tier 2 models with reasoning disabled (2–3 models × 2 protocols).

**Pro:** Maximally transparent tier structure. Gives you a clean primary cohort (8–9 models) AND preserves optional-reasoning models as paired comparisons. The tier framework maps directly to a methods section that reviewers can follow. Covers frontier models in primary cohort (claude, gpt-5.2).
**Con:** Still can't cleanly include mandatory-reasoning models in primary cohort.

---

## Recommendation

**Option E (Tiered) or Option B (Reasoning as Treatment)** are the most defensible:

- If the primary contribution is *induction vs instruction findings*: use **Option E** — keeps the primary analysis clean, places reasoning models where they belong, uses existing data for paired insight
- If reasoning dynamics are interesting enough to be a secondary contribution: use **Option B** — the within-model paired comparison is a cleaner design and a more compelling supplementary finding

**Immediate zero-cost step** available regardless of choice: re-analyze existing data and **stratify the current results by reasoning tier** (annotation only, no new runs). This establishes the pattern and reveals whether mandatory-reasoning models look systematically different — which informs whether Option B is worth the investment.

---

## Verification

After deciding on a strategy:
1. Update `configs/models.yaml` and `configs/models_reasoning.yaml` to reflect chosen tiers
2. For any re-runs with reasoning disabled: add `reasoning_enabled: false` to model args in the config
3. Check `logs/token_usage.md` post-run to confirm 0 reasoning tokens for disabled models
4. In analysis notebooks: add a `reasoning_tier` column to the model metadata, use it to stratify heatmaps/trajectories
