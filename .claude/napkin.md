# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-02-13 | self | Used `generate_fn` as parameter name in solver | Inspect-AI solver signature requires param be named exactly `generate` (type narrowing issue) |

## User Preferences
- Project is single-purpose (induction vs instruction evals) - no need for extra subfolder nesting
- Use OpenRouter as model provider

## Patterns That Work
- Inspect-AI Sample input can be `list[ChatMessage]` for pre-built conversations
- Use `MemoryDataset` for dynamically generated datasets
- Use `store.set()` / `store.get()` in solvers to pass data between steps
- Inspect-viz uses Quarto notebooks (`.qmd`) for publishing - not raw .ipynb
- `scores_by_factor()` requires boolean factor - decompose conditions into natural pairs:
  - `value_pattern` vs `value_target` → factor: instruction_value_aligned
  - `factual_pattern` vs `factual_target` → factor: instruction_truth_aligned
  - `neutral` → use hint as factor

## Patterns That Don't Work
- Conditions with `pattern_data_key` (language, persona, code, preference, style_long) require `data/hardcoded_responses/*.json` files to exist — will fail at sample build time if missing

## Domain Notes
- Inspect-AI evaluation framework for testing LLM behavior under induction pressure
- Protocol 1: Behavioral baseline (does model follow instruction or pattern?)
- Protocol 2: Self-prediction (can model predict its own behavior?)
- Key variables: Pattern (P), Target (T), induction strength (N turns)
- Condition pairs isolate a single dimension of variation for cleaner analysis
- Key comparison: does instruction alignment (with values/truth) affect following rates?
