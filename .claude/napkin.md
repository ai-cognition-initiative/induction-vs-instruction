# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-02-13 | self | Used `generate_fn` as parameter name in solver | Inspect-AI solver signature requires param be named exactly `generate` (type narrowing issue) |
| 2026-02-13 | self | Tried to build a custom runner when inspect has native patterns | Use `eval_set()` Python API for parameterized grids — this IS the inspect-native way. Don't try dynamic `@task` generation or unified meta-scorers. |
| 2026-02-13 | self | Tried to make `@task` accept comma-separated lists to avoid a runner script | Keep `@task` functions simple (single condition, single n_turns). Grid expansion belongs in a script calling `eval_set()`. |

## User Preferences
- Project is single-purpose (induction vs instruction evals) - no need for extra subfolder nesting
- Use OpenRouter as model provider
- Don't reinvent the wheel — always check framework docs for native patterns before building custom solutions
- Keep things transparent and controllable — user should see exactly what's running

## Patterns That Work
- Inspect-AI Sample input can be `list[ChatMessage]` for pre-built conversations
- Use `MemoryDataset` for dynamically generated datasets
- Use `store.set()` / `store.get()` in solvers to pass data between steps
- Inspect-viz uses Quarto notebooks (`.qmd`) for publishing - not raw .ipynb
- `scores_by_factor()` requires boolean factor - decompose conditions into natural pairs:
  - `value_pattern` vs `value_target` → factor: instruction_value_aligned
  - `factual_pattern` vs `factual_target` → factor: instruction_truth_aligned
  - `neutral` → use hint as factor

## Inspect-AI Patterns
- `inspect eval` + `-T` / `--task-config`: single task runs. Model gen settings (temperature etc.) are CLI flags here.
- `eval_set()` Python API: the native way to run parameterized grids (conditions x Ns x models). Docs show `[task_fn(params) for params in grid]` pattern.
- `inspect eval-set` CLI: runs task files/directories with retries. `--task-config` is GLOBAL — applies same params to all tasks, cannot vary per task.
- `@task` functions return ONE Task. No list, no dynamic generation. Keep them simple.
- Different conditions need different scorers, so they must be separate Tasks (not mixed in one dataset) unless you write a meta-scorer dispatching on metadata.
- `Task(epochs=N)` repeats the dataset N times — use this instead of generating N duplicate samples.
- inspect-ai reads `.env` automatically (has its own dotenv loading). No need for `load_dotenv()` in task files.
- **Reasoning models (o-series, Claude extended thinking, DeepSeek-R1, etc.):**
  - `state.output.completion` and `message.text` only include `ContentText` blocks — `ContentReasoning` is excluded
  - Scorers using `.completion` are already safe and won't score thinking tokens
  - Reasoning content preserved in logs, displayed separately in Inspect View
  - Access reasoning via `message.content` list filtering for `type == "reasoning"`
  - Reasoning tokens tracked in `ModelUsage.reasoning_tokens`
  - CLI options: `--reasoning-effort`, `--reasoning-tokens`, `--reasoning-summary`, `--reasoning-history`

## Patterns That Don't Work
- Conditions with `pattern_data_key` (language, persona, code, preference, style_long) require `data/hardcoded_responses/*.json` files to exist — will fail at sample build time if missing

## Domain Notes
- Inspect-AI evaluation framework for testing LLM behavior under induction pressure
- Protocol 1: Behavioral baseline (does model follow instruction or pattern?)
- Protocol 2: Self-prediction (can model predict its own behavior?)
- Key variables: Pattern (P), Target (T), induction strength (N turns)
- Condition pairs isolate a single dimension of variation for cleaner analysis
- Key comparison: does instruction alignment (with values/truth) affect following rates?
