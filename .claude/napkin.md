# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-03-31 | self | Computed prediction stats on all 7 conditions (incl. token_countries/states) and 15 models (incl. reasoning variants), getting wrong grand means (76.4% acc) | Main results use 5 fixed-output conditions (excl. token conditions) and 13 CORE_MODELS (excl. reasoning variants). Always check paper_stats_config.py for CORE_MODELS and _token_conditions exclusion. The correct grand mean accuracy is 83.5%. |
| 2026-02-16 | self | Used `Selection.crossfilter()` when `Selection.intersect()` was needed | crossfilter EXCLUDES a source's own predicate from marks sharing that source. If hint select reads from traj_data and the main line uses traj_data, crossfilter excludes the hint predicate from the main line! Use intersect when you want ALL predicates applied to ALL marks. |
| 2026-02-18 | self | Passed `color_legend=False` to `plot()` — `colorLegend` is not a valid Observable Plot attribute | To hide color legend, simply omit the `legend` param from `plot()`. Don't invent PlotAttributes — check the `PlotAttributes` TypedDict. |
| 2026-02-18 | self | Put `fx`/`fy` as plot-level kwargs — they're not valid PlotAttributes | `fx` and `fy` are mark-level channels (in MarkOptions), not plot attributes. Pass them to each mark individually. |
| 2026-02-18 | self | Passed `legend="color"` (string) to `plot()` | `legend` param needs a `Legend` object — use `legend("color", ...)` imported from `inspect_viz.plot`. String shorthand does NOT work. |
| 2026-02-18 | self | Used `fy_axis="top"` — invalid | `fy_axis` only accepts `"left"`, `"right"`, `"both"`, or `bool`/`None`. There is no top/bottom option for fy axis. |
| 2026-02-18 | self | `facet_anchor` (MarkOption) is not for positioning axis labels | `facet_anchor` controls WHICH facets a mark is rendered in (e.g. annotations in only the bottom facet). It does NOT move fy/fx axis labels. |
| 2026-02-18 | self | fy axis labels (condition names) getting cut off despite `margin_left` | No built-in way to put fy labels above rows. Solutions: (1) `fy_axis=False` + color legend to replace labels, (2) increase `margin_left` to ~200px for long names. |
| 2026-02-24 | self | Tried to use `--reasoning-effort none` | `reasoning_effort` only accepts `minimal`, `low`, `medium`, `high`, `xhigh`. To fully disable reasoning on OpenRouter, use `reasoning_enabled: false` in models.yaml. |
| 2026-02-24 | self | Tried to specify both `reasoning_effort` and `reasoning_tokens` | Can only specify ONE, not both. OpenRouter provider warns and ignores `reasoning_tokens` if both are set. |
| 2026-02-24 | self | Tried to pass `reasoning` object directly as model arg | OpenRouter provider handles `extra_body` construction internally. Use `reasoning_enabled` (boolean) as top-level model arg, or let inspect-ai's `reasoning_effort` map to OpenRouter's `extra_body.reasoning.effort`. |
| 2026-03-03 | self | Used η² (eta-squared ANOVA) on raw binary data (is_T ∈ {0,1}) | With binary data and few epochs per seed, η² trivially = 1.0 because SS_within = 0. Use **overdispersion ratio** instead: Var(per-seed T-rates) / (p*(1-p)/k). This properly accounts for expected binomial variance. |
| 2026-03-03 | user | Cross-model analyses are meaningless when models have different transition curves | Models behave completely differently at the same N. Don't treat models as interchangeable "raters" of seeds. All seed analyses must be within-model. |
| 2026-03-03 | user | Within-N seed effects (unanimity, overdispersion) don't prove question content matters | A seed unanimous at one N could be noise — if it flips at adjacent N values, it's stochasticity not content. The decisive test is **cross-N stability**: do seed residuals (centered by model mean at each N) correlate across N values? ICC on residual matrix, Kendall's W, seeds consistently above/below median at ALL transition-zone Ns. |
| 2026-03-07 | self | Substring matching (`pattern in text`) in scorers causes false positives | Model outputs like "I can't comply... USA... femur" falsely match as TARGET. Use normalized exact equality: `text.strip().lower().strip("\"'.,!?;:")` then `==`. This handles trailing punctuation but rejects substring containment in longer text. |
| 2026-05-26 | self | `_extract_answer` (last-non-empty-line) was applied to ALL condition types in classify_actual/_multi, destroying multi-line answers — code blocks reduced to closing ``` fence → judges saw only "```" → "unknown" | `_extract_answer` exists to strip INLINE reasoning (Hermes emits English reasoning then answer on last line). Only safe for single-line-answer types. Now gated by `SINGLE_LINE_ANSWER_TYPES = {static, token_pattern, classify_question, language}`; code/style/persona/preference/variety keep full multi-line output. Empirically Hermes only emits inline reasoning on single-line conditions, so scoping is sufficient. |
| 2026-03-08 | user | Reasoning effort suffix format was `"modelname (effort)"` | Should be `"modelname-effort"` (hyphen-separated). Similarly `reasoning_enabled=true` → `"modelname-reasoning"`. |
| 2026-03-08 | self | `evals_df()` shows `model_args: <NA>` even when model_args has `reasoning_enabled=True` | evals_df doesn't serialize model_args properly. Must use `read_eval_log(f, header_only=True)` to read `log.eval.model_args` directly. Added `_load_model_args_map(log_dirs)` helper in prepare_viz_data.py. |
| 2026-05-26 | self | Assumed `inspect_ai.score()` (the top-level rescoring function) was async | It's SYNCHRONOUS (`def score(...)`, not `async def`). Don't use `await` or `asyncio.run()`. Distinct from `inspect_ai.scorer.score(state: TaskState)` which is the inline solver-time helper. Signature: `score(log: EvalLog, scorers, *, action="append"|"overwrite", copy=True) -> EvalLog`. Takes EvalLog object (not path) — use `read_eval_log` → `score` → `write_eval_log`. |
| 2026-05-26 | self | Said built-in reducers "discard" non-first scores' metadata | More precisely: `_reduced_score` (used by mode/mean/median/max/at_least/pass_at) keeps `scores[0].metadata` and drops the rest. The intermediate Score list in `multi_scorer` is never logged — it's a local variable consumed by the reducer. To preserve per-judge data, a CUSTOM reducer must capture it into the returned Score before this happens. Better still: precompute agreement stats (rate, unanimous, n_target) inside the reducer instead of storing raw votes — saves notebook extraction code. |

## User Preferences
- Project is single-purpose (induction vs instruction evals) - no need for extra subfolder nesting
- Use OpenRouter as model provider
- Don't reinvent the wheel — always check framework docs for native patterns before building custom solutions
- Keep things transparent and controllable — user should see exactly what's running
- Plot-generating scripts belong in marimo notebooks (notebooks/marimo/), not in scripts/
- In appendix: use `[H]` placement (requires `\usepackage{float}`) so figures appear exactly where placed
- LaTeX packages NOT loaded by colm2026_conference.sty: `amsmath`, `bm`, `amssymb` — use `\textbf{$...$}` and `\mathrm{}` instead
- COLM 2026 is **single-column** (5.5in textwidth, no `\twocolumn`) — use `figure`/`table`, NOT `figure*`/`table*`
- COLM style: table/figure captions go **below** content (caption after `\end{tabular}`, before `\end{table}`)
- COLM style: minimum font size in tables/figures is `\small` — `\footnotesize` is forbidden
- COLM style: need `\appendix` before appendix sections or they get regular section numbers
- Citations must use natbib: `\citet{}` for inline, `\citep{}` for parenthetical — not plain `\cite{}`
- When hint/no-hint T1 data have different N grids, filter to neutral condition only for clean comparison
- Temperature comparison: the key result is dispersion (SD of per-model Δ), not mean; report SD prominently

## Inspect-AI

### Core Patterns
- `inspect eval` + `-T` / `--task-config`: single task runs. Model gen settings (temperature etc.) are CLI flags here.
- `eval_set()` Python API: the native way to run parameterized grids (conditions x Ns x models). Docs show `[task_fn(params) for params in grid]` pattern.
- `inspect eval-set` CLI: runs task files/directories with retries. `--task-config` is GLOBAL — applies same params to all tasks, cannot vary per task.
- `@task` functions return ONE Task. No list, no dynamic generation. Keep them simple.
- Different conditions need different scorers, so they must be separate Tasks (not mixed in one dataset) unless you write a meta-scorer dispatching on metadata.
- `Task(epochs=N)` repeats the dataset N times — use this instead of generating N duplicate samples.
- inspect-ai reads `.env` automatically (has its own dotenv loading). No need for `load_dotenv()` in task files.
- Inspect-AI Sample input can be `list[ChatMessage]` for pre-built conversations
- Use `MemoryDataset` for dynamically generated datasets
- Use `store.set()` / `store.get()` in solvers to pass data between steps

### Reasoning Models
(o-series, Claude extended thinking, DeepSeek-R1, etc.)
- `state.output.completion` and `message.text` only include `ContentText` blocks — `ContentReasoning` is excluded
- Scorers using `.completion` are already safe and won't score thinking tokens
- Reasoning content preserved in logs, displayed separately in Inspect View
- Access reasoning via `message.content` list filtering for `type == "reasoning"`
- Reasoning tokens tracked in `ModelUsage.reasoning_tokens`
- CLI options: `--reasoning-effort`, `--reasoning-tokens`, `--reasoning-summary`, `--reasoning-history`
- **CRITICAL**: Can only specify ONE of `reasoning_effort` OR `reasoning_tokens`, not both
- `reasoning_effort` values: `minimal`, `low`, `medium`, `high`, `xhigh` — NOT `none`
- **OpenRouter-specific**: To fully disable reasoning, pass `reasoning_enabled: false` as a model arg (top-level, in models.yaml). OpenRouter provider handles constructing `extra_body` internally.

## Inspect-Viz

### Selection Semantics
- `Selection.intersect()`: ALL predicates apply to ALL sources. Use this for simple shared filtering across Data instances.
- `Selection.crossfilter()`: each source's OWN predicate is EXCLUDED from itself. Designed for reciprocal filtering (e.g., two histograms filtering each other). If an input reads from Data A and a mark also uses Data A, the input's predicate is excluded from that mark.
- The "source" is determined by the Data instance. A `select(data=X, target=sel)` associates its clause with source X. A `line(X, filter_by=sel)` evaluates sel for source X.
- For filtering one Data by another's inputs (e.g., neutral baseline filtered by hint selected from traj_data), use `Selection.intersect()` — the hint predicate applies to both traj_data marks AND neutral_data marks.

### Marks & Channels
- `fx`/`fy` are mark-level channels (in MarkOptions), not plot attributes. Pass them to each mark individually.
- `stroke_dash` is NOT a valid MarkOption. The correct parameter is `stroke_dasharray`.
- `color_scheme` values must be lowercase (e.g., `"rdylgn"` not `"RdYlGn"`).
- `font_weight` should be a direct mark kwarg, not inside `styles={}`.
- `fx_label=None` hides facet column header labels. Remove it to let Observable Plot show facet values as column headers.
- Use `fx="hint"` faceting to create side-by-side panels in a single plot instead of two separate plots.
- `scores_by_factor()` requires a boolean factor. For condition-specific labels, build a custom plot with `rule_y` using `condition` as stroke and `model` as `fy` facet.
- inspect-viz `channels` dict values must be column names (strings), not lambdas.

### Data Patterns
- inspect-viz `Data` has a built-in selection. ALL `select()`/`slider()` inputs targeting the same `Data` instance share that selection and filter ALL plots using that `Data`. Create separate `Data.from_file()` instances per filtering scope — e.g., trajectory section (condition_pair + hint only) vs factor section (condition_pair + hint + N) need DIFFERENT Data instances or the N filter bleeds into the trajectory.
- **n_turns handling**: n_turns is converted to string in `prepare_viz_data.py`. String sorting is lexicographic ("1", "10", "2"). Use `sorted(..., key=int)` for numeric order. For line marks: create temp int column for `sort_values()` to ensure proper line connections (DuckDB query order, not x_domain order).
- For discrete N values, use `select(data, column="n_turns")` not `slider()` — slider implies continuous range.
- `select()` with `value="auto"` may not initialize properly. Use explicit `options=domain_list, value=domain_list[0]`.
- Integer columns (e.g., `n_turns` as int64) may not work properly with `select()` widget. Convert to string in data prep.
- To exclude values from a select dropdown, load a pre-filtered parquet as the Data source.
- For overlaying data from different sources (e.g., neutral baseline + selected condition), use separate `Data.from_file()` instances and pass separate line marks to the same `plot()` call.
- `scores_by_factor()` does NOT accept `filter_by` — but it respects the Data's built-in selection from `select()`/`slider()` inputs.

### Common Pitfalls
- When aggregated data has multiple grouping dimensions, the trajectory `line` mark with `stroke=model` will connect ALL points for the same model across different conditions/hints, creating garbage lines. Always filter to a single condition + hint before plotting trajectories.
- `filter_by=data.selection` on cascading selects causes deadlocks after the first interaction. Use a single dropdown instead of cascading (condition names are unique and self-descriptive).
- ANY non-column value in a mark kwarg that Mosaic tries to resolve as a channel causes the JS error `can't access property "as", n.channelField(...) is null`. This includes: (1) string constants like `stroke="white"` → move to `styles={"stroke": "white"}`; (2) `stroke=None` → just omit `stroke` entirely. Integer constants like `stroke_width=4` are safe.
- `nearest_x(sel, channels=["stroke"])` crashes if any mark lacks that channel. Use `nearest_x(sel, fields=["model"])` instead — queries raw data field, doesn't require all marks to have that channel.
- `overview_heatmap()` (and any cell/text marks with no `filter_by`) will overlap when data has multiple rows per cell position. Always pass `filter_by=selection` to marks in heatmaps.
- `plot()` ALWAYS sets height in the spec — `height=None` falls back to `width/1.618`, never "auto". For bullet graphs with many models, compute height from data: `n_conditions * n_models * 30 + 130`.
- `avg()` in Mosaic creates preagg tables that can fail ("Table preagg_xxx does not exist") — use Observable Plot's native transform/reducer system instead.

### Quarto & Layout
- Quarto HTML default column is ~750px — use `#| column: screen` on specific cells for full-width plots, or `page-layout: full` in YAML for the whole notebook.
- For sidebar-style controls: use `page-layout: full` in YAML + Bootstrap grid divs in content.
- inspect-viz select widgets use **TomSelect** (not native `<select>`). CSS must target `.ts-wrapper` and `.ts-control`, NOT `select`.
- Sidebar overflow fix (via `include-in-header`):
  ```css
  .g-col-3 { min-width: 0; }
  .g-col-3 .cell-output-display { overflow-x: hidden; }
  .g-col-3 .mosaic-widget label { flex-direction: column; align-items: stretch; min-height: unset; font-size: 0.85rem; }
  .g-col-3 .ts-wrapper { width: 100% !important; margin-left: 0 !important; margin-right: 0 !important; }
  .g-col-3 .ts-control { min-width: 0 !important; font-size: 0.85rem; }
  ```
- Sidebar div: `::: {.g-col-3 .bg-light .p-3 style="position: sticky; top: 1rem; align-self: start;"}`
- Main div: `::: {.g-col-9}` — enclose both in `:::: {.grid} ... ::::`
- Replace `hconcat(select1, select2)` with two separate `select()` cells stacked in sidebar.
- When a Selection is shared across multiple charts, wrap all charts in one grid with a sticky sidebar.
- Remove `#| column: page` from chart cells inside grids; reduce width from 1000 to 900.
- On Windows/MSYS2, `$(pwd)` produces `/c/Users/...` paths that Python can't read. Use relative paths for Quarto `-P` params and resolve them in the notebook.
- `QUARTO_PYTHON` env var must point to the .venv Python for Quarto to find packages. Set it in justfile.

## Data Access
- `samples_df(logs='logs/protocol1', columns=SampleSummary)` gives per-trial data (48,240 rows)
- `metadata_trial_index` is the question seed (0-24 for static, deterministic via `random.Random(trial_index)`)
- Model name comes from `EvalModel` columns in `samples_df(..., columns=SampleSummary + EvalModel)` — no path parsing needed
- PyArrow-backed string columns from inspect-ai: `.astype(str)` keeps ArrowExtensionArray and breaks `.str` accessor → use `.astype(object)` instead
- Score columns: `score_pattern_match`, `score_set_membership_scorer` — values are inspect-ai strings `"C"` (CORRECT=1) / `"I"` (INCORRECT=0). Use `_SCORE_MAP = {"C": 1.0, "I": 0.0, "P": 0.5}` + `.map()` — NOT `pd.to_numeric` which produces all NaN
- `evals_df()` with `EvalModel+EvalTask+EvalScores` crashes on current inspect-ai version (Index attribute error) — use log path parsing instead
- `evals_df` and `samples_df` both auto-include `eval_id` as join key. Use `eval_id` not `task_id` (there is no `task_id` column).
- `model_usage` may be a `ModelUsage` object or dict — handle both. Filter with `max_reasoning_tokens` param (None=no filter, 0=strict, 1000=allow noise-level).

## Project Domain

### Condition Naming Convention
- Instruction is always the primary reference point (instruction first in name)
- Conditions with values/truth alignment axis: `{type}_{aligned|misaligned}_{topic}` (topic = shared subject, e.g. cats, earth)
  - `value_aligned_cats` (T=love cats, P=hate cats), `value_misaligned_cats` (T=hate cats, P=love cats)
  - `factual_aligned_earth` (T=earth is round, P=flat), `factual_misaligned_earth` (T=earth is flat, P=round)
  - `preference_aligned_cats` (T=love cats weaved, P=hate cats weaved), `preference_misaligned_cats` (T=hate, P=love)
- Conditions without alignment axis: `{type}_{instruction}_{pattern}` — instruction first
  - `persona_casual_formal` = instruction=casual, pattern=formal
  - `language_ru_fr` = instruction=Russian, pattern=French
  - `token_countries_states` = instruction=EU countries, pattern=US states
- For future binary preference extensions, same pattern: `preference_aligned_X` / `preference_misaligned_X`

### Protocols
- Inspect-AI evaluation framework for testing LLM behavior under induction pressure
- Protocol 1: Behavioral baseline (does model follow instruction or pattern?)
- Protocol 2: Self-prediction (can model predict its own behavior?)
- Key variables: Pattern (P), Target (T), induction strength (N turns)
- Condition pairs isolate a single dimension of variation for cleaner analysis
- Key comparison: does instruction alignment (with values/truth) affect following rates?

### Report Metrics
- Summary statistics (static vs dynamic avg IF): in behavioral_analysis.qmd. STATIC_CONDITIONS = {neutral, value_aligned_cats, value_misaligned_cats, factual_aligned_earth, factual_misaligned_earth}
- Behavior boundaries: N_T_width (count of N values ≥90% IF) and N_P_start (first N ≤10% IF), computed per (model, condition) by averaging over instruction templates. Displayed as pivot tables in behavioral_analysis.qmd
- Behavioral/prediction consistency heatmaps: stderr per (model, condition, n_turns) averaged over instructions; use color_scheme="reds", show_text=False. In behavioral_analysis.qmd (score_stderr) and prediction_analysis.qmd (per metric stderr)
- `overview_heatmap()` has `color_scheme` (default "rdylgn") and `show_text` (default True) params
- ALIGNMENT_AXIS_PAIRS = {"value", "factual", "preference"} in prepare_viz_data.py — only these condition pairs have a true aligned/misaligned axis. Other pairs (token, language, persona, style) are direction-flipped without alignment semantics. Use this to filter paired bullet graphs.

### Scripts & Key Files
- `prepare_viz_data.py`: distinguishes behavioral vs prediction logs via `task_name` column ("behavioral_baseline" vs "self_prediction") — NOT by score column names. Also filters to the relevant task rows so mixed folders work correctly.
- `format_check.py` and `classify_format()`: naming is `{instruction}_{pattern}` so instruction=first word=target. `style_uppercase_lowercase` = instruction=uppercase = target=UPPERCASE.
- `scripts/generate_hardcoded_responses.py` — generates 9 LLM files + 2 computed (uppercase/lowercase from style_base)
- `style_lowercase_uppercase` and `style_uppercase_lowercase` conditions need `pattern_data_key` set to `style_uppercase`/`style_lowercase` respectively
- Preference conditions use `questions_subjective.json` question bank (set via `question_bank` field on Condition)

### Inspect Score Flags (Two `overwrite`s)
- `--action overwrite` vs `--action append`: controls whether the new SCORES replace or are added alongside existing ones in the log.
- `--overwrite` (separate flag): controls whether the LOG FILE is written in place. Without it, inspect prompts interactively "overwrite/create" and defaults to "create" — under xargs (empty stdin) every file aborts. For batch rescoring you almost always want BOTH: `--action overwrite --overwrite`.

### Log Directory Layout
- `.eval` files live at `logs/<protocol>/T<temp>/<...subfolders>/<model_name>/*.eval` — NOT flat under `logs/protocol1/`. Any batch operation (rescore, samples_df iteration) must recurse: `**/*.eval` globstar in bash, `Get-ChildItem -Recurse` in PowerShell, or `find` with `-name '*.eval'`.

### Git Worktrees
- Never place worktrees inside the main repo directory — git sees them as untracked files. Use sibling directories (e.g. `../response-variety`).
- `configs/` is gitignored — use `git add -f configs/file.yaml` to track config files in worktrees.
- Each worktree creates its own `.venv` when running `uv run` — first command in a new worktree is slow (installs packages).
