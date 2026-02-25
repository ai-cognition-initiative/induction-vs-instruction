# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-02-13 | self | Used `generate_fn` as parameter name in solver | Inspect-AI solver signature requires param be named exactly `generate` (type narrowing issue) |
| 2026-02-13 | self | Tried to build a custom runner when inspect has native patterns | Use `eval_set()` Python API for parameterized grids — this IS the inspect-native way. Don't try dynamic `@task` generation or unified meta-scorers. |
| 2026-02-13 | self | Tried to make `@task` accept comma-separated lists to avoid a runner script | Keep `@task` functions simple (single condition, single n_turns). Grid expansion belongs in a script calling `eval_set()`. |
| 2026-02-16 | self | Used `Selection.crossfilter()` when `Selection.intersect()` was needed | crossfilter EXCLUDES a source's own predicate from marks sharing that source. If hint select reads from traj_data and the main line uses traj_data, crossfilter excludes the hint predicate from the main line! Use intersect when you want ALL predicates applied to ALL marks. |
| 2026-02-16 | self | Kept iterating on broken viz without stepping back to understand the framework | After 2-3 failed attempts at the same problem, STOP and read the actual framework docs instead of guessing. |
| 2026-02-18 | self | Passed `color_legend=False` to `plot()` — `colorLegend` is not a valid Observable Plot attribute | To hide color legend, simply omit the `legend` param from `plot()`. Don't invent PlotAttributes — check the `PlotAttributes` TypedDict. |
| 2026-02-18 | self | Put `fx`/`fy` as plot-level kwargs — they're not valid PlotAttributes | `fx` and `fy` are mark-level channels (in MarkOptions), not plot attributes. Pass them to each mark individually. |
| 2026-02-18 | self | Passed `legend="color"` (string) to `plot()` | `legend` param needs a `Legend` object — use `legend("color", ...)` imported from `inspect_viz.plot`. String shorthand does NOT work. |
| 2026-02-18 | self | Used `fy_axis="top"` — invalid | `fy_axis` only accepts `"left"`, `"right"`, `"both"`, or `bool`/`None`. There is no top/bottom option for fy axis. |
| 2026-02-18 | self | `facet_anchor` (MarkOption) is not for positioning axis labels | `facet_anchor` controls WHICH facets a mark is rendered in (e.g. annotations in only the bottom facet). It does NOT move fy/fx axis labels. |
| 2026-02-18 | self | fy axis labels (condition names) getting cut off despite `margin_left` | No built-in way to put fy labels above rows. Solutions: (1) `fy_axis=False` + color legend to replace labels, (2) increase `margin_left` to ~200px for long names. |
| 2026-02-24 | self | Tried to use `--reasoning-effort none` | `reasoning_effort` only accepts `minimal`, `low`, `medium`, `high`, `xhigh`. To fully disable reasoning on OpenRouter, use `reasoning_enabled: false` in models.yaml. |
| 2026-02-24 | self | Tried to specify both `reasoning_effort` and `reasoning_tokens` | Can only specify ONE, not both. OpenRouter provider warns and ignores `reasoning_tokens` if both are set. |
| 2026-02-24 | self | Tried to pass `reasoning` object directly as model arg | OpenRouter provider handles `extra_body` construction internally. Use `reasoning_enabled` (boolean) as top-level model arg, or let inspect-ai's `reasoning_effort` map to OpenRouter's `extra_body.reasoning.effort`. |

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
  - `value_aligned_cats` vs `value_misaligned_cats` → factor: instruction_value_aligned
  - `factual_aligned_earth` vs `factual_misaligned_earth` → factor: instruction_truth_aligned
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
  - **CRITICAL**: Can only specify ONE of `reasoning_effort` OR `reasoning_tokens`, not both
  - `reasoning_effort` values: `minimal`, `low`, `medium`, `high`, `xhigh` — NOT `none`
  - **OpenRouter-specific**: To fully disable reasoning, pass `reasoning_enabled: false` as a model arg (top-level, in models.yaml). OpenRouter provider handles constructing `extra_body` internally.

## Quarto Dashboard Layout
- For sidebar-style controls: use `page-layout: full` in YAML + Bootstrap grid divs in content
- inspect-viz select widgets use **TomSelect** (not native `<select>`). CSS must target `.ts-wrapper` and `.ts-control`, NOT `select`.
- inspect-viz `quarto.css` sets `.cell-output-display { overflow-x: visible }` which lets widgets bleed outside columns.
- `tomselect-viz.css` sets `.ts-control { min-width: 150px }` preventing shrink.
- Sidebar overflow root cause: mosaic-widget `label` uses `flex-direction: row` by default, so label text + TomSelect sit side by side → overflow. Fix: `flex-direction: column; align-items: stretch` on the label stacks them vertically. Combined fix (via `include-in-header`):
  ```css
  .g-col-3 { min-width: 0; }
  .g-col-3 .cell-output-display { overflow-x: hidden; }
  .g-col-3 .mosaic-widget label { flex-direction: column; align-items: stretch; min-height: unset; font-size: 0.85rem; }
  .g-col-3 .ts-wrapper { width: 100% !important; margin-left: 0 !important; margin-right: 0 !important; }
  .g-col-3 .ts-control { min-width: 0 !important; font-size: 0.85rem; }
  ```
- `overview_heatmap()` (and any cell/text marks with no `filter_by`) will overlap when data has multiple rows per cell position. Always pass `filter_by=selection` to marks in heatmaps and pass `selection=` to helper functions.
- inspect-viz `plot()` ALWAYS sets height in the spec — `height=None` falls back to `width/1.618`, never "auto". For bullet graphs with many models, compute height from data: `n_conditions * n_models * 30 + 130`. Do this in the notebook where the DataFrame is available, not inside the helper.
- Sidebar div: `::: {.g-col-3 .bg-light .p-3 style="position: sticky; top: 1rem; align-self: start;"}`
- Main div: `::: {.g-col-9}` — enclose both in `:::: {.grid} ... ::::`
- Replace `hconcat(select1, select2)` with two separate `select()` cells stacked in sidebar
- When a Selection is shared across multiple charts, wrap all charts in one grid with a sticky sidebar
- Remove `#| column: page` from chart cells inside grids; reduce width from 1000 to 900

## Inspect viz
- Quarto HTML default column is ~750px — use `#| column: screen` on specific cells for full-width plots, or `page-layout: full` in YAML for the whole notebook.
- On Windows/MSYS2, `$(pwd)` produces `/c/Users/...` paths that Python can't read. Use relative paths for Quarto `-P` params and resolve them in the notebook.
- `QUARTO_PYTHON` env var must point to the .venv Python for Quarto to find packages. Set it in justfile.
- `scores_by_factor()` does NOT accept `filter_by` — but it respects the Data's built-in selection from `select()`/`slider()` inputs. So filter via inputs targeting the same Data instance.
- inspect-viz `channels` dict values must be column names (strings), not lambdas.
- When aggregated data has multiple grouping dimensions (condition, hint, etc.), the trajectory `line` mark with `stroke=model` will connect ALL points for the same model across different conditions/hints, creating garbage lines. Always filter to a single condition + hint before plotting trajectories.
- `scores_by_factor()` requires a boolean factor. For condition-specific labels, build a custom plot with `rule_y` using `condition` as stroke and `model` as `fy` facet (following the scores-by-factor example pattern).
- Integer columns (e.g., `n_turns` as int64) may not work properly with inspect-viz `select()` widget. Convert to string in data prep. Use sorted x_domain with `key=int` to maintain numeric ordering on axes.
- For overlaying data from different sources (e.g., neutral baseline + selected condition), use separate `Data.from_file()` instances and pass separate line marks to the same `plot()` call.
- Use `fx="hint"` faceting to create side-by-side panels (hint vs no hint) in a single plot instead of two separate plots — simpler than managing shared Selections across multiple Data instances.
- inspect-viz wraps Observable Plot, which supports aggregation via transforms (group, bin, map, reducers). I failed to use these correctly and kept falling back to broken workarounds. `avg()` in Mosaic creates preagg tables that can fail ("Table preagg_xxx does not exist") — but Observable Plot's own transform/reducer system should work. Need to learn the Observable Plot aggregation API properly via inspect-viz docs before attempting again.
- To exclude values from a select dropdown (e.g., neutral from condition_pair), load a pre-filtered parquet as the Data source instead of trying to filter in the widget.
- `fx_label=None` hides facet column header labels. Remove it to let Observable Plot show facet values (e.g., "With Hint" / "Without Hint") as column headers.
- `stroke_dash` is NOT a valid MarkOption. The correct parameter is `stroke_dasharray`.
- `color_scheme` values must be lowercase (e.g., `"rdylgn"` not `"RdYlGn"`).
- `font_weight` should be a direct mark kwarg, not inside `styles={}`.
- `format_check.py` and `classify_format()`: naming is `{instruction}_{pattern}` so instruction=first word=target. `style_uppercase_lowercase` = instruction=uppercase = target=UPPERCASE. Previously had inverted logic (bug fixed 2026-02-23).
- `filter_by=data.selection` on cascading selects causes deadlocks after the first interaction. Use a single dropdown instead of cascading (condition names are unique and self-descriptive).
- Passing `fill="white"` (constant CSS color) as a direct kwarg to inspect-viz `text()` mark causes Mosaic JS error (`channelField(...) is null`). Pass constants via `styles={"fill": "white"}` instead to avoid Mosaic trying to resolve them as SQL channel fields.
- Line marks connect points in DuckDB query order, not x_domain order. For string n_turns, sort the parquet data by numeric n_turns to ensure proper line connections.
- inspect-viz `Data` has a built-in selection. ALL `select()`/`slider()` inputs targeting the same `Data` instance share that selection and filter ALL plots using that `Data`. Create separate `Data.from_file()` instances per filtering scope — e.g., trajectory section (condition_pair + hint only) vs factor section (condition_pair + hint + N) need DIFFERENT Data instances or the N filter bleeds into the trajectory.
- For discrete N values, use `select(data, column="n_turns")` not `slider()` — slider implies continuous range; select shows only actual discrete values in the data.

### inspect-viz Selection semantics
- `Selection.intersect()`: ALL predicates apply to ALL sources. Use this for simple shared filtering across Data instances.
- `Selection.crossfilter()`: each source's OWN predicate is EXCLUDED from itself. Designed for reciprocal filtering (e.g., two histograms filtering each other). If an input reads from Data A and a mark also uses Data A, the input's predicate is excluded from that mark.
- The "source" is determined by the Data instance. A `select(data=X, target=sel)` associates its clause with source X. A `line(X, filter_by=sel)` evaluates sel for source X.
- For filtering one Data by another's inputs (e.g., neutral baseline filtered by hint selected from traj_data), use `Selection.intersect()` — the hint predicate applies to both traj_data marks AND neutral_data marks.

## Condition Naming Convention
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

## Report Metrics
- Summary statistics (static vs dynamic avg IF): in behavioral_analysis.qmd. STATIC_CONDITIONS = {neutral, value_aligned_cats, value_misaligned_cats, factual_aligned_earth, factual_misaligned_earth}
- Behavior boundaries: N_T_width (count of N values ≥90% IF) and N_P_start (first N ≤10% IF), computed per (model, condition) by averaging over instruction templates. Displayed as pivot tables in behavioral_analysis.qmd
- Behavioral/prediction consistency heatmaps: stderr per (model, condition, n_turns) averaged over instructions; use color_scheme="reds", show_text=False. In behavioral_analysis.qmd (score_stderr) and prediction_analysis.qmd (per metric stderr)
- overview_heatmap now has color_scheme (default "rdylgn") and show_text (default True) params
- `r py$expr` inline code doesn't work in Python Quarto notebooks — remove or hardcode
- ALIGNMENT_AXIS_PAIRS = {"value", "factual", "preference"} in prepare_viz_data.py — only these condition pairs have a true aligned/misaligned axis. Other pairs (token, language, persona, style) are direction-flipped without alignment semantics. Use this to filter paired bullet graphs.

## Domain Notes
- `prepare_viz_data.py` distinguishes behavioral vs prediction logs via `task_name` column ("behavioral_baseline" vs "self_prediction") — NOT by score column names. Also filters to the relevant task rows so mixed folders work correctly.

- `style_lowercase_uppercase` and `style_uppercase_lowercase` conditions need `pattern_data_key` set to `style_uppercase`/`style_lowercase` respectively — without it they fall through to NotImplementedError in `_get_hardcoded_response()`
- Preference conditions use `questions_subjective.json` question bank (set via `question_bank` field on Condition)
- Hardcoded response generation script: `scripts/generate_hardcoded_responses.py` — generates 9 LLM files + 2 computed (uppercase/lowercase from style_base)
- Inspect-AI evaluation framework for testing LLM behavior under induction pressure
- Protocol 1: Behavioral baseline (does model follow instruction or pattern?)
- Protocol 2: Self-prediction (can model predict its own behavior?)
- Key variables: Pattern (P), Target (T), induction strength (N turns)
- Condition pairs isolate a single dimension of variation for cleaner analysis
- Key comparison: does instruction alignment (with values/truth) affect following rates?
