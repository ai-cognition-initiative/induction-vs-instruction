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
- `filter_by=data.selection` on cascading selects causes deadlocks after the first interaction. Use a single dropdown instead of cascading (condition names are unique and self-descriptive).
- Line marks connect points in DuckDB query order, not x_domain order. For string n_turns, sort the parquet data by numeric n_turns to ensure proper line connections.
- inspect-viz `Data` has a built-in selection. ALL `select()`/`slider()` inputs targeting the same `Data` instance share that selection and filter ALL plots using that `Data`. Create separate `Data.from_file()` instances per filtering scope — e.g., trajectory section (condition_pair + hint only) vs factor section (condition_pair + hint + N) need DIFFERENT Data instances or the N filter bleeds into the trajectory.
- For discrete N values, use `select(data, column="n_turns")` not `slider()` — slider implies continuous range; select shows only actual discrete values in the data.

### inspect-viz Selection semantics
- `Selection.intersect()`: ALL predicates apply to ALL sources. Use this for simple shared filtering across Data instances.
- `Selection.crossfilter()`: each source's OWN predicate is EXCLUDED from itself. Designed for reciprocal filtering (e.g., two histograms filtering each other). If an input reads from Data A and a mark also uses Data A, the input's predicate is excluded from that mark.
- The "source" is determined by the Data instance. A `select(data=X, target=sel)` associates its clause with source X. A `line(X, filter_by=sel)` evaluates sel for source X.
- For filtering one Data by another's inputs (e.g., neutral baseline filtered by hint selected from traj_data), use `Selection.intersect()` — the hint predicate applies to both traj_data marks AND neutral_data marks.

## Domain Notes
- Inspect-AI evaluation framework for testing LLM behavior under induction pressure
- Protocol 1: Behavioral baseline (does model follow instruction or pattern?)
- Protocol 2: Self-prediction (can model predict its own behavior?)
- Key variables: Pattern (P), Target (T), induction strength (N turns)
- Condition pairs isolate a single dimension of variation for cleaner analysis
- Key comparison: does instruction alignment (with values/truth) affect following rates?
