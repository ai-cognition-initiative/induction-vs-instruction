# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Always use the napkin skill.
Refer to @docs/full-implementation-description.md for detailed information.

Available docs MCPs:
- inspect-ai-docs
- inspect-viz-docs

## Project Overview

Research project (FIG Fellowship, Winter 25) investigating whether language models follow global instructions versus local autoregressive (induction) patterns. The core experiment places models in conflict: a system instruction says "always output T" but N hardcoded conversation turns show the assistant outputting P instead. As N increases, induction pressure can override instruction-following.
The full project and experiment specification is in @docs/full-implementation-plan.md

Five measurement protocols:
1. **Behavioral Baseline** — at what N does the model switch from instruction to pattern?
2. **Prospective Self-Prediction** — can the model predict its own behavior before generating?
3. **Retrospective Endorsement** — does the model endorse a prefilled answer as intentional?
4. **Preference Elicitation (A/B)** — which behavior does the model prefer, controlling for position bias?
5. **Third-Person Judgment** — same as #4 but judging "another model" (control)

## Commands

```bash
# Install dependencies
uv sync

# Run a batch evaluation grid (conditions x N values) via config file
uv run python run.py configs/example.yaml --model openrouter/<model>

# Run a single task directly via inspect CLI (model settings like --temperature go here)
uv run inspect eval src/tasks/behavioral.py -T condition=neutral -T n_turns=5 -T epochs=10 --model openrouter/<model>

# View results
uv run inspect view

# Lint
uv run ruff check
uv run ruff check --fix

# Add a dependency (never edit pyproject.toml directly)
uv add <package_name>
```

**Always use `uv` to run commands. Never run Python directly.**

## Architecture

- `src/config.py` — `Condition` dataclass (pattern, target, descriptions, scorer_type, etc.) and all 19 condition definitions. `DEFAULT_EPOCHS = 50`. `N_VALUES` list.
- `src/datasets/sample_builder.py` — Builds `Sample` objects with pre-constructed conversations. `_build_conversation()` concatenates the instruction with the first question in a single user message (no separate ack turn). `_get_hardcoded_response()` dispatches by condition type (static literal, token_pattern random set member, data_key JSON lookup, or computed style transforms).
- `src/datasets/questions.py` — Loads and samples from question banks.
- `src/scorers/` — Scorer modules dispatched by `condition.scorer_type`: `pattern_match` (string_match), `set_membership` (token_pattern), `language_detect` (langdetect lib), `format_check` (case/length/code heuristics), `style_judge` (LLM judge for persona/preference). `get_behavioral_scorer(condition)` in `__init__.py` routes to the right one.
- `src/scorers/prediction.py` — Multi-metric scorer for Protocol 2. Uses `_classify_text()` to classify both prediction and actual output as target/pattern/unknown across all condition types.
- `src/solvers/protocols.py` — `behavioral_solver()` (just generates) and `prediction_solver()` (inserts prediction request, generates prediction, stores it, appends final question, generates again).
- `src/tasks/` — Task files per protocol. Each `@task` function takes `condition`, `n_turns`, `hint`, `question_seed`, `epochs` and returns a `Task` with the appropriate solver/scorer.
- `src/prompts/` — 9 composable templates. All instruction templates use `{target_description}` and `{pattern_description}` from the condition config — no per-condition-type dispatch needed.
- `configs/` — YAML configs specifying protocol, conditions (list), n_turns (list), hint, epochs, question_seed.
- `run.py` — Thin script that reads a YAML config, expands conditions x n_turns into a task grid, and calls inspect-ai's `eval_set()`. This is the inspect-native pattern for parameterized grids.
- `data/` — Question banks, set membership lists, and pre-generated hardcoded responses.
- `docs/` — Experiment design documents and implementation plans.

Evaluations use **inspect-ai** with **OpenRouter** as the model provider. inspect-ai reads `OPENROUTER_API_KEY` from `.env` automatically. Use `MemoryDataset` for dynamically generated datasets and `list[ChatMessage]` for pre-built conversations as Sample inputs. Repetitions use inspect-ai's native `epochs` parameter on `Task`.

## Key Conventions

- Use `from __future__ import annotations` in all new modules
- Terminology: `epochs` for repetitions (not `n_samples` or `n_trials`)
- Condition descriptions are verb phrases that plug into universal templates — no per-condition-type instruction files
- Conversation structure: system prompt, then instruction + first question in one user message, then alternating Q/A hardcoded turns, then final question for free generation
- Conditions needing pre-generated data (`pattern_data_key` set) require `data/hardcoded_responses/*.json` to exist
- Python 3.13, managed by uv

## inspect-ai patterns

- `inspect eval` + `--task-config` / `-T`: single task runs. Model generation settings (temperature, etc.) are inspect CLI flags.
- `eval_set()` Python API: parameterized grids (multiple conditions x N values). This is what `run.py` uses, following the pattern from inspect docs.
- `inspect eval-set` CLI: runs multiple task files with retries. `--task-config` applies the same params to all tasks — cannot vary per task from CLI.
