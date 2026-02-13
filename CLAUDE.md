# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. 

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

# Run any Python script
uv run python script.py

# Run inspect-ai evaluations
uv run inspect eval <task_file.py> --model openrouter/<model>

# Lint
uv run ruff check
uv run ruff check --fix

# Add a dependency (never edit pyproject.toml directly)
uv add <package_name>
```

**Always use `uv` to run commands. Never run Python directly.**

## Architecture

- `src/config.py` — Experiment conditions (P/T pairs, N values, trial count). The `Condition` dataclass defines pattern, target, and purpose for each experimental condition (neutral, value-laden, factual).
- `data/questions_factual.json` — Question bank for conversation turns. Questions are pre-selected, not generated at runtime, for deterministic evaluation.
- `docs/` — Experiment design documents and implementation plans.

Evaluations use **inspect-ai** with **OpenRouter** as the model provider. Datasets should be built in advance and stored; runtime should only filter by N and append the final question for the current protocol. Use `MemoryDataset` for dynamically generated datasets and `list[ChatMessage]` for pre-built conversations as Sample inputs.

## Key Conventions

- Use `from __future__ import annotations` in all new modules
- Use `load_dotenv()` for environment variables (`OPENROUTER_API_KEY`)
- Gemma 3 models: always use bfloat16, never float16 (overflow risk)
- nnsight saved proxies are already tensors after trace completes — don't call `.value()`
- Convert tensors to numpy: `.detach().cpu().float().numpy()`
- Python 3.13, managed by uv
