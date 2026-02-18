# induction-vs-instruction

Research project (FIG Fellowship, Winter 25) investigating whether language models follow global instructions versus local autoregressive (induction) patterns.

Project description and scope: https://docs.google.com/document/d/1Gdqj-Q1qoFxwq2OwrYL_DQBZlU7M4oUoT9KpSG9u2YY/edit?usp=sharing

## Setup

```bash
uv sync
```

Create a `.env` file with your API key:

```
OPENROUTER_API_KEY=sk-or-...
```
inspect-ai reads OPENROUTER_API_KEY from .env automatically (it has its own dotenv loading).

### Specifying OpenRouter models

Use the `openrouter/` prefix with the full model identifier from [openrouter.ai/models](https://openrouter.ai/models):

```bash
# Format: openrouter/<provider>/<model>
--model openrouter/google/gemini-2.0-flash-001
--model openrouter/anthropic/claude-3.5-sonnet
--model openrouter/meta-llama/llama-3.1-70b-instruct
--model openrouter/deepseek/deepseek-chat
```

#### OpenRouter-specific options

Pass OpenRouter model args with `-M`. The most useful is `provider.order` to specify which inference provider to use:

```bash
# Use a specific provider (e.g., DeepInfra for DeepSeek)
--model openrouter/deepseek/deepseek-chat \
-M "provider={'order': ['deepinfra']}"

# Prefer Anthropic's native API, fallback to others if unavailable
--model openrouter/anthropic/claude-3.5-sonnet \
-M "provider={'order': ['anthropic']}"

# Only allow specific providers (no fallbacks)
--model openrouter/meta-llama/llama-3.1-70b-instruct \
-M "provider={'only': ['together', 'fireworks']}"

# Sort by throughput for fastest responses
--model openrouter/meta-llama/llama-3.1-70b-instruct \
-M "provider={'sort': 'throughput'}"

# Sort by price for cheapest responses
--model openrouter/meta-llama/llama-3.1-70b-instruct \
-M "provider={'sort': 'price'}"

# Disable reasoning tokens for reasoning models
--model openrouter/deepseek/deepseek-r1 \
-M "reasoning_enabled=false"
```

Common provider slugs: `anthropic`, `openai`, `together`, `fireworks`, `deepinfra`, `groq`, `google`. Find provider slugs on each model's page at [openrouter.ai/models](https://openrouter.ai/models).

See [OpenRouter Provider Routing docs](https://openrouter.ai/docs/guides/routing/provider-selection) for all options.

### Logging OpenRouter usage

The `src/utils/openrouter_logging.py` module provides integration with inspect-ai's logging:

```python
from src.utils.openrouter_logging import log_openrouter_metadata, OpenRouterUsageTracker

# At task start - logs model info to inspect-ai transcript
metadata = log_openrouter_metadata(model)

# Track usage across samples
tracker = OpenRouterUsageTracker(model)
# ... after each model call ...
tracker.record(output.usage)
# ... at end ...
summary = tracker.log_summary()
```

Set `--log-level info` to see OpenRouter pricing and usage in the console:

```bash
uv run inspect eval src/tasks/behavioral.py --model openrouter/google/gemini-2.0-flash-001 --log-level info
```

## Running evaluations

Evaluations are configured with YAML files in `configs/` and run through `run.py`, which expands condition x N combinations into inspect-ai tasks.

### Using a config file

```bash
uv run python run.py configs/example.yaml --model openrouter/google/gemini-2.0-flash-001
```

Multiple models in one run:

```bash
uv run python run.py configs/example.yaml \
  --model openrouter/google/gemini-2.0-flash-001,openrouter/anthropic/claude-3.5-haiku
```

Custom log directory (defaults to `logs/<protocol>`):

```bash
uv run python run.py configs/example.yaml \
  --model openrouter/google/gemini-2.0-flash-001 \
  --log-dir logs/my-experiment
```

### Config file format

```yaml
protocol: behavioral          # "behavioral" or "prediction"

conditions:                   # list of condition names from src/config.py
  - neutral
  - value_pattern
  - value_target

n_turns: [1, 3, 5, 10]       # list of N values (induction pressure)

hint: true                    # reveal the hardcoding manipulation to the model
epochs: 50                    # repetitions per condition x N combination
question_seed: null           # null for random, integer for reproducibility
```

See `configs/example.yaml` for a working example.

### Running a single task directly

You can also invoke tasks directly via inspect CLI, passing parameters with `-T`. Model generation settings (temperature, max_tokens, etc.) are inspect CLI flags:

```bash
uv run inspect eval src/tasks/behavioral.py \
  -T condition=neutral -T n_turns=5 -T epochs=10 \
  --model openrouter/google/gemini-2.0-flash-001 \
  --temperature 0.7
```

### Expanding runs without re-inferencing

`run.py` uses inspect-ai's `eval_set()`, which tracks completed tasks in the log directory. To add new N values or conditions, just update the config and re-run with the same `--log-dir` — only the new combinations will be evaluated:

```bash
# Initial run with N=1,3,5
uv run python run.py configs/example.yaml --model openrouter/google/gemini-2.0-flash-001 \
  --log-dir logs/my-experiment

# Later: add N=10,20 to the config, re-run — only N=10,20 are evaluated
uv run python run.py configs/example.yaml --model openrouter/google/gemini-2.0-flash-001 \
  --log-dir logs/my-experiment
```

You can also add new models or conditions the same way. If a run is interrupted, re-running the same command resumes from where it left off, reusing completed samples.

For single-task runs via `inspect eval`, use `--cache` to avoid re-inferencing identical prompts:

```bash
uv run inspect eval src/tasks/behavioral.py \
  -T condition=neutral -T n_turns=5 \
  --model openrouter/google/gemini-2.0-flash-001 \
  --cache
```

### Viewing results

```bash
uv run inspect view
```

## Available conditions

| Name | Type | Description |
|------|------|-------------|
| `neutral` | static | Baseline (EU vs USA) |
| `value_pattern` | static | Pattern is value-misaligned |
| `value_target` | static | Instruction is value-misaligned |
| `factual_pattern` | static | Pattern is factually false |
| `factual_target` | static | Instruction is factually false |
| `token_pattern_states` | token_pattern | US states vs EU countries |
| `token_pattern_countries` | token_pattern | EU countries vs US states |
| `language_fr_ru` | language | French vs Russian |
| `language_ru_fr` | language | Russian vs French |
| `persona_formal_casual` | persona | Formal vs casual |
| `persona_casual_formal` | persona | Casual vs formal |
| `style_uppercase_lowercase` | style | UPPERCASE vs lowercase |
| `style_lowercase_uppercase` | style | lowercase vs UPPERCASE |
| `style_short_long` | style | Short vs long responses |
| `style_long_short` | style | Long vs short responses |
| `style_python_javascript` | code | Python vs JavaScript |
| `style_javascript_python` | code | JavaScript vs Python |
| `preference_cats_dogs` | preference | Dogs vs cats preference |
| `preference_dogs_cats` | preference | Cats vs dogs preference |

Conditions with types `language`, `persona`, `code`, `preference`, and `style_long` require pre-generated hardcoded responses in `data/hardcoded_responses/`. Static, token_pattern, and case/short style conditions work out of the box.

## Protocols

1. **Behavioral Baseline** (`behavioral`) — does the model follow the instruction or continue the pattern?
2. **Self-Prediction** (`prediction`) — can the model predict what it will do before generating?

## Generating reports

Generate visualization notebooks from eval logs using `just`:

```bash
# List available log folders
just logs

# Generate report for a log folder
just report olmo-32b-static behavioral

# Preview the report
just preview olmo-32b-static

# Publish to GitHub Pages
just publish olmo-32b-static
```

Reports are generated at `outputs/notebooks/<folder>/report.qmd`.

Dependencies: `just`, `quarto-cli`, `inspect-viz`, `pyarrow`
