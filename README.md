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

## Running evaluations

Evaluations are configured with YAML files in `configs/` and run through `run.py`, which expands condition x N combinations into inspect-ai tasks.

### Using a config file

```bash
uv run python run.py configs/example.yaml --model openrouter/google/gemini-2.0-flash-001
```

Model generation settings are passed as extra CLI flags:

```bash
uv run python run.py configs/example.yaml \
  --model openrouter/google/gemini-2.0-flash-001 \
  --temperature 0.7 \
  --max-tokens 256
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

You can also invoke tasks directly via inspect CLI, passing parameters with `-T`:

```bash
uv run inspect eval src/tasks/behavioral.py -T condition=neutral -T n_turns=5 -T epochs=10 \
  --model openrouter/google/gemini-2.0-flash-001
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
