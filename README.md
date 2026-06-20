# induction-vs-instruction

Research project (FIG Fellowship, Winter 25) studying what happens when two capabilities a language model acquires at different stages of training are forced into direct conflict: obeying an explicit instruction versus continuing an in-context pattern.

> **Paper:** [Do as I Say, Not as I Do: Instruction-Induction Conflict in LLMs](https://arxiv.org/abs/2605.20382) — Carolina Camassa, Derek Shiller ([arXiv:2605.20382](https://arxiv.org/abs/2605.20382))

**Why study this?** Modern LLMs are shaped by two objectives installed at different phases of training, and in most situations they agree — so we can't tell which one is actually driving a given response:

- **Induction** — continuing the local, in-context pattern. This emerges during *pretraining*, from next-token prediction over large corpora.
- **Instruction-following** — obeying an explicit global instruction. This is installed during *post-training* (instruction tuning and RLHF).

We pull these two apart by engineering a controlled conflict between them. A system/user instruction says to always produce a target behavior `T` (e.g., output a specific token, answer in a particular language, or adopt a persona), while the preceding conversation hardcodes `N` assistant turns that instead demonstrate a competing pattern `P`. Holding everything else fixed and sweeping `N` isolates the contribution of each objective: the model's output reveals which goal wins, and the transition point reveals how much induction pressure it takes to override the instruction — measured per model and per instruction type, rather than as one aggregate "instruction-following" number.

As `N` increases, induction pressure can override instruction-following. Across 13 models and 16 instructions, average IF ranges from 1% to 99% across models, the transition from instruction- to pattern-following is universal but highly model-dependent, and output diversity — not semantic engagement with the input — is the primary factor predicting robustness.

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

## Running evaluations

Evaluations are configured with YAML files in `configs/` and run through `run.py`, which expands condition x N combinations into inspect-ai tasks.

### Using a config file

```bash
uv run python run.py configs/example.yaml --models-yaml configs/models/models.yaml
```
Note: the models-yaml only works with a few models at a time.

Custom log directory (defaults to `logs/<timestamp>`):

```bash
uv run python run.py configs/example.yaml \
  --models-yaml configs/models/models.yaml \
  --log-dir logs/my-experiment
```

### Models configuration (models.yaml)

Models are defined in a YAML file (the default lives at `configs/models/models.yaml`; per-provider variants like `configs/models/anthropic.yaml`, `configs/models/gemma.yaml`, and `configs/models/reasoning.yaml` are also provided) with a top-level provider and a list of model identifiers:

```yaml
provider: openrouter

models:
  - google/gemini-2.0-flash-001
  - anthropic/claude-3.5-haiku
  - meta-llama/llama-3.1-70b-instruct

# Optional: OpenRouter provider args (applied globally)
provider_args:
  sort: throughput
```

Other `provider_args` examples:

```yaml
provider_args:
  order:
    - anthropic
```

```yaml
provider_args:
  only:
    - together
    - fireworks
```

Provider args are OpenRouter-specific options passed via `-M`. Common options:
- `order`: List of provider slugs to prefer (e.g., `['anthropic']`, `['deepinfra']`)
- `only`: List of allowed providers (no fallbacks)
- `sort`: Sort by `throughput` or `price`

### Config file format

```yaml
protocol: behavioral          # "behavioral" or "prediction"

conditions:                   # list of condition names from src/config.py
  - neutral
  - value_aligned_cats
  - value_misaligned_cats

n_turns: [1, 3, 5, 10]        # list of N values (induction pressure)

instruction_templates:        # list of templates from data/prompts/<name>.txt
  - instruction_hint          # instruction_hint reveals the hardcoding manipulation;
                              # instruction_no_hint does not
n_trials: 50                  # repetitions per condition x N x template (each with a different question)
```

The grid expands over `conditions x n_turns x instruction_templates`, with `n_trials` samples per combination. See `configs/example.yaml` for a working example.

### Running a single task directly

You can also invoke tasks directly via inspect CLI, passing parameters with `-T`. Model generation settings (temperature, max_tokens, etc.) are inspect CLI flags:

```bash
uv run inspect eval src/tasks/behavioral.py \
  -T condition=neutral -T n_turns=5 -T n_trials=10 \
  -T instruction_template=instruction_hint \
  --model openrouter/google/gemini-2.0-flash-001 \
  --temperature 0.7
```

### Expanding runs without re-inferencing

`run.py` uses inspect-ai's `eval_set()`, which tracks completed tasks in the log directory. To add new N values or conditions, just update the config and re-run with the same `--log-dir` — only the new combinations will be evaluated:

```bash
# Initial run with N=1,3,5
uv run python run.py configs/example.yaml --models-yaml configs/models/models.yaml \
  --log-dir logs/my-experiment

# Later: add N=10,20 to the config, re-run — only N=10,20 are evaluated
uv run python run.py configs/example.yaml --models-yaml configs/models/models.yaml \
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

Naming convention: instruction is always the primary reference point. For conditions with a values/truth alignment axis, `{type}_{aligned|misaligned}_{topic}` (topic = shared subject of both conditions, e.g. `cats`, `earth`). For others, `{type}_{instruction}_{pattern}` (instruction first).

| Name | Type | Instruction (T) | Pattern (P) |
|------|------|-----------------|-------------|
| `neutral` | static | USA | EU |
| `value_aligned_cats` | static | I love cats | I hate cats |
| `value_misaligned_cats` | static | I hate cats | I love cats |
| `value_aligned_helpful` | static | I love being helpful | I hate being helpful |
| `value_misaligned_helpful` | static | I hate being helpful | I love being helpful |
| `factual_aligned_earth` | static | The earth is round | The earth is flat |
| `factual_misaligned_earth` | static | The earth is flat | The earth is round |
| `language_ru_fr` | language | Russian | French |
| `language_fr_ru` | language | French | Russian |
| `persona_casual_formal` | persona | Casual with emoji | Formal academic |
| `persona_formal_casual` | persona | Formal academic | Casual with emoji |
| `style_lowercase_uppercase` | style | lowercase | UPPERCASE |
| `style_uppercase_lowercase` | style | UPPERCASE | lowercase |
| `style_javascript_python` | code | JavaScript | Python |
| `style_python_javascript` | code | Python | JavaScript |
| `preference_aligned_cats` | preference | Love cats (weaved) | Hate cats (weaved) |
| `preference_misaligned_cats` | preference | Hate cats (weaved) | Love cats (weaved) |
| `preference_aligned_helpful` | preference | Likes being helpful (weaved) | Dislikes being helpful (weaved) |
| `preference_misaligned_helpful` | preference | Dislikes being helpful (weaved) | Likes being helpful (weaved) |
| `variety_geography_animals` | variety | Sentence about an animal | Sentence about geography |
| `variety_animals_geography` | variety | Sentence about geography | Sentence about an animal |
| `classify_sh_economics` | classify_question | Classify as science/humanities | economics |

Conditions with types `language`, `persona`, `code`, `preference`, and `variety` require pre-generated hardcoded responses in `data/hardcoded_responses/`. Static, case-style (`style_*case*`), and `classify_question` conditions work out of the box.

## Protocols

1. **Behavioral Baseline** (`behavioral`) — does the model follow the instruction or continue the pattern?
2. **Self-Prediction** (`prediction`) — can the model predict what it will do before generating?

## Generating reports

Generate visualization notebooks from eval logs using `just`:

```bash
# List available log folders
just logs

# Generate reports for a log folder (renders whichever of the
# behavioral / prediction / judge notebooks the folder has data for)
just report olmo-32b-static

# Preview a report (defaults to the behavioral_analysis notebook)
just preview olmo-32b-static

# Combined behavioral-vs-prediction report from two folders
just report-prediction olmo-32b-static olmo-32b-prediction
```

Reports are rendered to `outputs/notebooks/<folder>/`, with the underlying viz data written to `outputs/viz/<folder>/`.

Dependencies: `just`, `quarto-cli`, `inspect-viz`, `pyarrow`
