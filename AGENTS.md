# AGENTS.md

This file contains guidelines for agentic coding assistants working in this repository.

## Project Overview

This is a research project investigating internal mechanisms that influence whether language models follow global instructions versus local autoregressive patterns. 

## Package Management

**ALWAYS use `uv` to run commands. Never run Python directly.**

```bash
# Install dependencies
uv sync

# Run any Python script
uv run python script.py

# Run linting
uv run ruff check
uv run ruff check --fix
```

## Code Style

### Imports
- Use `from __future__ import annotations` for type hints in new modules
- Group imports: standard library, third-party, local
- Sort imports alphabetically within each group

### Formatting & Linting
This project uses **ruff** for linting. Run before committing:
```bash
uv run ruff check
uv run ruff check --fix  # Auto-fix issues
```

### Naming Conventions
- Functions: `snake_case` (e.g., `compute_axis`, `load_activations`)
- Classes: `PascalCase` (e.g., `ConversationEncoder`, `ModelConfig`)
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Error Handling
- Use specific exceptions (`ValueError`, `FileNotFoundError`, etc.)
- Raise descriptive error messages with context
- Check file existence before loading

## Activation Loading Methodology

## Nnsight and Activations

### Saved Proxies Are Already Tensors
After trace completes, saved proxies ARE tensors - don't call `.value()`:
```python
with model.trace(input_ids):
    saved = activation.save()

result = torch.stack([saved], dim=0)
```

### Converting to NumPy
Use `.detach().cpu().float().numpy()` to convert:
```python
array = torch.stack(results).detach().cpu().float().numpy()
```

### Gemma 3 Precision
**Always use bfloat16 (not float16) for Gemma 3 activations:**
```python
tensor.bfloat16().cpu()  # CORRECT - range ~3.4e38
tensor.half().cpu()      # WRONG - range ~65504, will overflow
```

### Memory Management
```python
with torch.no_grad():
    with model.trace(input_ids):
        # tracing code

if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Logit Lens Analysis

### Gemma Models Require Final RMSNorm
```python
normalized = torch.nn.functional.rms_norm(
    residual, (hidden_size,), final_norm_weight, eps=1e-6
)
logits = normalized @ W_U.T
```

### Use Logit Differences (Not Raw Probabilities)
```python
logit_diff = logit(token_A) - logit(token_B)
best_logit = logits[variants].max(dim=-1)
total_prob = probs[variants].sum(dim=-1)
```

## Data Loading

Use `load_dotenv()` for environment variables:
```python
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
```

## Dependencies
Never edit `pyproject.toml` directly. Use `uv add`:
```bash
uv add package_name
uv sync
```

## inspect-ai
### How to use Openrouter with inspect-ai:
To use the OpenRouter provider, install the openai package (which the OpenRouter service provides a compatible backend for), set your credentials, and specify a model using the --model option:

export OPENROUTER_API_KEY=your-openrouter-api-key
inspect eval arc.py --model openrouter/gryphe/mythomax-l2-13b

For the openrouter provider, the following custom model args (-M) are supported (click the argument name to see its docs on the OpenRouter site):
Argument 	Example
models 	-M "models=anthropic/claude-3.5-sonnet, gryphe/mythomax-l2-13b"
provider 	-M "provider={ 'quantizations': ['int8'] }" -> look up how to specify a specific provider in the Openrouter docs: https://openrouter.ai/docs/guides/routing/provider-selection
transforms 	-M "transforms=['middle-out']"
reasoning_enabled 	-M "reasoning_enabled=false"
