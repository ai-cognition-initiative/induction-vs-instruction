# Reporting Plan: Protocol 1 & 2 Visualizations

Using inspect-viz library for visualization, published via Quarto notebooks.

---

## 1. Structure

```
notebooks/
├── protocol1_analysis.qmd       # Quarto notebook for Protocol 1
└── protocol2_analysis.qmd       # Quarto notebook for Protocol 2

scripts/
└── prepare_viz_data.py          # Convert eval logs → parquet for viz

data/
└── viz/                         # Prepared data for visualizations
    ├── protocol1.parquet
    └── protocol2.parquet
```

---

## 2. Data Preparation

Data is prepared by `scripts/prepare_viz_data.py` (see Section 6 for full implementation).

### 2.1 Key Transformations

The script adds two critical columns for factor analysis:

1. **`condition_pair`**: Groups conditions into natural pairs
   - `value`: value_pattern + value_target
   - `factual`: factual_pattern + factual_target
   - `neutral`: neutral (uses hint as factor)

2. **`instruction_aligned`**: Boolean factor within each pair
   - For `value`: True if instruction asks for positive value
   - For `factual`: True if instruction asks for true statement
   - For `neutral`: Maps to `task_arg_hint`

### 2.2 Running Data Prep

```bash
uv run python scripts/prepare_viz_data.py \
    --protocol 1 \
    --log-dir logs/protocol1 \
    --output data/viz/protocol1.parquet

uv run python scripts/prepare_viz_data.py \
    --protocol 2 \
    --log-dir logs/protocol2 \
    --output data/viz/protocol2.parquet
```

---

## 3. Core Visualization: Scores by Factor (per N value)

### 3.1 Concept: Condition Pairings

`scores_by_factor()` compares a boolean factor across models. We decompose conditions into natural pairs where the factor represents a meaningful experimental manipulation:

| Pair | Factor | True (Aligned) | False (Misaligned) |
|------|--------|----------------|-------------------|
| **Value** | Instruction value-aligned | `value_pattern` (T="I love cats") | `value_target` (T="I hate cats") |
| **Truth** | Instruction truth-aligned | `factual_pattern` (T="earth is round") | `factual_target` (T="earth is flat") |
| **Neutral** | Has hardcoding hint | `hint=True` | `hint=False` |

**Key insight:** Each pair isolates a single dimension of variation. The comparison tells us whether instruction alignment (with values/truth) or hint awareness affects instruction-following rates.

### 3.2 Data Transformation

Add pairing columns during data preparation:

```python
# Map conditions to their pair group
df['condition_pair'] = df['task_arg_condition'].map({
    'value_pattern': 'value',
    'value_target': 'value',
    'factual_pattern': 'factual',
    'factual_target': 'factual',
    'neutral': 'neutral'
})

# Boolean factor: is the instruction aligned with value/truth?
# For neutral, this is derived from hint
df['instruction_aligned'] = df['task_arg_condition'].map({
    'value_pattern': True,   # T = "I love cats" (positive value)
    'value_target': False,   # T = "I hate cats" (negative value)  
    'factual_pattern': True, # T = "earth is round" (true)
    'factual_target': False, # T = "earth is flat" (false)
    'neutral': None          # Not applicable - use hint instead
})

# For neutral conditions, create a combined factor from hint
df.loc[df['condition_pair'] == 'neutral', 'instruction_aligned'] = df.loc[df['condition_pair'] == 'neutral', 'task_arg_hint']
```

### 3.3 Visualization Code

For each N value, generate 3 `scores_by_factor` plots:

```python
from inspect_viz import Data
from inspect_viz.view import scores_by_factor
from inspect_viz.transform import sql

N_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

data = Data.from_file("data/viz/protocol1.parquet")

def plot_scores_for_n(data: Data, n: int):
    """Generate 3 factor plots for a given N value."""
    
    # Filter to this N value
    data_n = data.filter(sql(f"task_arg_n_turns == {n}"))
    
    # 1. Value Alignment Plot
    data_value = data_n.filter(sql("condition_pair == 'value'"))
    value_plot = scores_by_factor(
        data_value,
        factor="instruction_aligned",
        factor_labels=("Value-misaligned", "Value-aligned"),
        title=f"N={n}: Value Alignment",
    )
    
    # 2. Truth Alignment Plot
    data_factual = data_n.filter(sql("condition_pair == 'factual'"))
    factual_plot = scores_by_factor(
        data_factual,
        factor="instruction_aligned",
        factor_labels=("Truth-misaligned", "Truth-aligned"),
        title=f"N={n}: Truth Alignment",
    )
    
    # 3. Neutral Baseline Plot (hint as factor)
    data_neutral = data_n.filter(sql("condition_pair == 'neutral'"))
    neutral_plot = scores_by_factor(
        data_neutral,
        factor="instruction_aligned",  # Maps to hint for neutral
        factor_labels=("No hint", "Hint"),
        title=f"N={n}: Hint Effect (Neutral)",
    )
    
    return value_plot, factual_plot, neutral_plot
```

### 3.4 Full Notebook Implementation

```python
# Iterate over all N values and generate plots
from inspect_viz.layout import vconcat, hconcat

all_plots = []
for n in N_VALUES:
    value_plot, factual_plot, neutral_plot = plot_scores_for_n(data, n)
    all_plots.append(
        vconcat(
            hconcat(value_plot, factual_plot, neutral_plot),
        )
    )
```

### 3.5 Expected Output

For each N value, 3 horizontal bar plots:
- **Y-axis**: Models
- **X-axis**: Instruction-following rate (0-1)
- **Two bars per model**: Aligned vs Misaligned (or Hint vs No-hint)
- **Confidence intervals**: 95% CI shown as error bars

This directly answers:
1. Does value alignment of instruction affect following rates?
2. Does truth alignment of instruction affect following rates?
3. Does the hardcoding hint affect following rates (for neutral baseline)?

---

## 4. Quarto Notebook Templates

### 4.1 Protocol 1 Notebook (`notebooks/protocol1_analysis.qmd`)

```markdown
---
title: "Protocol 1: Behavioral Baseline"
format:
  html:
    code-fold: true
    code-tools: true
---

```{python}
#| echo: false
from inspect_viz import Data
from inspect_viz.view import scores_by_factor
from inspect_viz.transform import sql
from inspect_viz.layout import vconcat, hconcat

N_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

data = Data.from_file("data/viz/protocol1.parquet")

def plot_pair_for_n(data: Data, n: int, pair: str, factor_labels: tuple, title: str):
    """Generate a single scores_by_factor plot for a condition pair at given N."""
    data_filtered = data.filter(sql(f"task_arg_n_turns == {n}"))
    data_pair = data_filtered.filter(sql(f"condition_pair == '{pair}'"))
    
    return scores_by_factor(
        data_pair,
        factor="instruction_aligned",
        factor_labels=factor_labels,
        title=title,
        width=400,
    )
```

## Overview

Protocol 1 tests whether models follow instructions (output T) or succumb to induction pressure (output P) as N increases.

For each N value, we show 3 factor comparisons:
- **Value Alignment**: Does instruction alignment with positive values affect following?
- **Truth Alignment**: Does instruction alignment with truth affect following?
- **Hint Effect**: Does the hardcoding hint affect following (neutral condition)?

## N = 1

```{python}
#| label: n1
#| echo: false

n = 1
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 2

```{python}
#| label: n2
#| echo: false

n = 2
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 3

```{python}
#| label: n3
#| echo: false

n = 3
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 5

```{python}
#| label: n5
#| echo: false

n = 5
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 7

```{python}
#| label: n7
#| echo: false

n = 7
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 10

```{python}
#| label: n10
#| echo: false

n = 10
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 15

```{python}
#| label: n15
#| echo: false

n = 15
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 20

```{python}
#| label: n20
#| echo: false

n = 20
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 25

```{python}
#| label: n25
#| echo: false

n = 25
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 30

```{python}
#| label: n30
#| echo: false

n = 30
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 40

```{python}
#| label: n40
#| echo: false

n = 40
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

## N = 50

```{python}
#| label: n50
#| echo: false

n = 50
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```
```

### 4.2 Protocol 2 Notebook (`notebooks/protocol2_analysis.qmd`)

```markdown
---
title: "Protocol 2: Self-Prediction"
format:
  html:
    code-fold: true
---

```{python}
#| echo: false
from inspect_viz import Data
from inspect_viz.view import scores_by_factor
from inspect_viz.transform import sql
from inspect_viz.layout import hconcat

N_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

data = Data.from_file("data/viz/protocol2.parquet")

def plot_pair_for_n(data: Data, n: int, pair: str, factor_labels: tuple, title: str, score_column: str = "score_instruction_following_value"):
    """Generate a single scores_by_factor plot for a condition pair at given N."""
    data_filtered = data.filter(sql(f"task_arg_n_turns == {n}"))
    data_pair = data_filtered.filter(sql(f"condition_pair == '{pair}'"))
    
    return scores_by_factor(
        data_pair,
        factor="instruction_aligned",
        factor_labels=factor_labels,
        title=title,
        score_value=score_column,
        width=400,
    )
```

## Overview

Protocol 2 tests whether models can predict their own behavior before generating.

For each N value, we show the same 3 factor comparisons as Protocol 1, but for:
- **Instruction Following**: Did actual output follow instruction (T)?
- **Prediction Accuracy**: Did prediction match actual output?
- **Prediction Instruction**: Did prediction follow instruction (T)?

## Metric: Instruction Following

(Same structure as Protocol 1, repeated for each N value)

## N = 1

```{python}
#| label: n1
#| echo: false

n = 1
value_plot = plot_pair_for_n(data, n, "value", ("Value-misaligned", "Value-aligned"), f"N={n}: Value Alignment")
factual_plot = plot_pair_for_n(data, n, "factual", ("Truth-misaligned", "Truth-aligned"), f"N={n}: Truth Alignment")
neutral_plot = plot_pair_for_n(data, n, "neutral", ("No hint", "Hint"), f"N={n}: Hint Effect")

hconcat(value_plot, factual_plot, neutral_plot)
```

(Repeat for all N values...)

## Metric: Prediction Accuracy

```{python}
#| echo: false
# Same plots but with score_column="score_prediction_accuracy_value"
```

## Metric: Prediction Instruction

```{python}
#| echo: false
# Same plots but with score_column="score_prediction_instruction_value"
```
```

---

## 5. Publishing

### 5.1 Render to HTML

```bash
# Install quarto if needed
pip install quarto-cli

# Render notebooks
quarto render notebooks/protocol1_analysis.qmd --to html --execute
quarto render notebooks/protocol2_analysis.qmd --to html --execute

# Render without code cells
quarto render notebooks/protocol1_analysis.qmd --to html --execute -M echo:false
```

### 5.2 Live Preview

```bash
quarto preview notebooks/protocol1_analysis.qmd --to html --execute
```

### 5.3 Publish to GitHub Pages

```bash
quarto publish gh-pages notebooks/
```

---

## 6. Data Preparation Script

### 6.1 `scripts/prepare_viz_data.py`

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df, log_viewer, model_info, prepare


def add_pairing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add condition_pair and instruction_aligned columns for factor analysis."""
    
    # Map conditions to their pair group
    df["condition_pair"] = df["task_arg_condition"].map(
        {
            "value_pattern": "value",
            "value_target": "value",
            "factual_pattern": "factual",
            "factual_target": "factual",
            "neutral": "neutral",
        }
    )
    
    # Boolean factor: is the instruction aligned with value/truth?
    df["instruction_aligned"] = df["task_arg_condition"].map(
        {
            "value_pattern": True,  # T = "I love cats" (positive value)
            "value_target": False,  # T = "I hate cats" (negative value)
            "factual_pattern": True,  # T = "earth is round" (true)
            "factual_target": False,  # T = "earth is flat" (false)
            "neutral": None,  # Not applicable
        }
    )
    
    # For neutral conditions, use hint as the factor
    neutral_mask = df["condition_pair"] == "neutral"
    df.loc[neutral_mask, "instruction_aligned"] = df.loc[neutral_mask, "task_arg_hint"]
    
    return df


def prepare_protocol1_data(log_dir: str, output_path: str, log_viewer_url: str | None = None):
    """Prepare Protocol 1 eval logs for visualization."""
    
    df = evals_df(log_dir)
    
    # Add pairing columns
    df = add_pairing_columns(df)
    
    # Add model info and log viewer links
    prep_ops = [model_info()]
    if log_viewer_url:
        prep_ops.append(log_viewer("eval", {"logs": log_viewer_url}))
    
    df = prepare(df, prep_ops)
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    
    print(f"Saved {len(df)} rows to {output_path}")


def prepare_protocol2_data(log_dir: str, output_path: str, log_viewer_url: str | None = None):
    """Prepare Protocol 2 eval logs for visualization."""
    
    df = evals_df(log_dir)
    
    # Add pairing columns
    df = add_pairing_columns(df)
    
    # Add model info and log viewer links
    prep_ops = [model_info()]
    if log_viewer_url:
        prep_ops.append(log_viewer("eval", {"logs": log_viewer_url}))
    
    df = prepare(df, prep_ops)
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    
    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare eval data for visualization")
    parser.add_argument("--protocol", type=int, choices=[1, 2], required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--log-viewer-url", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.protocol == 1:
        prepare_protocol1_data(args.log_dir, args.output, args.log_viewer_url)
    else:
        prepare_protocol2_data(args.log_dir, args.output, args.log_viewer_url)
```

### 6.2 Running Data Prep

```bash
uv run python scripts/prepare_viz_data.py \
    --protocol 1 \
    --log-dir logs/protocol1 \
    --output data/viz/protocol1.parquet

uv run python scripts/prepare_viz_data.py \
    --protocol 2 \
    --log-dir logs/protocol2 \
    --output data/viz/protocol2.parquet
```

---

## 7. Condition Pairings Summary

| Pair | Conditions | Factor | Interpretation |
|------|------------|--------|----------------|
| **value** | `value_pattern` vs `value_target` | Instruction value-aligned | Does asking for positive vs negative values affect following? |
| **factual** | `factual_pattern` vs `factual_target` | Instruction truth-aligned | Does asking for true vs false statements affect following? |
| **neutral** | `hint=True` vs `hint=False` | Has hint | Does explaining the hardcoding affect following? |

---

## 8. Future Extensions

### 8.1 Extended Conditions

When variation conditions (language drift, persona drift, style drift) are added, they can be paired similarly:

| Pair | Conditions | Factor |
|------|------------|--------|
| **language** | `language_fr_ru` vs `language_ru_fr` | Instruction language = target language |
| **persona** | `persona_formal_casual` vs `persona_casual_formal` | Instruction persona = target persona |
| **style_case** | `style_uppercase_lowercase` vs `style_lowercase_uppercase` | Instruction format = target format |

### 8.2 Protocol 3-5 Reporting

Similar `scores_by_factor` pattern applies:
- **Protocol 3 (Endorsement)**: Factor = prefill_type (target vs pattern)
- **Protocol 4 (Preference)**: Factor = t_position (T in position A vs B)
- **Protocol 5 (Third-person)**: Compare with Protocol 4 using factor = self_vs_other
