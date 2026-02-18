# Reporting Plan: Protocol 1 & 2 Visualizations

Using inspect-viz library for visualization, published via Quarto notebooks.

---

## 1. Structure

```
notebooks/
├── behavioral_analysis.qmd       # Quarto notebook for Protocol 1
└── prediction_analysis.qmd       # Quarto notebook for Protocol 2

scripts/
└── prepare_viz_data.py          # Convert eval logs → parquet for viz

data/
└── viz/                         # Prepared data for visualizations
    ├── .parquet
```
---

## 2. Data Preparation

Data is prepared by `scripts/prepare_viz_data.py`.

```python

When reading dataframes in inspect-ai, there are a number of pre-built column groups you can use to read various subsets of columns. For example:
To only read the relevant one for plotting, do:
from inspect_ai.analysis import (
     EvalModel, EvalTask, EvalScores, evals_df
)

evals = evals_df(
    logs=log_dir, 
    columns= EvalTask + EvalModel + EvalScores
)

evals.columns
Index(['eval_id', 'task_name', 'task_display_name', 'task_version',
       'task_file', 'task_attribs', 'task_arg_condition',
       'task_arg_instruction_template', 'task_arg_n_trials',
       'task_arg_n_turns', 'solver', 'solver_args', 'sandbox_type',
       'sandbox_config', 'model', 'model_base_url', 'model_args',
       'model_generate_config', 'model_roles',
       'score_instruction_following_accuracy',
       'score_instruction_following_stderr',
       'score_prediction_accuracy_accuracy',
       'score_prediction_accuracy_stderr',
       'score_prediction_instruction_accuracy',
       'score_prediction_instruction_stderr', 'log'],
      dtype='str')
```

Key columns for plotting:
- task_name: to distinguish between protocols
- task_arg_X: where X is one of [condition, instruction_template, n_turns]
- model
- metric columns: for each scorer, there are three columns at the eval level. score_{name}_accuracy, score_{name}_stderr. These are computed across samples for a given task configuration. This is the level of aggregation we want for the plotting functions, no need to load samples_df unless specified.
	

### 2.1 Key Transformations

Useful columns for analysis:
1. **`condition_pair`**: Groups conditions into natural pairs
   - `value`: value_aligned_cats + value_misaligned_cats
   - `factual`: factual_aligned_earth + factual_misaligned_earth
   - `neutral`: neutral (uses hint as factor)

2. **`instruction_aligned`**: Boolean factor within each pair
   - For `value`: True if instruction asks for positive value
   - For `factual`: True if instruction asks for true statement
   - For `neutral`: Maps to `task_arg_hint`

3. task configuration id: concatenate the value of all columns starting with `task_arg_` to obtain a readable identifier

#TODO: add here any other preprocessing steps useful for the plots in section 3.
---
## 3. Plots 
### 3.1 General rules for plots
- inspect-viz is built on top of observable plot. You have an observable plot skill, remember to load it
- use interactivity (dropdowns mostly): https://meridianlabs-ai.github.io/inspect_viz/components-interactivity.html
- the neutral condition results can be used as baseline, but keep in mind they also vary by the "instructiom" arg
- Aside from the neutral condition, all the others are paired: e.g. value_aligned_cats, value_misaligned_cats.. there is already a utility function that groups the pairs together. this is useful for faceting and splitting plots and for interactivity. But always remember that all conditions have multiple "instruction" values
- Always use evals_df unless instructed otherwise. evals_df, coompared to samples_df, already averages results across samples
- Never average across conditions or across instructions unless explicitly asked in the plot list

### 3.2 Plot list 
Note: You MUST be able to see the observable plot pages in order to perform these tasks. Alert the user if that is not the case.

- a labeled multi line chart: https://observablehq.com/@observablehq/plot-labeled-multi-line-chart with two dropdowns to select content: condition and instruction. then one labeled line per model
- a bullet graph: https://observablehq.com/@observablehq/plot-bullet-graph. You should show a set of KxV bullet graphs, where K is the number of conditions except neutral, and V is the number of instruction settings. The plots should be concatenated horizontally for instruction and vertically for condition. The "background data" (the grey bar) should be the accuracy score for the neutral condition. On the y axis, the model
- the same bullet graph as above, but instead of showing neutral as a background, you should split according to the pair (eg value_aligned_cats + value_misaligned_cats)

- A heatmap like the one that's already implemented, N on the x axis, condition + instruction combo on the y, dropdown to pick model

### Protocol 1: Behavioral

- `scores_by_factor()` compares a boolean factor across models. We decompose conditions into natural pairs where the factor represents a meaningful experimental manipulation:

| Pair | Factor | True (Aligned) | False (Misaligned) |
|------|--------|----------------|-------------------|
| **Value** | Instruction value-aligned | `value_aligned_cats` (T="I love cats") | `value_misaligned_cats` (T="I hate cats") |
| **Truth** | Instruction truth-aligned | `factual_aligned_earth` (T="earth is round") | `factual_misaligned_earth` (T="earth is flat") |
| **Neutral** | Has hardcoding hint | `hint=True` | `hint=False` |

**Key insight:** Each pair isolates a single dimension of variation. The comparison tells us whether instruction alignment (with values/truth) or hint awareness affects instruction-following rates.

## 4. Quarto Notebook Templates

### Protocol 1
TODO

### Protocol 2
TODO

---

## 5. Publishing

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

---

## 6. Data Preparation Script

### 6.1 `scripts/prepare_viz_data.py`
