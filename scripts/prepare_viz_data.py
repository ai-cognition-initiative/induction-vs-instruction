from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import EvalModel, EvalScores, EvalTask, evals_df


# --- Condition pairing logic ---

CONDITION_PAIR_MAP = {
    "value_pattern": "value",
    "value_target": "value",
    "factual_pattern": "factual",
    "factual_target": "factual",
    "neutral": "neutral",
    "token_pattern_states": "token_pattern",
    "token_pattern_countries": "token_pattern",
    "language_fr_ru": "language",
    "language_ru_fr": "language",
    "persona_formal_casual": "persona",
    "persona_casual_formal": "persona",
    "style_uppercase_lowercase": "style_case",
    "style_lowercase_uppercase": "style_case",
    "style_short_long": "style_length",
    "style_long_short": "style_length",
    "style_python_javascript": "style_code",
    "style_javascript_python": "style_code",
    "preference_cats_dogs": "preference",
    "preference_dogs_cats": "preference",
}

ALIGNED_CONDITIONS = {
    "value_pattern",
    "factual_pattern",
    "token_pattern_states",
    "language_fr_ru",
    "persona_formal_casual",
    "style_uppercase_lowercase",
    "style_short_long",
    "style_python_javascript",
    "preference_cats_dogs",
}

MISALIGNED_CONDITIONS = {
    "value_target",
    "factual_target",
    "token_pattern_countries",
    "language_ru_fr",
    "persona_casual_formal",
    "style_lowercase_uppercase",
    "style_long_short",
    "style_javascript_python",
    "preference_dogs_cats",
}


def add_pairing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add condition_pair and instruction_aligned columns."""
    df = df.copy()
    df["condition_pair"] = df["condition"].map(CONDITION_PAIR_MAP)
    df["instruction_aligned"] = df["condition"].apply(
        lambda c: True if c in ALIGNED_CONDITIONS
        else (False if c in MISALIGNED_CONDITIONS else None)
    )
    neutral_mask = df["condition"] == "neutral"
    df.loc[neutral_mask, "instruction_aligned"] = True
    return df


def _coalesce_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Unify multiple score_*_accuracy columns into a single score + score_stderr."""
    acc_cols = [c for c in df.columns if c.endswith("_accuracy")]
    stderr_cols = [c for c in df.columns if c.endswith("_stderr")]

    # Coalesce: take first non-null accuracy value per row
    df["score"] = pd.NA
    for col in acc_cols:
        df["score"] = df["score"].fillna(df[col])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # Match stderr to whichever accuracy column was used
    df["score_stderr"] = pd.NA
    for acc_col in acc_cols:
        stderr_col = acc_col.replace("_accuracy", "_stderr")
        if stderr_col in df.columns:
            mask = df["score_stderr"].isna() & df[acc_col].notna()
            df.loc[mask, "score_stderr"] = df.loc[mask, stderr_col]
    df["score_stderr"] = pd.to_numeric(df["score_stderr"], errors="coerce")

    return df


def prepare_data(log_dir: str, output_dir: str) -> None:
    """Prepare eval logs for visualization — outputs evals.parquet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    evals = evals_df(logs=log_dir, columns=EvalTask + EvalModel + EvalScores)

    # Rename task_arg columns to clean names
    df = evals.rename(columns={
        "task_arg_condition": "condition",
        "task_arg_n_turns": "n_turns",
        "task_arg_instruction_template": "instruction",
    })

    # Shorten model names (strip provider prefixes like "openrouter/google/")
    df["model"] = df["model"].str.replace(r"^openrouter/[^/]+/", "", regex=True)

    # Coalesce score columns
    df = _coalesce_scores(df)

    # Add pairing columns
    df = add_pairing_columns(df)

    # Convert n_turns to string for widget compatibility
    # Sort numerically first so line marks connect properly
    df["_sort"] = pd.to_numeric(df["n_turns"], errors="coerce")
    df = df.sort_values(["model", "condition", "instruction", "_sort"])
    df["n_turns"] = df["_sort"].astype(int).astype(str)
    df = df.drop(columns=["_sort"])

    # Select output columns
    output_cols = [
        "model", "condition", "condition_pair",
        "instruction_aligned", "instruction", "n_turns", "score", "score_stderr",
    ]
    df = df[output_cols].reset_index(drop=True)

    # Drop rows with no score
    df = df.dropna(subset=["score"])

    path = out / "evals.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Instructions: {sorted(df['instruction'].unique())}")
    print(f"  N values: {sorted(df['n_turns'].unique(), key=int)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare eval data for visualization")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    prepare_data(args.log_dir, args.output_dir)
