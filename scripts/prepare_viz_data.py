from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from inspect_ai.analysis import evals_df, samples_df


def add_pairing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add condition_pair and instruction_aligned columns for factor analysis."""

    df = df.copy()

    df["condition_pair"] = df["metadata_condition"].map(
        {
            "value_pattern": "value",
            "value_target": "value",
            "factual_pattern": "factual",
            "factual_target": "factual",
            "neutral": "neutral",
        }
    )

    condition_to_aligned = {
        "value_pattern": True,
        "value_target": False,
        "factual_pattern": True,
        "factual_target": False,
    }

    df["instruction_aligned"] = df["metadata_condition"].map(condition_to_aligned)

    # For neutral conditions, use hint as the factor (only if hint is boolean)
    neutral_mask = df["condition_pair"] == "neutral"
    hint_is_bool = df["metadata_hint"].apply(lambda x: isinstance(x, (bool, np.bool_)))
    df.loc[neutral_mask & hint_is_bool, "instruction_aligned"] = df.loc[
        neutral_mask & hint_is_bool, "metadata_hint"
    ]

    return df


def prepare_data(log_dir: str, output_path: str):
    """Prepare eval logs for visualization."""

    samples = samples_df(log_dir)
    evals = evals_df(log_dir)

    # Join samples with evals to get model
    df = samples.merge(evals[["eval_id", "model"]], on="eval_id")

    # Add pairing columns first
    df = add_pairing_columns(df)

    # Drop rows where condition_pair is NA (unknown conditions like token_pattern)
    df = df.dropna(subset=["condition_pair"])

    # Drop rows where instruction_aligned is NA (neutral without boolean hint)
    df = df.dropna(subset=["instruction_aligned"])

    # Drop rows where score_pattern_match is NA (uses different scorer)
    df = df[df["score_pattern_match"].notna()]

    # Compute score as binary: I (instruction) = 1, C (pattern) = 0
    df["score_binary"] = (df["score_pattern_match"] == "I").astype(int)

    # Aggregate by model, condition_pair, n_turns, instruction_aligned
    agg = (
        df.groupby(
            ["model", "condition_pair", "metadata_n_turns", "instruction_aligned"]
        )
        .agg(score_value=("score_binary", "mean"), count=("score_binary", "count"))
        .reset_index()
    )

    # Compute binomial standard error: sqrt(p*(1-p)/n)
    agg["score_stderr"] = np.sqrt(
        agg["score_value"] * (1 - agg["score_value"]) / agg["count"]
    )

    # Rename for consistency with inspect_viz expectations
    agg = agg.rename(columns={"metadata_n_turns": "task_arg_n_turns"})

    agg.to_parquet(output_path)

    print(f"Saved {len(agg)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare eval data for visualization")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    prepare_data(args.log_dir, args.output)
