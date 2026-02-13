from __future__ import annotations

import argparse

import pandas as pd
from inspect_ai.analysis import evals_df


def add_pairing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add condition_pair and instruction_aligned columns for factor analysis."""

    df = df.copy()

    df["condition_pair"] = df["task_arg_condition"].map(
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

    df["instruction_aligned"] = df["task_arg_condition"].map(condition_to_aligned)

    return df


def prepare_data(log_dir: str, output_path: str):
    """Prepare eval logs for visualization."""

    df = evals_df(log_dir)

    df = add_pairing_columns(df)

    df.to_parquet(output_path)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare eval data for visualization")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    prepare_data(args.log_dir, args.output)
