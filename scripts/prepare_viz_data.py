from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import evals_df, model_info, prepare


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

    df["instruction_aligned"] = df["task_arg_condition"].map(
        {
            "value_pattern": True,
            "value_target": False,
            "factual_pattern": True,
            "factual_target": False,
            "neutral": None,
        }
    )

    neutral_mask = df["condition_pair"] == "neutral"
    df.loc[neutral_mask, "instruction_aligned"] = df.loc[neutral_mask, "task_arg_hint"]

    return df


def prepare_data(log_dir: str, output_path: str):
    """Prepare eval logs for visualization."""

    df = evals_df(log_dir)

    df = add_pairing_columns(df)

    df = prepare(df, [model_info()])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare eval data for visualization")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    prepare_data(args.log_dir, args.output)
