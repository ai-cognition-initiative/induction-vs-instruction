from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from inspect_ai.analysis import EvalModel, EvalScores, EvalTask, evals_df


# --- Condition pairing logic ---

CONDITION_PAIR_MAP = {
    "value_aligned_cats": "value",
    "value_misaligned_cats": "value",
    "factual_aligned_earth": "factual",
    "factual_misaligned_earth": "factual",
    "neutral": "neutral",
    "token_countries_states": "token",
    "token_states_countries": "token",
    "language_ru_fr": "language",
    "language_fr_ru": "language",
    "persona_casual_formal": "persona",
    "persona_formal_casual": "persona",
    "style_lowercase_uppercase": "style_case",
    "style_uppercase_lowercase": "style_case",
    "style_javascript_python": "style_code",
    "style_python_javascript": "style_code",
    "preference_aligned_cats": "preference",
    "preference_misaligned_cats": "preference",
}

ALIGNED_CONDITIONS = {
    "value_aligned_cats",
    "factual_aligned_earth",
    "token_countries_states",
    "language_ru_fr",
    "persona_casual_formal",
    "style_lowercase_uppercase",
    "style_javascript_python",
    "preference_aligned_cats",
}

MISALIGNED_CONDITIONS = {
    "value_misaligned_cats",
    "factual_misaligned_earth",
    "token_states_countries",
    "language_fr_ru",
    "persona_formal_casual",
    "style_uppercase_lowercase",
    "style_short_long",
    "style_python_javascript",
    "preference_misaligned_cats",
}

# Prediction metric column rename map
PREDICTION_SCORE_RENAME = {
    "score_instruction_following_accuracy": "score_instruction_following",
    "score_instruction_following_stderr": "stderr_instruction_following",
    "score_prediction_accuracy_accuracy": "score_prediction_accuracy",
    "score_prediction_accuracy_stderr": "stderr_prediction_accuracy",
    "score_prediction_instruction_accuracy": "score_prediction_instruction",
    "score_prediction_instruction_stderr": "stderr_prediction_instruction",
}


def add_pairing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add condition_pair and instruction_aligned columns."""
    df = df.copy()
    df["condition_pair"] = df["condition"].map(CONDITION_PAIR_MAP)
    df["instruction_aligned"] = df["condition"].apply(
        lambda c: (
            True
            if c in ALIGNED_CONDITIONS
            else (False if c in MISALIGNED_CONDITIONS else None)
        )
    )
    neutral_mask = df["condition"] == "neutral"
    df.loc[neutral_mask, "instruction_aligned"] = True
    return df


def _coalesce_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Unify multiple score_*_accuracy columns into a single score + score_stderr."""
    acc_cols = [c for c in df.columns if c.endswith("_accuracy")]

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


def _rename_prediction_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Rename prediction metric columns to clean names."""
    return df.rename(columns=PREDICTION_SCORE_RENAME)


def _common_prep(log_dir: str, output_dir: str) -> tuple[pd.DataFrame, Path]:
    """Load evals_df, apply common renames and sorting. Returns (df, out_path)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    evals = evals_df(logs=log_dir, columns=EvalTask + EvalModel + EvalScores)

    df = evals.rename(
        columns={
            "task_arg_condition": "condition",
            "task_arg_n_turns": "n_turns",
            "task_arg_instruction_template": "instruction",
        }
    )

    # Shorten model names (strip provider prefixes like "openrouter/google/")
    df["model"] = df["model"].str.replace(r"^openrouter/[^/]+/", "", regex=True)

    # Add pairing columns
    df = add_pairing_columns(df)

    # Sort by numeric n_turns, convert to string for widget compatibility
    df["_sort"] = pd.to_numeric(df["n_turns"], errors="coerce")
    df = df.sort_values(["model", "condition", "instruction", "_sort"])
    df["n_turns"] = df["_sort"].astype(int).astype(str)
    df = df.drop(columns=["_sort"])

    return df, out


def prepare_behavioral(log_dir: str, output_dir: str) -> None:
    """Prepare behavioral eval logs — outputs evals.parquet."""
    df, out = _common_prep(log_dir, output_dir)

    if "task_name" not in df.columns or not (df["task_name"] == "behavioral_baseline").any():
        print(
            "No behavioral_baseline tasks found — this log directory may not contain behavioral logs."
        )
        sys.exit(1)

    df = df[df["task_name"] == "behavioral_baseline"]

    acc_cols = [c for c in df.columns if c.endswith("_accuracy")]
    if not acc_cols:
        print(
            "No score columns found — this log directory may not contain behavioral logs."
        )
        sys.exit(1)

    df = _coalesce_scores(df)
    df = df.dropna(subset=["score"])

    if df.empty:
        print("No behavioral scores found — skipping.")
        sys.exit(1)

    output_cols = [
        "model",
        "condition",
        "condition_pair",
        "instruction_aligned",
        "instruction",
        "n_turns",
        "score",
        "score_stderr",
    ]
    df = df[output_cols].reset_index(drop=True)

    path = out / "evals.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Instructions: {sorted(df['instruction'].unique())}")
    print(f"  N values: {sorted(df['n_turns'].unique(), key=int)}")


def prepare_prediction(log_dir: str, output_dir: str) -> None:
    """Prepare prediction eval logs — outputs evals_prediction.parquet (wide, 3 metrics)."""
    df, out = _common_prep(log_dir, output_dir)

    if "task_name" not in df.columns or not (df["task_name"] == "self_prediction").any():
        print(
            "No self_prediction tasks found — this log directory may not contain prediction logs."
        )
        sys.exit(1)

    df = df[df["task_name"] == "self_prediction"]

    required_col = "score_prediction_accuracy_accuracy"
    if required_col not in df.columns:
        print(
            f"Missing {required_col!r} — this log directory may not contain prediction logs."
        )
        sys.exit(1)

    df = _rename_prediction_scores(df)

    output_cols = [
        "model",
        "condition",
        "condition_pair",
        "instruction_aligned",
        "instruction",
        "n_turns",
        "score_instruction_following",
        "stderr_instruction_following",
        "score_prediction_accuracy",
        "stderr_prediction_accuracy",
        "score_prediction_instruction",
        "stderr_prediction_instruction",
    ]
    df = df[output_cols].reset_index(drop=True)
    df = df.dropna(subset=["score_instruction_following"])

    if df.empty:
        print("No prediction scores found — skipping.")
        sys.exit(1)

    path = out / "evals_prediction.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Instructions: {sorted(df['instruction'].unique())}")
    print(f"  N values: {sorted(df['n_turns'].unique(), key=int)}")


def prepare_combined(
    behavioral_log_dir: str, prediction_log_dir: str, output_dir: str
) -> None:
    """Combine behavioral and prediction logs for cross-protocol analysis.

    Outputs evals_combined.parquet with columns:
    - model, condition, condition_pair, instruction_aligned, instruction, n_turns
    - behavioral_score, behavioral_stderr (from protocol 1)
    - prediction_actual_score, prediction_actual_stderr (actual IF from protocol 2)
    - prediction_predicted_score, prediction_predicted_stderr (predicted IF from protocol 2)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_behavioral, _ = _common_prep(behavioral_log_dir, output_dir)
    df_prediction, _ = _common_prep(prediction_log_dir, output_dir)

    acc_cols = [c for c in df_behavioral.columns if c.endswith("_accuracy")]
    if not acc_cols:
        print("No score columns found in behavioral logs.")
        sys.exit(1)

    df_behavioral = _coalesce_scores(df_behavioral)

    if "score_prediction_accuracy_accuracy" not in df_prediction.columns:
        print("Missing prediction scores in prediction logs.")
        sys.exit(1)

    df_prediction = _rename_prediction_scores(df_prediction)

    key_cols = ["model", "condition", "instruction", "n_turns"]
    df_behavioral = df_behavioral[key_cols + ["score", "score_stderr"]].rename(
        columns={"score": "behavioral_score", "score_stderr": "behavioral_stderr"}
    )

    df_prediction = df_prediction[
        key_cols
        + [
            "condition_pair",
            "instruction_aligned",
            "score_instruction_following",
            "stderr_instruction_following",
            "score_prediction_instruction",
            "stderr_prediction_instruction",
        ]
    ].rename(
        columns={
            "score_instruction_following": "prediction_actual_score",
            "stderr_instruction_following": "prediction_actual_stderr",
            "score_prediction_instruction": "prediction_predicted_score",
            "stderr_prediction_instruction": "prediction_predicted_stderr",
        }
    )

    df_combined = df_prediction.merge(df_behavioral, on=key_cols, how="inner")

    if df_combined.empty:
        print("No matching rows between behavioral and prediction logs.")
        sys.exit(1)

    path = out / "evals_combined.parquet"
    df_combined.to_parquet(path, index=False)
    print(f"Saved {len(df_combined)} rows to {path}")
    print(f"  Models: {df_combined['model'].nunique()}")
    print(f"  Conditions: {sorted(df_combined['condition'].unique())}")
    print(f"  Instructions: {sorted(df_combined['instruction'].unique())}")
    print(f"  N values: {sorted(df_combined['n_turns'].unique(), key=int)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare eval data for visualization")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--protocol",
        choices=["behavioral", "prediction", "combined"],
        default="behavioral",
        help="Which protocol's logs to prepare (default: behavioral)",
    )
    parser.add_argument(
        "--log-dir-2",
        type=str,
        help="Second log directory (required for 'combined' protocol)",
    )
    args = parser.parse_args()

    if args.protocol == "behavioral":
        prepare_behavioral(args.log_dir, args.output_dir)
    elif args.protocol == "prediction":
        prepare_prediction(args.log_dir, args.output_dir)
    else:
        if not args.log_dir_2:
            parser.error("--log-dir-2 is required for 'combined' protocol")
        prepare_combined(args.log_dir, args.log_dir_2, args.output_dir)
