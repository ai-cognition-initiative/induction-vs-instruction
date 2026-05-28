from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
from inspect_ai.analysis import (
    EvalModel,
    EvalScores,
    EvalTask,
    SampleScores,
    SampleSummary,
    evals_df,
    samples_df,
)
from inspect_ai.log import read_eval_log


def _n_turns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert n_turns from int to string for inspect-viz select() widget compatibility."""
    df = df.copy()
    df["n_turns"] = df["n_turns"].astype(str)
    return df


# --- Condition pairing logic ---

CONDITION_PAIR_MAP = {
    "value_aligned_cats": "value",
    "value_misaligned_cats": "value",
    "value_aligned_helpful": "value",
    "value_misaligned_helpful": "value",
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
    "preference_aligned_helpful": "preference",
    "preference_misaligned_helpful": "preference",
}

ALIGNED_CONDITIONS = {
    "value_aligned_cats",
    "value_aligned_helpful",
    "factual_aligned_earth",
    "preference_aligned_cats",
    "preference_aligned_helpful",
    "token_countries_states",
    "language_fr_ru",
}

MISALIGNED_CONDITIONS = {
    "value_misaligned_cats",
    "value_misaligned_helpful",
    "factual_misaligned_earth",
    "preference_misaligned_cats",
    "preference_misaligned_helpful",
    "token_states_countries",
    "language_ru_fr",
}

# Condition pairs where "aligned" means instruction agrees with model values/truth.
# Other pairs (token, language, persona, style) are direction-flipped but neither
# direction is inherently more "aligned" than the other.
ALIGNMENT_AXIS_PAIRS = {"value", "factual", "preference"}

# Behavioral metric column rename map (multi-metric scorer)
BEHAVIORAL_SCORE_RENAME = {
    "score_instruction_following_accuracy": "score",
    "score_instruction_following_stderr": "score_stderr",
    "score_unknown_accuracy": "score_unknown",
    "score_unknown_stderr": "stderr_unknown",
}

# Prediction metric column rename map
PREDICTION_SCORE_RENAME = {
    "score_instruction_following_accuracy": "score_instruction_following",
    "score_instruction_following_stderr": "stderr_instruction_following",
    "score_prediction_accuracy_accuracy": "score_prediction_accuracy",
    "score_prediction_accuracy_stderr": "stderr_prediction_accuracy",
    "score_prediction_instruction_accuracy": "score_prediction_instruction",
    "score_prediction_instruction_stderr": "stderr_prediction_instruction",
    "score_actual_unknown_accuracy": "score_actual_unknown",
    "score_actual_unknown_stderr": "stderr_actual_unknown",
    "score_prediction_unknown_accuracy": "score_prediction_unknown",
    "score_prediction_unknown_stderr": "stderr_prediction_unknown",
}


def _get_reasoning_suffix_from_config(config_str: object) -> str | None:
    """Extract non-default reasoning_effort from a model_generate_config JSON string."""
    if not isinstance(config_str, str):
        return None
    try:
        effort = json.loads(config_str).get("reasoning_effort")
        return effort if effort and effort != "none" else None
    except (ValueError, TypeError):
        return None


def _load_model_args_map(log_dirs: str | Sequence[str]) -> dict[str, dict]:
    """Build eval_id → model_args dict by reading log headers.

    evals_df() does not expose model_args (always NA), so we read raw log
    headers to extract reasoning_enabled and other model-level settings.
    """
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]

    result: dict[str, dict] = {}
    for log_dir in log_dirs:
        for f in sorted(Path(log_dir).glob("*.eval")):
            try:
                log = read_eval_log(str(f), header_only=True)
                result[log.eval.eval_id] = log.eval.model_args or {}
            except Exception:
                pass
    return result


def _get_reasoning_suffix(config_str: object, model_args: dict | None) -> str | None:
    """Return reasoning suffix for model name, or None if no reasoning active.

    Checks (in order):
      1. reasoning_enabled=True in model_args → returns "reasoning"
      2. non-default reasoning_effort in model_generate_config → returns that effort string
    """
    if model_args and model_args.get("reasoning_enabled") is True:
        return "reasoning"
    return _get_reasoning_suffix_from_config(config_str)


def _get_reasoning_tokens(model_usage: object) -> int:
    """Extract reasoning_tokens from a ModelUsage object or dict."""
    if model_usage is None:
        return 0
    if isinstance(model_usage, dict):
        return model_usage.get("reasoning_tokens", 0) or 0
    return getattr(model_usage, "reasoning_tokens", 0) or 0


def load_reasoning_tokens(log_dirs: str | Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame with (eval_id, reasoning_tokens) summed across samples.

    Uses samples_df() to read actual token usage — the ground truth for whether
    reasoning was active during an eval, regardless of configuration.

    Requires inspect-ai >= 0.3.93 (sample summaries with model_usage).
    """
    if isinstance(log_dirs, str):
        log_dirs = [log_dirs]

    dfs = []
    for log_dir in log_dirs:
        s = samples_df(logs=log_dir, columns=SampleSummary)
        dfs.append(s)
    all_samples = pd.concat(dfs, ignore_index=True)

    all_samples["_rt"] = all_samples["model_usage"].apply(_get_reasoning_tokens)
    return (
        all_samples.groupby("eval_id")["_rt"]
        .sum()
        .reset_index()
        .rename(columns={"_rt": "reasoning_tokens"})
    )


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

    # Shorten model names (extract just the model name after the last /)
    df["model"] = df["model"].astype(str).str.split("/").str[-1]
    df["model"] = df["model"].str.replace(r"^(openai-|anthropic-)", "", regex=True)
    df["model"] = df["model"].str.lower()

    # Append reasoning suffix to model name when non-default (e.g. "modelname-medium", "modelname-reasoning")
    _args_map = _load_model_args_map(log_dir)
    _model_args_col = df["eval_id"].map(_args_map)
    _suffix = pd.Series(
        [
            _get_reasoning_suffix(cfg, args)
            for cfg, args in zip(df["model_generate_config"], _model_args_col)
        ],
        index=df.index,
    )
    _mask = _suffix.notna()
    df.loc[_mask, "model"] = df.loc[_mask, "model"] + "-" + _suffix[_mask]

    # Add pairing columns
    df = add_pairing_columns(df)

    # Sort by numeric n_turns, convert to string for widget compatibility
    df["_sort"] = pd.to_numeric(df["n_turns"], errors="coerce")
    df = df.sort_values(["model", "condition", "instruction", "_sort"])
    df["n_turns"] = df["_sort"].astype(int).astype(str)
    df = df.drop(columns=["_sort"])

    # Annotate with actual reasoning token usage (ground truth from sample-level data)
    reasoning = load_reasoning_tokens(log_dir)
    df = df.merge(reasoning, on="eval_id", how="left")
    df["reasoning_tokens"] = df["reasoning_tokens"].fillna(0).astype(int)

    return df, out


def prepare_behavioral(log_dir: str, output_dir: str) -> None:
    """Prepare behavioral eval logs — outputs evals.parquet."""
    df, out = _common_prep(log_dir, output_dir)

    if (
        "task_name" not in df.columns
        or not (df["task_name"] == "behavioral_baseline").any()
    ):
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

    # Coalesce all *_accuracy columns into a single score (handles mixed old/new scorer logs)
    df = _coalesce_scores(df)
    # Rename unknown metric columns from multi-metric scorer if present
    unknown_renames = {
        k: v for k, v in BEHAVIORAL_SCORE_RENAME.items() if "unknown" in k
    }
    df = df.rename(columns=unknown_renames)
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
        "score_unknown",
        "stderr_unknown",
        "reasoning_tokens",
    ]
    # Keep only columns that exist (backwards compat with older logs without unknown metric)
    output_cols = [c for c in output_cols if c in df.columns]
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

    if (
        "task_name" not in df.columns
        or not (df["task_name"] == "self_prediction").any()
    ):
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
        "score_actual_unknown",
        "stderr_actual_unknown",
        "score_prediction_unknown",
        "stderr_prediction_unknown",
        "reasoning_tokens",
    ]
    # Keep only columns that exist (backwards compat with older logs without unknown metrics)
    output_cols = [c for c in output_cols if c in df.columns]
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

    df_combined = _merge_behavioral_prediction(df_behavioral, df_prediction)

    path = out / "evals_combined.parquet"
    df_combined.to_parquet(path, index=False)
    print(f"Saved {len(df_combined)} rows to {path}")
    print(f"  Models: {df_combined['model'].nunique()}")
    print(f"  Conditions: {sorted(df_combined['condition'].unique())}")
    print(f"  Instructions: {sorted(df_combined['instruction'].unique())}")
    print(f"  N values: {sorted(df_combined['n_turns'].unique(), key=int)}")


# --- Reusable functions for multi-folder merging ---


def load_evals_from_folders(log_dirs: Sequence[str]) -> pd.DataFrame:
    """Load and concatenate evals from multiple log directories."""
    dfs = []
    for log_dir in log_dirs:
        path = Path(log_dir)
        if not path.exists():
            raise FileNotFoundError(f"Log directory not found: {log_dir}")
        evals = evals_df(logs=log_dir, columns=EvalTask + EvalModel + EvalScores)
        dfs.append(evals)
    return pd.concat(dfs, ignore_index=True)


def process_raw_evals(
    df: pd.DataFrame,
    model_args_map: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Apply common transformations to raw evals dataframe.

    Args:
        df: Raw evals dataframe from load_evals_from_folders().
        model_args_map: Optional eval_id → model_args dict from _load_model_args_map().
            Required to detect reasoning_enabled=True for model name suffixes.
    """
    df = df.rename(
        columns={
            "task_arg_condition": "condition",
            "task_arg_n_turns": "n_turns",
            "task_arg_instruction_template": "instruction",
        }
    )
    df["model"] = df["model"].astype(str).str.split("/").str[-1]
    df["model"] = df["model"].str.replace(r"^(openai-|anthropic-)", "", regex=True)
    df["model"] = df["model"].str.lower()

    # Append reasoning suffix to model name when non-default (e.g. "modelname-medium", "modelname-reasoning")
    if "model_generate_config" in df.columns or model_args_map:
        _model_args_col = (
            df["eval_id"].map(model_args_map) if model_args_map else pd.Series(
                [None] * len(df), index=df.index
            )
        )
        _config_col = df.get("model_generate_config", pd.Series([None] * len(df), index=df.index))
        _suffix = pd.Series(
            [
                _get_reasoning_suffix(cfg, args)
                for cfg, args in zip(_config_col, _model_args_col)
            ],
            index=df.index,
        )
        _mask = _suffix.notna()
        df.loc[_mask, "model"] = df.loc[_mask, "model"] + "-" + _suffix[_mask]

    df = add_pairing_columns(df)
    df["n_turns"] = pd.to_numeric(df["n_turns"], errors="coerce").astype(int)
    df = df.sort_values(["model", "condition", "instruction", "n_turns"])
    return df


def process_behavioral_df(
    df: pd.DataFrame, max_reasoning_tokens: int | None = None
) -> pd.DataFrame:
    """Process a dataframe containing behavioral logs.

    Args:
        df: Raw evals dataframe (output of process_raw_evals, with reasoning_tokens if annotated).
        max_reasoning_tokens: If set, exclude models where total reasoning_tokens exceeds
            this value. Use 0 for strict non-reasoning-only filter, or e.g. 1000 to allow
            noise-level tokens (some non-reasoning models emit a small number).
    """
    if (
        "task_name" not in df.columns
        or not (df["task_name"] == "behavioral_baseline").any()
    ):
        raise ValueError("No behavioral_baseline tasks found in dataframe")

    df = df[df["task_name"] == "behavioral_baseline"]
    acc_cols = [c for c in df.columns if c.endswith("_accuracy")]
    if not acc_cols:
        raise ValueError("No score columns found in behavioral dataframe")

    # Coalesce all *_accuracy columns into a single score (handles mixed old/new scorer logs)
    df = _coalesce_scores(df)
    # Rename unknown metric columns from multi-metric scorer if present
    unknown_renames = {
        k: v for k, v in BEHAVIORAL_SCORE_RENAME.items() if "unknown" in k
    }
    df = df.rename(columns=unknown_renames)
    df = df.dropna(subset=["score"])

    if df.empty:
        raise ValueError("No behavioral scores found")

    if max_reasoning_tokens is not None and "reasoning_tokens" in df.columns:
        df = df[df["reasoning_tokens"] <= max_reasoning_tokens]
        if df.empty:
            raise ValueError(
                f"No behavioral scores after filtering to max_reasoning_tokens={max_reasoning_tokens}"
            )

    output_cols = [
        "model",
        "condition",
        "condition_pair",
        "instruction_aligned",
        "instruction",
        "n_turns",
        "score",
        "score_stderr",
        "score_unknown",
        "stderr_unknown",
    ]
    if "reasoning_tokens" in df.columns:
        output_cols.append("reasoning_tokens")
    # Keep only columns that exist (backwards compat with older logs without unknown metric)
    output_cols = [c for c in output_cols if c in df.columns]
    return df[output_cols].reset_index(drop=True)


def process_prediction_df(
    df: pd.DataFrame, max_reasoning_tokens: int | None = None
) -> pd.DataFrame:
    """Process a dataframe containing prediction logs.

    Args:
        df: Raw evals dataframe (output of process_raw_evals, with reasoning_tokens if annotated).
        max_reasoning_tokens: If set, exclude models where total reasoning_tokens exceeds
            this value. See process_behavioral_df for details.
    """
    if (
        "task_name" not in df.columns
        or not (df["task_name"] == "self_prediction").any()
    ):
        raise ValueError("No self_prediction tasks found in dataframe")

    df = df[df["task_name"] == "self_prediction"]
    if "score_prediction_accuracy_accuracy" not in df.columns:
        raise ValueError("Missing prediction scores in dataframe")

    df = _rename_prediction_scores(df)

    if max_reasoning_tokens is not None and "reasoning_tokens" in df.columns:
        df = df[df["reasoning_tokens"] <= max_reasoning_tokens]
        if df.empty:
            raise ValueError(
                f"No prediction scores after filtering to max_reasoning_tokens={max_reasoning_tokens}"
            )

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
        "score_actual_unknown",
        "stderr_actual_unknown",
        "score_prediction_unknown",
        "stderr_prediction_unknown",
    ]
    if "reasoning_tokens" in df.columns:
        output_cols.append("reasoning_tokens")
    # Keep only columns that exist (backwards compat with older logs without unknown metrics)
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols].reset_index(drop=True)
    df = df.dropna(subset=["score_instruction_following"])

    if df.empty:
        raise ValueError("No prediction scores found")

    return df


def _merge_behavioral_prediction(
    df_behavioral: pd.DataFrame, df_prediction: pd.DataFrame
) -> pd.DataFrame:
    """Merge processed behavioral and prediction dataframes.

    Both dataframes should be raw evals (output of process_raw_evals).
    This function filters by task type and processes accordingly.
    """
    df_behavioral = df_behavioral.copy()
    df_prediction = df_prediction.copy()

    if "task_name" in df_behavioral.columns:
        df_behavioral = df_behavioral[
            df_behavioral["task_name"] == "behavioral_baseline"
        ]

    if "task_name" in df_prediction.columns:
        df_prediction = df_prediction[df_prediction["task_name"] == "self_prediction"]

    acc_cols = [c for c in df_behavioral.columns if c.endswith("_accuracy")]
    if not acc_cols:
        raise ValueError("No score columns found in behavioral logs")

    # Coalesce all *_accuracy columns into a single score (handles mixed old/new scorer logs)
    df_behavioral = _coalesce_scores(df_behavioral)
    unknown_renames = {
        k: v for k, v in BEHAVIORAL_SCORE_RENAME.items() if "unknown" in k
    }
    df_behavioral = df_behavioral.rename(columns=unknown_renames)

    if "score_prediction_accuracy_accuracy" not in df_prediction.columns:
        raise ValueError("Missing prediction scores in prediction logs")

    df_prediction = _rename_prediction_scores(df_prediction)

    key_cols = ["model", "condition", "instruction", "n_turns"]
    behavioral_cols = ["score", "score_stderr"]
    if "score_unknown" in df_behavioral.columns:
        behavioral_cols += ["score_unknown", "stderr_unknown"]
    df_behavioral = df_behavioral[key_cols + behavioral_cols].copy()
    rename_map = {"score": "behavioral_score", "score_stderr": "behavioral_stderr"}
    if "score_unknown" in df_behavioral.columns:
        rename_map["score_unknown"] = "behavioral_unknown"
        rename_map["stderr_unknown"] = "behavioral_unknown_stderr"
    df_behavioral = df_behavioral.rename(columns=rename_map)

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
    ].copy()
    df_prediction = df_prediction.rename(
        columns={
            "score_instruction_following": "prediction_actual_score",
            "stderr_instruction_following": "prediction_actual_stderr",
            "score_prediction_instruction": "prediction_predicted_score",
            "stderr_prediction_instruction": "prediction_predicted_stderr",
        }
    )

    df_combined = df_prediction.merge(df_behavioral, on=key_cols, how="inner")

    if df_combined.empty:
        raise ValueError("No matching rows between behavioral and prediction logs")

    return df_combined


def prepare_behavioral_multi(
    log_dirs: Sequence[str],
    output_dir: str,
    max_reasoning_tokens: int | None = None,
) -> None:
    """Prepare behavioral eval logs from multiple folders — outputs evals.parquet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_evals_from_folders(log_dirs)
    model_args_map = _load_model_args_map(log_dirs)
    df = process_raw_evals(df, model_args_map=model_args_map)

    reasoning = load_reasoning_tokens(log_dirs)
    df = df.merge(reasoning, on="eval_id", how="left")
    df["reasoning_tokens"] = df["reasoning_tokens"].fillna(0).astype(int)

    df = process_behavioral_df(df, max_reasoning_tokens=max_reasoning_tokens)
    df = _n_turns_to_string(df)

    path = out / "evals.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Instructions: {sorted(df['instruction'].unique())}")
    print(f"  N values: {sorted(df['n_turns'].unique(), key=int)}")
    if max_reasoning_tokens is not None:
        print(f"  Filtered: max_reasoning_tokens={max_reasoning_tokens}")


def prepare_prediction_multi(
    log_dirs: Sequence[str],
    output_dir: str,
    max_reasoning_tokens: int | None = None,
) -> None:
    """Prepare prediction eval logs from multiple folders — outputs evals_prediction.parquet."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_evals_from_folders(log_dirs)
    model_args_map = _load_model_args_map(log_dirs)
    df = process_raw_evals(df, model_args_map=model_args_map)

    reasoning = load_reasoning_tokens(log_dirs)
    df = df.merge(reasoning, on="eval_id", how="left")
    df["reasoning_tokens"] = df["reasoning_tokens"].fillna(0).astype(int)

    df = process_prediction_df(df, max_reasoning_tokens=max_reasoning_tokens)
    df = _n_turns_to_string(df)

    path = out / "evals_prediction.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Instructions: {sorted(df['instruction'].unique())}")
    print(f"  N values: {sorted(df['n_turns'].unique(), key=int)}")
    if max_reasoning_tokens is not None:
        print(f"  Filtered: max_reasoning_tokens={max_reasoning_tokens}")


def prepare_combined_multi(
    behavioral_log_dirs: Sequence[str],
    prediction_log_dirs: Sequence[str],
    output_dir: str,
    max_reasoning_tokens: int | None = None,
) -> None:
    """Combine behavioral and prediction logs from multiple folders."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_behavioral = load_evals_from_folders(behavioral_log_dirs)
    model_args_b = _load_model_args_map(behavioral_log_dirs)
    df_behavioral = process_raw_evals(df_behavioral, model_args_map=model_args_b)
    reasoning_b = load_reasoning_tokens(behavioral_log_dirs)
    df_behavioral = df_behavioral.merge(reasoning_b, on="eval_id", how="left")
    df_behavioral["reasoning_tokens"] = (
        df_behavioral["reasoning_tokens"].fillna(0).astype(int)
    )

    df_prediction = load_evals_from_folders(prediction_log_dirs)
    model_args_p = _load_model_args_map(prediction_log_dirs)
    df_prediction = process_raw_evals(df_prediction, model_args_map=model_args_p)
    reasoning_p = load_reasoning_tokens(prediction_log_dirs)
    df_prediction = df_prediction.merge(reasoning_p, on="eval_id", how="left")
    df_prediction["reasoning_tokens"] = (
        df_prediction["reasoning_tokens"].fillna(0).astype(int)
    )

    if max_reasoning_tokens is not None:
        if "reasoning_tokens" in df_behavioral.columns:
            df_behavioral = df_behavioral[
                df_behavioral["reasoning_tokens"] <= max_reasoning_tokens
            ]
        if "reasoning_tokens" in df_prediction.columns:
            df_prediction = df_prediction[
                df_prediction["reasoning_tokens"] <= max_reasoning_tokens
            ]

    df_combined = _merge_behavioral_prediction(df_behavioral, df_prediction)
    df_combined = _n_turns_to_string(df_combined)

    path = out / "evals_combined.parquet"
    df_combined.to_parquet(path, index=False)
    print(f"Saved {len(df_combined)} rows to {path}")
    print(f"  Models: {df_combined['model'].nunique()}")
    print(f"  Conditions: {sorted(df_combined['condition'].unique())}")
    print(f"  Instructions: {sorted(df_combined['instruction'].unique())}")
    print(f"  N values: {sorted(df_combined['n_turns'].unique(), key=int)}")


# --- LLM-judge inter-rater data extraction ---

# Conditions whose scorers run multi-judge LLM scoring. Used to identify which
# samples carry per-judge metadata for IRR analysis.
_CODE_CONDITIONS = {"style_python_javascript", "style_javascript_python"}


def _code_language_to_classification(language: str, condition_name: str) -> str:
    """Map a code-language judgment to target/pattern/unknown.

    Mirrors the direction logic in src/scorers/format_check.py.
    """
    if condition_name == "style_python_javascript":
        target_lang, pattern_lang = "python", "javascript"
    else:
        target_lang, pattern_lang = "javascript", "python"
    if language == target_lang:
        return "target"
    if language == pattern_lang:
        return "pattern"
    return "unknown"


def _explode_judges(meta: dict, condition: str) -> list[tuple[str, str]]:
    """Return list of (judge_id, classification) from one sample's scorer metadata.

    Handles both metadata shapes:
      - persona/preference/variety/language: has `judge_classifications`
      - style_python_javascript/style_javascript_python: has `judge_votes` keyed by
        judge_id with language values (python/javascript/unknown) — needs remap.
    """
    if not isinstance(meta, dict):
        return []

    classifications = meta.get("judge_classifications")
    if isinstance(classifications, dict) and classifications:
        return [(jid, cls) for jid, cls in classifications.items()]

    if condition in _CODE_CONDITIONS:
        votes = meta.get("judge_votes")
        if isinstance(votes, dict) and votes:
            return [
                (jid, _code_language_to_classification(lang, condition))
                for jid, lang in votes.items()
            ]

    return []


def prepare_judges(log_dir: str, output_dir: str) -> None:
    """Extract per-sample per-judge classifications to judges.parquet.

    One row per (sample, judge). Used by the inter-rater agreement notebook.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = samples_df(
        logs=log_dir,
        columns=EvalModel + EvalTask + SampleSummary + SampleScores,
    )

    if "task_arg_condition" not in samples.columns:
        print("No condition info in samples — skipping judges.")
        sys.exit(1)

    samples = samples.rename(
        columns={
            "task_arg_condition": "condition",
            "task_arg_n_turns": "n_turns",
            "task_arg_instruction_template": "instruction",
        }
    )

    # Match model-name normalization used by _common_prep
    samples["model"] = samples["model"].astype(str).str.split("/").str[-1]
    samples["model"] = samples["model"].str.replace(
        r"^(openai-|anthropic-)", "", regex=True
    )
    samples["model"] = samples["model"].str.lower()

    samples = add_pairing_columns(samples)
    samples["n_turns"] = (
        pd.to_numeric(samples["n_turns"], errors="coerce").astype(int).astype(str)
    )

    # Locate the metadata column for the LLM-judge scorer in this sample set.
    # SampleScores expands to score_<scorer>_metadata; LLM-judge scorers in this
    # project are style_scorer / language_scorer / format_scorer.
    meta_cols = [c for c in samples.columns if c.endswith("_metadata")]
    if not meta_cols:
        print("No score metadata columns — skipping judges.")
        sys.exit(1)

    records: list[dict] = []
    for _, row in samples.iterrows():
        condition = row.get("condition")
        if not isinstance(condition, str):
            continue
        for col in meta_cols:
            meta = row[col]
            judges = _explode_judges(meta, condition)
            if not judges:
                continue
            agreement_rate = meta.get("agreement_rate") if isinstance(meta, dict) else None
            unanimous = meta.get("unanimous") if isinstance(meta, dict) else None
            for judge_id, classification in judges:
                records.append(
                    {
                        "eval_id": row.get("eval_id"),
                        "sample_id": row.get("sample_id"),
                        "epoch": row.get("epoch"),
                        "model": row["model"],
                        "condition": condition,
                        "condition_pair": row.get("condition_pair"),
                        "instruction_aligned": row.get("instruction_aligned"),
                        "instruction": row.get("instruction"),
                        "n_turns": row["n_turns"],
                        "judge_id": judge_id,
                        "classification": classification,
                        "agreement_rate": agreement_rate,
                        "unanimous": unanimous,
                    }
                )
            break  # one scorer per sample

    if not records:
        print("No LLM-judge samples found — skipping.")
        sys.exit(1)

    df = pd.DataFrame.from_records(records)
    path = out / "judges.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} judge-rows to {path}")
    print(f"  Samples: {df['sample_id'].nunique()}")
    print(f"  Judges: {sorted(df['judge_id'].unique())}")
    print(f"  Models: {df['model'].nunique()}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  N values: {sorted(df['n_turns'].unique(), key=int)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare eval data for visualization")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--protocol",
        choices=["behavioral", "prediction", "combined", "judges"],
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
    elif args.protocol == "judges":
        prepare_judges(args.log_dir, args.output_dir)
    else:
        if not args.log_dir_2:
            parser.error("--log-dir-2 is required for 'combined' protocol")
        prepare_combined(args.log_dir, args.log_dir_2, args.output_dir)
