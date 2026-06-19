"""Extract per-model statistics for paper tables.

Outputs:
  - results_table_data.json   (per-model IF rates, first N, gap, prediction stats)
  - reasoning_table_data.json (reasoning vs non-reasoning comparison)
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from scipy import stats
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

STATIC_PATH = ROOT / "outputs" / "viz" / "static"
DYNAMIC_PATH = ROOT / "outputs" / "viz" / "dynamic"
TOKEN_CONDITIONS = ["token_states_countries", "token_countries_states"]

MODEL_ALIASES: dict[str, str] = {
    "claude-4.6-sonnet": "claude-sonnet-4.6",  # Gradient/legacy spelling
    "kimi-k2-instruct": "kimi-k2",  # Nebius/old OpenRouter spelling
    "kimi-k2-0905": "kimi-k2",  # OpenRouter 09-05 release spelling
}

DISPLAY_NAMES: dict[str, str] = {
    "claude-sonnet-4.6": "Claude 4.6 Sonnet",
    "claude-opus-4.6": "Claude Opus 4.6",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemma-3-12b-it": "Gemma-3 12B",
    "gemma-3-27b-it": "Gemma-3 27B",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2-medium": "GPT-5.2 (medium)",
    "hermes-4-70b": "Hermes-4 70B",
    "hermes-4-70b-reasoning": "Hermes-4 70B (reasoning)",
    "kimi-k2": "Kimi K2",
    "llama-3.1-70b-instruct": "Llama 3.1 70B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "olmo-3.1-32b-instruct": "OLMo 3.1 32B",
    "olmo-3.1-32b-instruct-dpo": "OLMo 3.1 32B (SFT+DPO)",
    "olmo-3.1-32b-instruct-sft": "OLMo 3.1 32B (SFT)",
    "qwen3-235b-a22b-instruct-2507": "Qwen3 235B A22B",
    "qwen3-30b-a3b-instruct-2507": "Qwen3 30B A3B",
}

CORE_MODELS = [
    "claude-sonnet-4.6", "claude-opus-4.6", "gemini-2.5-flash",
    "gemma-3-12b-it", "gemma-3-27b-it", "gpt-5.2",
    "hermes-4-70b", "kimi-k2",
    "llama-3.1-70b-instruct",
    "llama-3.3-70b-instruct",
    "olmo-3.1-32b-instruct", "qwen3-235b-a22b-instruct-2507",
    "qwen3-30b-a3b-instruct-2507",
]

REASONING_MODELS = ["gpt-5.2", "gpt-5.2-medium", "hermes-4-70b", "hermes-4-70b-reasoning"]


def normalize(df: pl.DataFrame) -> pl.DataFrame:
    models = df["model"].unique().to_list()
    full_map = {m: MODEL_ALIASES.get(m, m) for m in models}
    return df.with_columns(pl.col("model").replace(full_map))


def first_n_below(df: pl.DataFrame, threshold: float = 0.5) -> dict[str, str]:
    by_model_n = (
        df.with_columns(pl.col("n_turns").cast(pl.Int64).alias("n_int"))
        .group_by(["model", "n_int"])
        .agg(pl.col("score").mean().alias("mean_score"))
        .sort(["model", "n_int"])
    )
    below = (
        by_model_n.filter(pl.col("mean_score") <= threshold)
        .sort(["model", "n_int"])
        .group_by("model")
        .agg(pl.col("n_int").first().alias("first_n"))
    )
    result: dict[str, str] = {}
    for model in df["model"].unique().to_list():
        row = below.filter(pl.col("model") == model)
        result[model] = str(int(row["first_n"][0])) if len(row) > 0 else ">"
    return result


def avg_if_with_ci(df: pl.DataFrame) -> dict[str, dict]:
    agg = (
        df.group_by("model")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("sd"),
            pl.col("score").len().alias("n"),
        )
        .with_columns((pl.col("sd") / pl.col("n").sqrt()).alias("se"))
        .with_columns(
            (pl.col("mean") - 1.96 * pl.col("se")).alias("ci_lo"),
            (pl.col("mean") + 1.96 * pl.col("se")).alias("ci_hi"),
        )
    )
    result: dict[str, dict] = {}
    for row in agg.iter_rows(named=True):
        result[row["model"]] = {
            "mean": round(row["mean"], 3),
            "ci_lo": round(row["ci_lo"], 3),
            "ci_hi": round(row["ci_hi"], 3),
        }
    return result


def alignment_gap(df: pl.DataFrame) -> dict[str, dict]:
    align_df = df.filter(pl.col("condition_pair").is_in(["value", "factual"]))
    per_model = (
        align_df.group_by(["model", "instruction_aligned"])
        .agg(pl.col("score").mean().alias("mean_if"))
    )
    aligned = per_model.filter(pl.col("instruction_aligned")).sort("model")
    misaligned = per_model.filter(~pl.col("instruction_aligned")).sort("model")
    joined = aligned.join(
        misaligned, on="model", suffix="_m"
    ).with_columns(
        (pl.col("mean_if") - pl.col("mean_if_m")).alias("gap")
    )
    result: dict[str, dict] = {}
    for row in joined.iter_rows(named=True):
        result[row["model"]] = {
            "aligned": round(row["mean_if"], 3),
            "misaligned": round(row["mean_if_m"], 3),
            "gap": round(row["gap"], 3),
        }
    return result


def prediction_stats(df: pl.DataFrame) -> dict[str, dict]:
    metrics = [
        ("score_prediction_accuracy", "accuracy"),
        ("score_prediction_instruction", "predicted_if"),
        ("score_instruction_following", "actual_if"),
    ]
    agg_exprs = []
    for src, alias in metrics:
        agg_exprs.extend([
            pl.col(src).mean().alias(alias),
            pl.col(src).std().alias(f"{alias}_sd"),
            pl.col(src).len().alias(f"{alias}_n"),
        ])
    summary = df.group_by("model").agg(agg_exprs)
    result: dict[str, dict] = {}
    for row in summary.iter_rows(named=True):
        entry: dict[str, float] = {}
        for _, alias in metrics:
            mean = row[alias]
            se = row[f"{alias}_sd"] / (row[f"{alias}_n"] ** 0.5)
            entry[alias] = round(mean, 3)
            entry[f"{alias}_ci_lo"] = round(mean - 1.96 * se, 3)
            entry[f"{alias}_ci_hi"] = round(mean + 1.96 * se, 3)
        result[row["model"]] = entry
    return result


def main() -> None:
    # --- Load datasets ---
    static_raw = normalize(
        pl.read_parquet(STATIC_PATH / "evals.parquet").filter(
            pl.col("instruction") == "instruction_no_hint"
        )
    )
    dynamic_raw = normalize(
        pl.read_parquet(DYNAMIC_PATH / "evals.parquet").filter(
            pl.col("instruction") == "instruction_no_hint"
        )
    )
    pred_raw = normalize(
        pl.read_parquet(STATIC_PATH / "evals_prediction.parquet").filter(
            pl.col("instruction") == "instruction_no_hint"
        )
    )

    static_df = static_raw.filter(
        pl.col("model").is_in(CORE_MODELS),
        ~pl.col("condition").is_in(TOKEN_CONDITIONS),
    )
    dynamic_df = dynamic_raw.filter(pl.col("model").is_in(CORE_MODELS))
    pred_df = pred_raw.filter(
        pl.col("model").is_in(CORE_MODELS),
        ~pl.col("condition").is_in(TOKEN_CONDITIONS),
    )

    # --- Per-model stats ---
    static_avg = avg_if_with_ci(static_df)
    dynamic_avg = avg_if_with_ci(dynamic_df)
    static_first_n = first_n_below(static_df)
    dynamic_first_n = first_n_below(dynamic_df)
    gap = alignment_gap(static_df)
    pred = prediction_stats(pred_df)

    # --- Combined results table ---
    results: list[dict] = []
    all_models = sorted(set(CORE_MODELS) & set(static_avg.keys()))
    for model in all_models:
        display = DISPLAY_NAMES.get(model, model)
        entry: dict = {
            "model": model,
            "display_name": display,
            "static_avg_if": static_avg.get(model, {}).get("mean", "—"),
            "static_avg_ci_lo": static_avg.get(model, {}).get("ci_lo", ""),
            "static_avg_ci_hi": static_avg.get(model, {}).get("ci_hi", ""),
            "static_first_n": static_first_n.get(model, ">"),
            "dynamic_avg_if": dynamic_avg.get(model, {}).get("mean", "—"),
            "dynamic_avg_ci_lo": dynamic_avg.get(model, {}).get("ci_lo", ""),
            "dynamic_avg_ci_hi": dynamic_avg.get(model, {}).get("ci_hi", ""),
            "dynamic_first_n": dynamic_first_n.get(model, ">"),
            "align_gap": gap.get(model, {}).get("gap", "—"),
            "align_aligned": gap.get(model, {}).get("aligned", "—"),
            "align_misaligned": gap.get(model, {}).get("misaligned", "—"),
        }
        if model in pred:
            entry.update({
                "pred_accuracy": pred[model]["accuracy"],
                "pred_accuracy_ci_lo": pred[model]["accuracy_ci_lo"],
                "pred_accuracy_ci_hi": pred[model]["accuracy_ci_hi"],
                "predicted_if": pred[model]["predicted_if"],
                "predicted_if_ci_lo": pred[model]["predicted_if_ci_lo"],
                "predicted_if_ci_hi": pred[model]["predicted_if_ci_hi"],
            })
        results.append(entry)

    # Sort by static avg IF descending
    results.sort(key=lambda x: x["static_avg_if"] if isinstance(x["static_avg_if"], float) else 0, reverse=True)

    # --- Reasoning table ---
    reasoning_models = ["gpt-5.2", "gpt-5.2-medium", "hermes-4-70b", "hermes-4-70b-reasoning"]
    static_r = static_raw.filter(pl.col("model").is_in(reasoning_models))
    dynamic_r = dynamic_raw.filter(pl.col("model").is_in(reasoning_models))
    static_r_avg = avg_if_with_ci(static_r)
    dynamic_r_avg = avg_if_with_ci(dynamic_r)
    static_r_first_n = first_n_below(static_r)
    dynamic_r_first_n = first_n_below(dynamic_r)

    reasoning_results: list[dict] = []
    for model in reasoning_models:
        if model not in static_r_avg:
            continue
        reasoning_results.append({
            "model": model,
            "display_name": DISPLAY_NAMES.get(model, model),
            "is_reasoning": model in ["gpt-5.2-medium", "hermes-4-70b-reasoning"],
            "static_avg_if": static_r_avg[model]["mean"],
            "static_avg_ci_lo": static_r_avg[model]["ci_lo"],
            "static_avg_ci_hi": static_r_avg[model]["ci_hi"],
            "static_first_n": static_r_first_n.get(model, ">"),
            "dynamic_avg_if": dynamic_r_avg.get(model, {}).get("mean", "—"),
            "dynamic_avg_ci_lo": dynamic_r_avg.get(model, {}).get("ci_lo", ""),
            "dynamic_avg_ci_hi": dynamic_r_avg.get(model, {}).get("ci_hi", ""),
            "dynamic_first_n": dynamic_r_first_n.get(model, ">"),
        })

    out_dir = ROOT / "outputs" / "paper_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results_table_data.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "reasoning_table_data.json", "w") as f:
        json.dump(reasoning_results, f, indent=2)

    print("=== Results table data ===")
    for row in results:
        print(
            f"{row['display_name']:<28} "
            f"fix={row['static_avg_if']:.3f} [{row.get('static_avg_ci_lo','')}, {row.get('static_avg_ci_hi','')}]  "
            f"N50={row['static_first_n']}  "
            f"task={row.get('dynamic_avg_if', '—')}  "
            f"gap={row['align_gap']}  "
            f"pred_acc={row.get('pred_accuracy', '—')}"
        )

    print("\n=== Reasoning table data ===")
    for row in reasoning_results:
        print(
            f"{'(R) ' if row['is_reasoning'] else '    '}{row['display_name']:<30} "
            f"fix={row['static_avg_if']:.3f}  N50={row['static_first_n']}  "
            f"task={row.get('dynamic_avg_if','—')}"
        )


if __name__ == "__main__":
    main()
