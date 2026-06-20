"""Rebuttal figures: per-model instruction-following under no-hint vs. hint.

Two overlapping-bar charts (one per condition family). For each model the no-hint
baseline is drawn as a wide muted bar and the hint result as a narrower coloured bar
on top, so the exposed sliver shows the increase (or, where the bar shrinks, the
regression). Saves PNGs to docs/rebuttals/figures/.

Run: uv run python scripts/plot_hint_effect.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plotting_utils import DISPLAY_NAMES  # noqa: E402

CORE = [
    "claude-opus-4.6", "claude-sonnet-4.6", "gemini-2.5-flash", "gemma-3-12b-it",
    "gemma-3-27b-it", "gpt-5.2", "hermes-4-70b", "kimi-k2", "llama-3.1-70b-instruct",
    "llama-3.3-70b-instruct", "olmo-3.1-32b-instruct", "qwen3-235b-a22b-instruct-2507",
    "qwen3-30b-a3b-instruct-2507",
]
FIXED = [
    "neutral", "value_aligned_helpful", "value_misaligned_helpful",
    "factual_aligned_earth", "factual_misaligned_earth",
]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "rebuttals" / "figures"

NO_HINT_COLOR = "#c7c7c7"
HINT_COLOR = "#2a9d8f"
UP_COLOR = "#2a9d8f"
DOWN_COLOR = "#e76f51"


def per_model(path: str, conds: list[str] | None = None) -> pl.DataFrame:
    df = pl.read_parquet(ROOT / path).filter(pl.col("model").is_in(CORE))
    if conds is not None:
        df = df.filter(pl.col("condition").is_in(conds))
    return df.group_by("model").agg(pl.col("score").mean().alias("if"))


def _draw_panel(ax, hint_path, base_path, conds, title):
    """Draw the per-model no-hint vs. hint overlapping-bar chart onto ``ax``.

    Returns the mean per-model delta.
    """
    h = per_model(hint_path, conds).rename({"if": "hint"})
    b = per_model(base_path, conds).rename({"if": "nohint"})
    m = (
        h.join(b, on="model")
        .with_columns((pl.col("hint") - pl.col("nohint")).alias("delta"))
        .sort("nohint")
    )
    models = m["model"].to_list()
    labels = [DISPLAY_NAMES.get(x, x) for x in models]
    nohint = m["nohint"].to_list()
    hint = m["hint"].to_list()
    delta = m["delta"].to_list()
    x = range(len(models))

    # wide muted no-hint bar behind, narrower hint bar in front
    ax.bar(x, nohint, width=0.78, color=NO_HINT_COLOR, label="No hint", zorder=2)
    ax.bar(x, hint, width=0.40, color=HINT_COLOR, label="Hint", zorder=3)

    # delta annotations above the taller of the two bars
    for xi, nh, ht, d in zip(x, nohint, hint, delta):
        top = max(nh, ht)
        if abs(d) < 0.005:
            col, txt = "#888888", "0.00"
        else:
            col, txt = (UP_COLOR if d > 0 else DOWN_COLOR), f"{d:+.2f}"
        ax.text(xi, top + 0.015, txt, ha="center", va="bottom",
                fontsize=8, color=col, fontweight="bold", zorder=4)

    mean_d = sum(delta) / len(delta)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{title}\nMean Δ = {mean_d:+.2f} IF (hint − no hint)",
                 fontsize=12, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", zorder=0)
    ax.set_axisbelow(True)
    return mean_d


def make_figure(hint_path, base_path, conds, title, fname):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    mean_d = _draw_panel(ax, hint_path, base_path, conds, title)
    ax.set_ylabel("Mean instruction-following rate", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  (mean delta {mean_d:+.3f})")


def make_combined(fname="hint_effect_combined.png"):
    """Single figure with fixed-output (left) and task-based (right) panels."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True)
    _draw_panel(
        axl, "outputs/viz/static-hint/evals.parquet",
        "outputs/viz/static/evals.parquet", FIXED,
        "Fixed-output conditions",
    )
    _draw_panel(
        axr, "outputs/viz/dynamic-hint/evals.parquet",
        "outputs/viz/dynamic/evals.parquet", None,
        "Task-based conditions",
    )
    axl.set_ylabel("Mean instruction-following rate", fontsize=11)
    axl.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / fname
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    make_figure(
        "outputs/viz/static-hint/evals.parquet", "outputs/viz/static/evals.parquet",
        FIXED, "Hint effect — fixed-output conditions", "hint_effect_fixed.png",
    )
    make_figure(
        "outputs/viz/dynamic-hint/evals.parquet", "outputs/viz/dynamic/evals.parquet",
        None, "Hint effect — task-based conditions", "hint_effect_task.png",
    )
    make_combined()


if __name__ == "__main__":
    main()
