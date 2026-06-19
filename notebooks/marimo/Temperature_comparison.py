# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "polars",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import polars as pl
    from pathlib import Path

    alt.data_transformers.disable_max_rows()
    return Path, alt, mo, pl


@app.cell
def _(mo):
    mo.md(r"""
    # Temperature Comparison Analysis: T0 vs T1

    Comparing model behavior across two temperature settings:
    - **T0**: 35 samples per configuration, `instruction_no_hint` only
    - **T1**: 15 samples per configuration, both `instruction_no_hint` and `instruction_hint`

    Analysis focuses on the intersection of models and conditions across both datasets.
    """)
    return


@app.cell
def _(Path, pl):
    _root = Path(__file__).resolve().parent.parent.parent
    _t0_path = _root / "outputs" / "viz" / "static"
    _t1_path = _root / "outputs" / "viz" / "static-T1"

    t0_raw = pl.read_parquet(_t0_path / "evals.parquet")
    t1_raw = pl.read_parquet(_t1_path / "evals.parquet")

    t0_raw = t0_raw.with_columns(pl.lit("T0").alias("temperature"))
    t1_raw = t1_raw.with_columns(pl.lit("T1").alias("temperature"))

    t0_aligned = pl.read_parquet(_t0_path / "_aligned.parquet").with_columns(
        pl.lit("T0").alias("temperature")
    )
    t1_aligned = pl.read_parquet(_t1_path / "_aligned.parquet").with_columns(
        pl.lit("T1").alias("temperature")
    )

    t0_misaligned = pl.read_parquet(_t0_path / "_misaligned.parquet").with_columns(
        pl.lit("T0").alias("temperature")
    )
    t1_misaligned = pl.read_parquet(_t1_path / "_misaligned.parquet").with_columns(
        pl.lit("T1").alias("temperature")
    )
    return t0_raw, t1_raw


@app.cell
def _(mo, t0_raw, t1_raw):
    MODEL_MATCHES = {
        "claude-sonnet-4.6": "claude-sonnet-4.6",
        "gemma-3-12b-it": "gemma-3-12b-it",
        "gemma-3-27b-it": "gemma-3-27b-it",
        "gpt-5.2": "gpt-5.2",
        "llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
        "gemini-2.5-flash": "gemini-2.5-flash",
    }

    T0_TO_COMMON = {k: k for k in MODEL_MATCHES.keys()}
    T1_TO_COMMON = {v: k for k, v in MODEL_MATCHES.items()}

    t0_models = set(t0_raw["model"].unique().to_list())
    t1_models = set(t1_raw["model"].unique().to_list())

    common_conditions = list(
        set(t0_raw["condition"].unique().to_list())
        & set(t1_raw["condition"].unique().to_list())
    )
    common_conditions = sorted(common_conditions)

    mo.md(f"""
    **Model Mapping ({len(MODEL_MATCHES)} comparable models):**
    {chr(10).join(f"- T0: {k} ↔ T1: {v}" for k, v in MODEL_MATCHES.items())}

    **Common Conditions ({len(common_conditions)}):** {", ".join(common_conditions)}
    """)
    return MODEL_MATCHES, T1_TO_COMMON, common_conditions


@app.cell
def _(mo):
    mo.md("""
    ## Data Filtering & Controls
    """)
    return


@app.cell
def _(MODEL_MATCHES, common_conditions, mo):
    instruction_select = mo.ui.dropdown(
        options=["instruction_no_hint", "instruction_hint"],
        value="instruction_no_hint",
        label="Instruction Setting (T1)",
    )

    condition_select = mo.ui.multiselect(
        options=common_conditions,
        value=common_conditions,
        label="Conditions to include",
    )

    model_select = mo.ui.multiselect(
        options=list(MODEL_MATCHES.keys()),
        value=list(MODEL_MATCHES.keys()),
        label="Models to include",
    )

    n_turns_range = mo.ui.range_slider(
        start=1,
        stop=50,
        value=[1, 50],
        step=1,
        label="N-turns range",
    )

    mo.hstack(
        [instruction_select, condition_select, model_select, n_turns_range],
        widths="equal",
    )
    return condition_select, instruction_select, model_select, n_turns_range


@app.cell
def _(
    MODEL_MATCHES,
    T1_TO_COMMON,
    condition_select,
    instruction_select,
    model_select,
    n_turns_range,
    pl,
    t0_raw,
    t1_raw,
):
    _selected_models = model_select.value
    _selected_conditions = condition_select.value
    _n_lo, _n_hi = n_turns_range.value

    t0_filtered = (
        t0_raw.filter(
            pl.col("model").is_in(_selected_models),
            pl.col("condition").is_in(_selected_conditions),
            pl.col("instruction") == "instruction_no_hint",
        )
        .with_columns(
            pl.col("n_turns").cast(pl.Int64).alias("n_turns_int"),
            pl.col("model").replace(MODEL_MATCHES).alias("model_common"),
        )
        .filter((pl.col("n_turns_int") >= _n_lo) & (pl.col("n_turns_int") <= _n_hi))
    )

    t1_filtered = (
        t1_raw.filter(
            pl.col("model").is_in(MODEL_MATCHES.values()),
            pl.col("condition").is_in(_selected_conditions),
            pl.col("instruction") == instruction_select.value,
        )
        .with_columns(
            pl.col("n_turns").cast(pl.Int64).alias("n_turns_int"),
            pl.col("model").replace(T1_TO_COMMON).alias("model_common"),
        )
        .filter((pl.col("n_turns_int") >= _n_lo) & (pl.col("n_turns_int") <= _n_hi))
    )

    combined = pl.concat([t0_filtered, t1_filtered])
    return combined, t0_filtered, t1_filtered


@app.cell
def _(mo):
    mo.md("""
    ## Chart 1: Average IF Rate by Model (T0 vs T1)
    """)
    return


@app.cell
def _(alt, combined, pl):
    _avg_by_model_temp = (
        combined.group_by(["model_common", "temperature"])
        .agg(pl.col("score").mean().alias("avg_if_rate"))
        .sort(["model_common", "temperature"])
    )

    chart1 = (
        alt.Chart(_avg_by_model_temp.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X(
                "temperature:N", title=None, axis=alt.Axis(labels=False, ticks=False)
            ),
            y=alt.Y(
                "avg_if_rate:Q", title="Avg IF Rate", scale=alt.Scale(domain=[0, 1])
            ),
            color=alt.Color(
                "temperature:N",
                scale=alt.Scale(domain=["T0", "T1"], range=["#4c78a8", "#f58518"]),
                legend=alt.Legend(title="Temperature"),
            ),
            column=alt.Column(
                "model_common:N", title="Model", header=alt.Header(labelAngle=-45)
            ),
            tooltip=[
                "model_common",
                "temperature",
                alt.Tooltip("avg_if_rate:Q", format=".3f"),
            ],
        )
        .properties(width=60, height=300)
    )
    chart1
    return


@app.cell
def _(mo):
    mo.md("""
    ## Chart 2: IF Rate Over N Turns (T0 vs T1)
    """)
    return


@app.cell
def _(alt, combined, model_select, pl):
    _avg_by_n = (
        combined.group_by(["model_common", "temperature", "n_turns_int"])
        .agg(
            pl.col("score").mean().alias("if_rate"),
            pl.col("score_stderr").mean().alias("stderr"),
        )
        .sort(["model_common", "temperature", "n_turns_int"])
    )

    _avg_by_n_pd = _avg_by_n.to_pandas()

    _selected = (
        model_select.value[:4] if len(model_select.value) > 4 else model_select.value
    )
    _avg_by_n_pd = _avg_by_n_pd[_avg_by_n_pd["model_common"].isin(_selected)]

    line = (
        alt.Chart(_avg_by_n_pd)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_turns_int:Q", title="N Turns"),
            y=alt.Y("if_rate:Q", title="IF Rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "temperature:N",
                scale=alt.Scale(domain=["T0", "T1"], range=["#4c78a8", "#f58518"]),
            ),
            strokeDash=alt.StrokeDash(
                "model_common:N", legend=alt.Legend(title="Model")
            ),
            tooltip=[
                "model_common",
                "temperature",
                "n_turns_int",
                alt.Tooltip("if_rate:Q", format=".3f"),
            ],
        )
    )

    band = (
        alt.Chart(_avg_by_n_pd)
        .mark_errorband(extent="stderr")
        .encode(
            x="n_turns_int:Q",
            y=alt.Y("if_rate:Q", title="IF Rate"),
            color="temperature:N",
        )
    )

    chart2 = (
        (line)
        .facet(
            facet=alt.Facet("model_common:N", title="Model"),
            columns=2,
        )
        .properties(title="IF Rate Over N Turns by Temperature")
        .resolve_scale(y="shared")
    )
    chart2
    return


@app.cell
def _(mo):
    mo.md("""
    ## Chart 3: IF Rate Heatmap by Model x Condition
    """)
    return


@app.cell
def _(alt, combined, pl):
    _heatmap_data = combined.group_by(["model_common", "condition", "temperature"]).agg(
        pl.col("score").mean().alias("if_rate")
    )

    chart3 = (
        alt.Chart(_heatmap_data.to_pandas())
        .mark_rect()
        .encode(
            x=alt.X("condition:N", title="Condition", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("model_common:N", title="Model"),
            color=alt.Color(
                "if_rate:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                legend=alt.Legend(title="IF Rate"),
            ),
            tooltip=[
                "model_common",
                "condition",
                "temperature",
                alt.Tooltip("if_rate:Q", format=".3f"),
            ],
        )
        .facet(facet=alt.Facet("temperature:N", title="Temperature"), columns=2)
        .properties(title="IF Rate by Model and Condition", width=400, height=250)
    )
    chart3
    return


@app.cell
def _(mo):
    mo.md("""
    ## Chart 4: T0 vs T1 Scatter (Per Model-Condition Pair)
    """)
    return


@app.cell
def _(alt, pl, t0_filtered, t1_filtered):
    _t0_avg = t0_filtered.group_by(["model_common", "condition"]).agg(
        pl.col("score").mean().alias("if_rate_t0")
    )

    _t1_avg = t1_filtered.group_by(["model_common", "condition"]).agg(
        pl.col("score").mean().alias("if_rate_t1")
    )

    _scatter_df = _t0_avg.join(_t1_avg, on=["model_common", "condition"], how="inner")

    _scatter_pd = _scatter_df.to_pandas()

    points = (
        alt.Chart(_scatter_pd)
        .mark_circle(size=80)
        .encode(
            x=alt.X("if_rate_t0:Q", title="T0 IF Rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("if_rate_t1:Q", title="T1 IF Rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_common:N", legend=alt.Legend(title="Model")),
            shape=alt.Shape("condition:N", legend=alt.Legend(title="Condition")),
            tooltip=[
                "model_common",
                "condition",
                alt.Tooltip("if_rate_t0:Q", format=".3f"),
                alt.Tooltip("if_rate_t1:Q", format=".3f"),
            ],
        )
    )

    identity = (
        alt.Chart(_scatter_pd)
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x=alt.X("if_rate_t0:Q"), y=alt.Y("if_rate_t0:Q"))
    )

    chart4 = (points + identity).properties(
        title="T0 vs T1 IF Rate (identity line = no change)",
        width=500,
        height=500,
    )
    chart4
    return


@app.cell
def _(mo):
    mo.md("""
    ## Chart 5: Distribution of IF Rates (Box Plot)
    """)
    return


@app.cell
def _(alt, combined, pl):
    _box_data = combined.group_by(["model_common", "condition", "temperature"]).agg(
        pl.col("score").mean().alias("if_rate")
    )

    chart5 = (
        alt.Chart(_box_data.to_pandas())
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("temperature:N", title="Temperature"),
            y=alt.Y("if_rate:Q", title="IF Rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "temperature:N",
                scale=alt.Scale(domain=["T0", "T1"], range=["#4c78a8", "#f58518"]),
            ),
        )
        .facet(facet=alt.Facet("model_common:N", title="Model"), columns=3)
        .properties(
            title="Distribution of IF Rates by Temperature", width=80, height=150
        )
    )
    chart5
    return


@app.cell
def _(mo):
    mo.md("""
    ## Chart 6: Delta (T1 - T0) by Model and Condition
    """)
    return


@app.cell
def _(alt, pl, t0_filtered, t1_filtered):
    _t0_delta = t0_filtered.group_by(["model_common", "condition"]).agg(
        pl.col("score").mean().alias("t0_rate")
    )
    _t1_delta = t1_filtered.group_by(["model_common", "condition"]).agg(
        pl.col("score").mean().alias("t1_rate")
    )

    _delta_df = _t0_delta.join(_t1_delta, on=["model_common", "condition"], how="inner")
    _delta_df = _delta_df.with_columns(
        (pl.col("t1_rate") - pl.col("t0_rate")).alias("delta")
    )

    chart6 = (
        alt.Chart(_delta_df.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X(
                "delta:Q", title="Delta (T1 - T0)", scale=alt.Scale(domain=[-0.3, 0.3])
            ),
            y=alt.Y("model_common:N", title="Model"),
            color=alt.Color(
                "delta:Q",
                scale=alt.Scale(scheme="redblue", domain=[-0.2, 0.2]),
                legend=alt.Legend(title="Delta"),
            ),
            row=alt.Row(
                "condition:N", title="Condition", header=alt.Header(labelAngle=0)
            ),
            tooltip=[
                "model_common",
                "condition",
                alt.Tooltip("t0_rate:Q", format=".3f"),
                alt.Tooltip("t1_rate:Q", format=".3f"),
                alt.Tooltip("delta:Q", format=".3f"),
            ],
        )
        .properties(title="Temperature Impact: T1 - T0 Delta", width=400, height=50)
    )
    chart6
    return


@app.cell
def _(mo):
    mo.md("""
    ## Chart 7: Summary Statistics Table
    """)
    return


@app.cell
def _(combined, pl):
    summary_table = (
        combined.group_by(["model_common", "temperature"])
        .agg(
            pl.col("score").mean().alias("mean_if_rate"),
            pl.col("score").std().alias("std_if_rate"),
            pl.col("score").min().alias("min_if_rate"),
            pl.col("score").max().alias("max_if_rate"),
            pl.len().alias("n_samples"),
        )
        .sort(["model_common", "temperature"])
    )
    summary_table
    return


@app.cell
def _(mo):
    mo.md("""
    ## Chart 8: Instruction Hint Effect (T1 Only)
    """)
    return


@app.cell
def _(MODEL_MATCHES, T1_TO_COMMON, alt, condition_select, mo, pl, t1_raw):
    hint_instruction_select = mo.ui.dropdown(
        options=["instruction_no_hint", "instruction_hint"],
        value="instruction_hint",
        label="Compare to",
    )
    hint_instruction_select

    _t1_no_hint = (
        t1_raw.filter(
            pl.col("model").is_in(MODEL_MATCHES.values()),
            pl.col("condition").is_in(condition_select.value),
            pl.col("instruction") == "instruction_no_hint",
        )
        .group_by(["model", "condition"])
        .agg(pl.col("score").mean().alias("no_hint_rate"))
        .with_columns(pl.col("model").replace(T1_TO_COMMON).alias("model_common"))
    )

    _t1_hint = (
        t1_raw.filter(
            pl.col("model").is_in(MODEL_MATCHES.values()),
            pl.col("condition").is_in(condition_select.value),
            pl.col("instruction") == "instruction_hint",
        )
        .group_by(["model", "condition"])
        .agg(pl.col("score").mean().alias("hint_rate"))
        .with_columns(pl.col("model").replace(T1_TO_COMMON).alias("model_common"))
    )

    _hint_compare = _t1_no_hint.join(
        _t1_hint, on=["model_common", "condition"], how="inner"
    )
    _hint_compare = _hint_compare.with_columns(
        (pl.col("hint_rate") - pl.col("no_hint_rate")).alias("hint_effect")
    )

    chart8 = (
        alt.Chart(_hint_compare.to_pandas())
        .mark_bar()
        .encode(
            x=alt.X(
                "hint_effect:Q",
                title="Hint Effect (hint - no_hint)",
                scale=alt.Scale(domain=[-0.5, 0.5]),
            ),
            y=alt.Y("model_common:N", title="Model"),
            color=alt.Color(
                "hint_effect:Q",
                scale=alt.Scale(scheme="redblue", domain=[-0.3, 0.3], reverse=True),
            ),
            row=alt.Row(
                "condition:N", title="Condition", header=alt.Header(labelAngle=0)
            ),
            tooltip=[
                "model_common",
                "condition",
                alt.Tooltip("no_hint_rate:Q", format=".3f"),
                alt.Tooltip("hint_rate:Q", format=".3f"),
                alt.Tooltip("hint_effect:Q", format=".3f"),
            ],
        )
        .properties(title="Effect of Instruction Hint (T1 only)", width=400, height=50)
    )
    chart8
    return


if __name__ == "__main__":
    app.run()
