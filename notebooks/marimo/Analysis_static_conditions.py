import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import json
    from pathlib import Path
    from pyobsplot import Plot, js
    from src.plotting_utils import (
        BENCHMARKS,
        COLOR_SCHEME,
        DISPLAY_NAMES,
        nudge_labels,
        prep_benchmark_data,
        make_scatter_chart,
        make_radar_chart,
        LIKERT_OFFSET_JS,
        CATEGORY_ORDER,
        CATEGORY_COLORS,
    )

    return (
        CATEGORY_COLORS,
        CATEGORY_ORDER,
        COLOR_SCHEME,
        DISPLAY_NAMES,
        Path,
        Plot,
        alt,
        js,
        json,
        make_radar_chart,
        make_scatter_chart,
        mo,
        nudge_labels,
        pd,
        prep_benchmark_data,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analysis of induction vs prediction experiments - Fixed-output conditions

    Your job is to implement all plots in this notebook. Unless specified otherwise, the plotting library should be Altair. You might be provided with links to specific plot examples, if so, use that implementation. Familiarize yourself with the experiment first. Code cells marked as todo are where the code is going to go. Add descriptions to markdown cells where appropriate.

    Useful skill/MCPs: marimo, observable-plot, napkin (for info about data wrangling and plotting)

    Acronyms:
    - IF: instruction-following
    - PF: pattern-following (cases where the models falls prey to induction)
    - IT: instruction template
    - N: number of turns, goes from 1 to 50 but it does not cover all values. Be careful with the data format.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading

    - Load data from outputs\viz\static. Different files will be needed for different plots.
    - Load model capability scores from data\model_capability_scores.json. the key score is intelligence_index
    """)
    return


@app.cell
def _(Path, json, pd):
    _root = Path(__file__).resolve().parent.parent.parent
    _fixed_output = _root / "outputs" / "viz" / "static"

    # Token-pattern conditions excluded: they produce varying responses (random set members),
    # not a single fixed token, so they don't belong in the fixed-output analysis.
    _exclude_conditions = ["token_states_countries", "token_countries_states"]
    # ["value_aligned_helpful", "value_misaligned_helpful"]

    evals_all = pd.read_parquet(_fixed_output / "evals.parquet")
    evals_all = evals_all[~evals_all["condition"].isin(_exclude_conditions)].copy()
    evals = evals_all[evals_all["instruction"] == "instruction_no_hint"].copy()
    evals["n_turns_int"] = evals["n_turns"].astype(int)

    combined_errors_all = pd.read_parquet(
        _fixed_output / "_combined_errors.parquet"
    )
    combined_errors_all = combined_errors_all[
        ~combined_errors_all["condition"].isin(_exclude_conditions)
    ].copy()
    combined_errors = combined_errors_all[
        combined_errors_all["instruction"] == "instruction_no_hint"
    ].copy()
    combined_errors["n_turns_int"] = combined_errors["n_turns"].astype(int)

    evals_prediction_all = pd.read_parquet(
        _fixed_output / "evals_prediction.parquet"
    )
    evals_prediction_all = evals_prediction_all[
        ~evals_prediction_all["condition"].isin(_exclude_conditions)
    ].copy()
    evals_prediction = evals_prediction_all[
        evals_prediction_all["instruction"] == "instruction_no_hint"
    ].copy()
    evals_prediction["n_turns_int"] = evals_prediction["n_turns"].astype(int)

    with open(_root / "data" / "model_capability_scores.json") as _f:
        _caps_raw = json.load(_f)

    reasoning_models = {
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
        "gpt-5.2-medium",
        "hermes-4-70b-reasoning",
        "kimi-k2.5",
        "minimax-m2.5",
    }


    def _get_caps(model_name, caps_data):
        mode = "true" if model_name in reasoning_models else "false"
        scores = caps_data.get(mode, {}) or {}
        return {
            "model": model_name,
            "gpqa": scores.get("gpqa"),
            "ifbench": scores.get("ifbench"),
        }


    caps_df = pd.DataFrame([_get_caps(k, v) for k, v in _caps_raw.items()])
    caps_df = caps_df.dropna(how="all", subset=["gpqa", "ifbench"])

    n_values_sorted = sorted(evals["n_turns"].unique(), key=int)
    all_models = sorted(evals_all["model"].unique().tolist())
    return (
        all_models,
        caps_df,
        combined_errors,
        evals,
        evals_all,
        evals_prediction,
        n_values_sorted,
        reasoning_models,
    )


@app.cell
def _(all_models, mo):
    model_exclusion = mo.ui.multiselect(
        options=all_models,
        value=[
            "hermes-4-70b-reasoning",
            "gpt-5.2-medium",
            "hermes-4-405b",
            #  "llama-3.1-70b-instruct",
        ],
        label="Models to exclude",
    )
    model_exclusion
    return (model_exclusion,)


@app.cell
def _(
    DISPLAY_NAMES,
    caps_df,
    combined_errors,
    evals,
    evals_all,
    evals_prediction,
    model_exclusion,
):
    _excluded = model_exclusion.value
    _rename = lambda df: df.assign(
        model=df["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
    )
    evals_filtered = _rename(
        evals[~evals["model"].isin(_excluded)].copy()
        if _excluded
        else evals.copy()
    )
    evals_all_filtered = _rename(
        evals_all[~evals_all["model"].isin(_excluded)].copy()
        if _excluded
        else evals_all.copy()
    )
    combined_errors_filtered = _rename(
        combined_errors[~combined_errors["model"].isin(_excluded)].copy()
        if _excluded
        else combined_errors.copy()
    )
    evals_prediction_filtered = _rename(
        evals_prediction[~evals_prediction["model"].isin(_excluded)].copy()
        if _excluded
        else evals_prediction.copy()
    )
    caps_df_filtered = _rename(
        caps_df[~caps_df["model"].isin(_excluded)].copy()
        if _excluded
        else caps_df.copy()
    )
    return (
        caps_df_filtered,
        combined_errors_filtered,
        evals_all_filtered,
        evals_filtered,
        evals_prediction_filtered,
    )


@app.cell
def _(Path):
    plots_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "plots"
        / "static"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    return (plots_dir,)


@app.cell
def _(combined_errors):
    combined_errors.model.unique()
    return


@app.cell
def _(evals):
    evals.model.unique()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavioral analysis
    - Plot A1 - stacked bar plot (Altair). For each model on the x axis, show three stacks: number of N values with IF >= THRESHOLD, number of N values with PF >= THRESHOLD, remaining N values in the dataset. Average across conditions.THRESHOLD should be set interactively in the notebook, with a default of 0.8.
    - Plot A2 - avg IF rate vs capability. Compute average IF rate and make a scatterplot of (IF rate, intelligence_index). Color is model.
    - Plot A3 - horizontal bar chart with avg IF rate + error bars to show uncertainty. Error bars come from the stderr metric in the dataframe. Faceted by condition (col size 3 or 4), y shows models. x axis should be fixed from 0 to 1. Add dropdown to select which instruction_template to show
    - Plot A4: vertical colored stripes of IF rate for increasing N: similar to https://observablehq.com/@observablehq/plot-stacked-unit-chart. But here the data is sorted by N, not by the value of the data itself. The quantity to plot is again the mean IF rate across condition per model (models on the y axis). Each stacked unit is the IF rate for one N value per increasing N, with color showing the average IF rate.
    - Plot A5: Plot the first N value (when N is sorted) where IF drops to THRESHOLD or less, against each of the capability values *3 scatterplots). THRESHOLD is set through another slider, not the same one as above.
    """)
    return


@app.cell
def _(mo):
    threshold_slider = mo.ui.slider(
        start=0.5, stop=1.0, value=0.7, step=0.05, label="IF/PF Threshold"
    )
    threshold_slider
    return (threshold_slider,)


@app.cell
def _(
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    alt,
    evals_filtered,
    n_values_sorted,
    pd,
    plots_dir,
    threshold_slider,
):
    # Plot A1: Horizontal stacked bar — N values classified as IF/PF/Mixed
    _threshold = threshold_slider.value
    _avg = (
        evals_filtered.groupby(["model", "n_turns"])["score"].mean().reset_index()
    )

    _unit_records = []
    for _model in _avg["model"].unique():
        _model_df = _avg[_avg["model"] == _model].sort_values(
            "n_turns", key=lambda s: s.astype(int)
        )
        for _, _r in _model_df.iterrows():
            _score = _r["score"]
            if _score >= _threshold:
                _cat = "IF-dominant"
            elif (1 - _score) >= _threshold:
                _cat = "PF-dominant"
            else:
                _cat = "Mixed"
            _unit_records.append({"model": _r["model"], "category": _cat})

    _n_total = len(n_values_sorted)

    _counts = (
        pd.DataFrame(_unit_records)
        .groupby(["model", "category"])
        .size()
        .reset_index(name="count")
    )
    # Ensure all categories present for all models
    _all_combos = pd.MultiIndex.from_product(
        [_counts["model"].unique(), CATEGORY_ORDER], names=["model", "category"]
    ).to_frame(index=False)
    _counts = _all_combos.merge(
        _counts, on=["model", "category"], how="left"
    ).fillna({"count": 0})
    _counts["count"] = _counts["count"].astype(int)

    # Sort models by IF-dominant count ascending (most IF-dominant on top)
    _if_counts = _counts[_counts["category"] == "IF-dominant"].set_index("model")[
        "count"
    ]
    _model_order = _if_counts.sort_values(ascending=True).index.tolist()

    # Stack order: PF-dominant left, Mixed middle, IF-dominant right
    _stack_order = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    _counts["stack_order"] = _counts["category"].map(_stack_order)

    _a1_chart = (
        alt.Chart(_counts)
        .mark_bar(cornerRadius=4)
        .encode(
            y=alt.Y("model:N", sort=_model_order, title=None),
            x=alt.X("count:Q", stack=True, title=f"N values (out of {_n_total})"),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=CATEGORY_ORDER, range=CATEGORY_COLORS),
                sort=CATEGORY_ORDER,
                title="Category",
            ),
            order=alt.Order("stack_order:Q"),
            tooltip=["model:N", "category:N", "count:Q"],
        )
        .properties(
            title=f"Model behavior by turn count (threshold={_threshold})",
            width=500,
            height=350,
        )
    )
    _a1_chart.save(
        str(plots_dir / "a1_model_behavior_stacked.png"), scale_factor=2
    )
    _a1_chart
    return


@app.cell
def _(evals_filtered, mo):
    _a1d_conditions = sorted(evals_filtered["condition"].unique().tolist())

    a1_detail_condition = mo.ui.dropdown(
        options=_a1d_conditions, value=_a1d_conditions[0], label="Condition"
    )
    a1_detail_condition
    return (a1_detail_condition,)


@app.cell
def _(
    CATEGORY_COLORS,
    CATEGORY_ORDER,
    a1_detail_condition,
    alt,
    evals_filtered,
    n_values_sorted,
    pd,
    threshold_slider,
):
    # A1 detail: same as A1 but for a single condition across all models (no save — interactive only)
    _threshold_d = threshold_slider.value
    _detail_df = evals_filtered[
        evals_filtered["condition"] == a1_detail_condition.value
    ]
    _avg_d = _detail_df.groupby(["model", "n_turns"])["score"].mean().reset_index()

    _detail_records = []
    for _model_d in _avg_d["model"].unique():
        _model_df_d = _avg_d[_avg_d["model"] == _model_d].sort_values(
            "n_turns", key=lambda s: s.astype(int)
        )
        for _, _row_d in _model_df_d.iterrows():
            _s = _row_d["score"]
            if _s >= _threshold_d:
                _cat_d = "IF-dominant"
            elif (1 - _s) >= _threshold_d:
                _cat_d = "PF-dominant"
            else:
                _cat_d = "Mixed"
            _detail_records.append({"model": _model_d, "category": _cat_d})

    _n_total_d = len(n_values_sorted)
    _counts_d = (
        pd.DataFrame(_detail_records)
        .groupby(["model", "category"])
        .size()
        .reset_index(name="count")
    )
    _all_combos_d = pd.MultiIndex.from_product(
        [_avg_d["model"].unique(), CATEGORY_ORDER], names=["model", "category"]
    ).to_frame(index=False)
    _counts_d = _all_combos_d.merge(
        _counts_d, on=["model", "category"], how="left"
    ).fillna({"count": 0})
    _counts_d["count"] = _counts_d["count"].astype(int)
    _stack_order_d = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    _counts_d["stack_order"] = _counts_d["category"].map(_stack_order_d)

    _if_counts_d = _counts_d[_counts_d["category"] == "IF-dominant"].set_index(
        "model"
    )["count"]
    _model_order_d = _if_counts_d.sort_values(ascending=True).index.tolist()

    (
        alt.Chart(_counts_d)
        .mark_bar(cornerRadius=4)
        .encode(
            y=alt.Y("model:N", sort=_model_order_d, title=None),
            x=alt.X(
                "count:Q",
                stack=True,
                title=f"Fraction of turns",
            ),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=CATEGORY_ORDER, range=CATEGORY_COLORS),
                sort=CATEGORY_ORDER,
                title="Category",
            ),
            order=alt.Order("stack_order:Q"),
            tooltip=["model:N", "category:N", "count:Q"],
        )
        .properties(
            title=f"Results for {a1_detail_condition.value} condition (threshold={_threshold_d})",
            width=500,
            height=350,
        )
    )
    return


@app.cell
def _(
    caps_df_filtered,
    evals_filtered,
    make_scatter_chart,
    plots_dir,
    prep_benchmark_data,
):
    # Plot A2: Scatter — avg IF rate vs capability (faceted by benchmark)
    _avg_if = (
        evals_filtered.groupby("model")["score"]
        .mean()
        .reset_index(name="avg_if_rate")
    )
    _a2_df = prep_benchmark_data(
        _avg_if, "avg_if_rate", caps_df_filtered, lambda *a: a[0]
    )
    _a2_chart = make_scatter_chart(
        _a2_df, "avg_if_rate", "Avg IF Rate", "Avg IF Rate vs Model Capability"
    )
    _a2_chart.save(str(plots_dir / "a2_if_rate_vs_capability.png"), scale_factor=2)
    _a2_chart
    return


@app.cell
def _(mo):
    instruction_dropdown = mo.ui.dropdown(
        options=["instruction_no_hint", "instruction_hint"],
        value="instruction_no_hint",
        label="Instruction template",
    )
    instruction_dropdown
    return (instruction_dropdown,)


@app.cell
def _(COLOR_SCHEME, alt, evals_all_filtered, instruction_dropdown, plots_dir):
    # Plot A3: Horizontal bar + error bars, faceted by condition
    _sel = instruction_dropdown.value
    _a3_data = evals_all_filtered[evals_all_filtered["instruction"] == _sel].copy()
    _a3_agg = (
        _a3_data.groupby(["model", "condition"])
        .agg(mean_score=("score", "mean"), mean_stderr=("score_stderr", "mean"))
        .reset_index()
    )
    _a3_agg["ci_lo"] = (_a3_agg["mean_score"] - _a3_agg["mean_stderr"]).clip(
        lower=0
    )
    _a3_agg["ci_hi"] = (_a3_agg["mean_score"] + _a3_agg["mean_stderr"]).clip(
        upper=1
    )

    _bars = (
        alt.Chart(_a3_agg)
        .mark_bar()
        .encode(
            x=alt.X(
                "mean_score:Q", title="IF Rate", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.Color(
                "model:N", scale=alt.Scale(scheme=COLOR_SCHEME), legend=None
            ),
            tooltip=["model", "condition", "mean_score:Q", "mean_stderr:Q"],
        )
    )
    _error = (
        alt.Chart(_a3_agg)
        .mark_errorbar()
        .encode(
            x=alt.X("ci_lo:Q", title=""),
            x2="ci_hi:Q",
            y=alt.Y("model:N", sort="-x"),
        )
    )
    _a3_chart = (
        (_bars + _error)
        .facet(facet=alt.Facet("condition:N", title="Condition"), columns=2)
        .resolve_scale(y="independent")
        .properties(title=f"IF Rate by Model and Condition")
    )
    _a3_chart.save(str(plots_dir / "a3_if_rate_by_condition.png"), scale_factor=2)
    _a3_chart
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Per-condition IF rate (averaged across models and turns)

    Each bar shows the grand mean IF rate for one condition, averaged over all included models
    and N values. Error bars show ±1 SE, accounting for two sources of variance:

    - **Between-cell** (across models & turns): variance of the cell means around the grand mean
    - **Within-cell** (resampling): each (model × N) cell has its own `score_stderr`

    SE = sqrt((Var_between + mean(σ²_within)) / K), where K = number of (model × N) cells.
    """)
    return


@app.cell
def _(alt, evals_all_filtered, instruction_dropdown, pd, plots_dir):
    # Plot A6: Per-condition IF rate averaged across models and N turns
    # SE combines between-cell variance and mean within-cell resampling variance
    _sel_a6 = instruction_dropdown.value
    _a6_data = evals_all_filtered[
        evals_all_filtered["instruction"] == _sel_a6
    ].copy()
    _a6_records = []
    for _cond, _grp in _a6_data.groupby("condition"):
        _cell_means = _grp["score"]
        _cell_ses = _grp["score_stderr"]
        _k = len(_cell_means)
        _mean_if = float(_cell_means.mean())
        _between_var = float(_cell_means.var(ddof=1)) if _k > 1 else 0.0
        _within_var = float((_cell_ses**2).mean())
        _se = ((_between_var + _within_var) / _k) ** 0.5
        _a6_records.append(
            {
                "condition": _cond,
                "mean_if": _mean_if,
                "se": _se,
                "k": _k,
                "ci_lo": max(0.0, _mean_if - _se),
                "ci_hi": min(1.0, _mean_if + _se),
            }
        )
    _a6_df = pd.DataFrame(_a6_records)
    _a6_bars = (
        alt.Chart(_a6_df)
        .mark_bar()
        .encode(
            x=alt.X("mean_if:Q", title="IF Rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("condition:N", sort="-x", title=None),
            tooltip=[
                "condition",
                alt.Tooltip("mean_if:Q", format=".3f", title="Mean IF Rate"),
                alt.Tooltip("se:Q", format=".4f", title="±SE"),
                alt.Tooltip("k:Q", title="# cells (model×N)"),
            ],
        )
    )
    _a6_err = (
        alt.Chart(_a6_df)
        .mark_errorbar(color="black", ticks=True)
        .encode(
            x=alt.X("ci_lo:Q", title=""),
            x2="ci_hi:Q",
            y=alt.Y("condition:N", sort="-x"),
        )
    )
    _a6_chart = (_a6_bars + _a6_err).properties(
        width=300,
        height=alt.Step(42),
        title=f"IF Rate per Condition — average over models & N",
    )
    _a6_chart.save(str(plots_dir / "a6_per_condition_if_rate.png"), scale_factor=2)
    _a6_chart
    return


@app.cell
def _(alt, evals_filtered, plots_dir):
    # Altair equivalent of A4: heatmap of IF rate — model × N turns
    _hm = (
        evals_filtered.groupby(["model", "n_turns", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="if_rate")
        .sort_values("n_turns_int")
    )
    _n_order = sorted(_hm["n_turns"].unique(), key=int)
    _rects = (
        alt.Chart(_hm)
        .mark_rect()
        .encode(
            x=alt.X("n_turns:O", sort=_n_order, title="N turns"),
            y=alt.Y(
                "model:N",
                sort=alt.EncodingSortField(
                    "if_rate", op="mean", order="descending"
                ),
                title=None,
            ),
            color=alt.Color(
                "if_rate:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                title="IF Rate",
                legend=None,
            ),
            tooltip=[
                "model",
                "n_turns",
                alt.Tooltip("if_rate:Q", format=".2f", title="IF Rate"),
            ],
        )
    )
    _texts = (
        alt.Chart(_hm)
        .mark_text(fontSize=16)
        .encode(
            x=alt.X("n_turns:O", sort=_n_order),
            y=alt.Y(
                "model:N",
                sort=alt.EncodingSortField(
                    "if_rate", op="mean", order="descending"
                ),
            ),
            text=alt.Text("if_rate:Q", format=".2f"),
            color=alt.condition(
                "datum.if_rate > 0.5", alt.value("black"), alt.value("white")
            ),
        )
    )
    # Title dropped: the LaTeX subcaption already labels this panel.
    # Larger fonts keep labels legible when the panel is scaled to ~0.48\linewidth.
    _hm_chart = (
        (_rects + _texts)
        .properties(
            width=max(500, len(_n_order) * 28),
            height=max(200, _hm["model"].nunique() * 30),
        )
        .configure_axis(labelFontSize=16, titleFontSize=17)
    )
    _hm_chart.save(str(plots_dir / "a4_if_rate_heatmap.png"), scale_factor=2)
    _hm_chart
    return


@app.cell
def _(DISPLAY_NAMES, all_models, mo):
    # Downstream uses evals_filtered which has model renamed to display names,
    # so selector options/values must also be display names for .isin() to match.
    _options = sorted({DISPLAY_NAMES.get(m, m) for m in all_models})
    archetype_selector = mo.ui.multiselect(
        options=_options,
        value=[
            DISPLAY_NAMES.get("gpt-5.2", "gpt-5.2"),
            DISPLAY_NAMES.get("gemma-3-27b-it", "gemma-3-27b-it"),
            DISPLAY_NAMES.get("claude-sonnet-4.6", "claude-sonnet-4.6"),
            DISPLAY_NAMES.get("kimi-k2", "kimi-k2"),
        ],
        label="Archetype models (pick 4–5 to compare transition shapes)",
    )
    archetype_selector
    return (archetype_selector,)


@app.cell
def _(COLOR_SCHEME, Plot, archetype_selector, evals_filtered, js, mo):
    # Archetype transition curves — P(output=T) vs N per selected model
    _tc_data = (
        evals_filtered[evals_filtered["model"].isin(archetype_selector.value)]
        .groupby(["model", "n_turns_int"])
        .agg(if_rate=("score", "mean"), stderr=("score_stderr", "mean"))
        .reset_index()
        .sort_values(["model", "n_turns_int"])
    )
    _tc_data["ci_lo"] = (_tc_data["if_rate"] - _tc_data["stderr"]).clip(lower=0)
    _tc_data["ci_hi"] = (_tc_data["if_rate"] + _tc_data["stderr"]).clip(upper=1)
    _tc_records = _tc_data.to_dict("records")

    mo.ui.anywidget(
        Plot.plot(
            {
                "x": {"label": "N turns"},
                "y": {"domain": [0, 1], "label": "P(output = T)  —  IF Rate"},
                "color": {"legend": True, "scheme": COLOR_SCHEME.capitalize()},
                "marks": [
                    Plot.areaY(
                        _tc_records,
                        {
                            "x": "n_turns_int",
                            "y1": "ci_lo",
                            "y2": "ci_hi",
                            "fill": "model",
                            "fillOpacity": 0.15,
                            "curve": "monotone-x",
                        },
                    ),
                    Plot.lineY(
                        _tc_records,
                        {
                            "x": "n_turns_int",
                            "y": "if_rate",
                            "stroke": "model",
                            "strokeWidth": 2,
                            "curve": "monotone-x",
                        },
                    ),
                    Plot.dot(
                        _tc_records,
                        {
                            "x": "n_turns_int",
                            "y": "if_rate",
                            "fill": "model",
                            "r": 3,
                            "title": js(
                                "d => `${d.model}: N=${d.n_turns_int}, IF=${d.if_rate.toFixed(2)}`"
                            ),
                        },
                    ),
                    # Plot.text(
                    #     _tc_records,
                    #     Plot.selectLast(
                    #         {
                    #             "x": "n_turns_int",
                    #             "y": "if_rate",
                    #             "z": "model",
                    #             "text": "model",
                    #             "textAnchor": "start",
                    #             "dx": 6,
                    #             "fontSize": 9,
                    #             "fill": "model",
                    #         }
                    #     ),
                    # ),
                    Plot.ruleY(
                        [0.5], {"stroke": "#ccc", "strokeDasharray": "4 2"}
                    ),
                ],
                "width": 750,
                "height": 400,
                "marginRight": 200,
                "title": "Transition Curves: Instruction following rate vs N",
            }
        )
    )
    return


@app.cell
def _(mo):
    threshold_slider_a5 = mo.ui.slider(
        start=0.1, stop=0.9, value=0.5, step=0.05, label="IF drop threshold (A5)"
    )
    threshold_slider_a5
    return (threshold_slider_a5,)


@app.cell
def _(
    caps_df_filtered,
    evals_filtered,
    make_scatter_chart,
    n_values_sorted,
    nudge_labels,
    pd,
    plots_dir,
    prep_benchmark_data,
    threshold_slider_a5,
):
    # Plot A5: First N where IF drops to threshold, vs capability benchmarks
    _threshold_a5 = threshold_slider_a5.value
    _avg_a5 = (
        evals_filtered.groupby(["model", "n_turns_int"])["score"]
        .mean()
        .reset_index()
    )
    _max_n = max(n_values_sorted, key=int)
    _sentinel = int(_max_n) + 5

    _records_a5 = []
    for _model in _avg_a5["model"].unique():
        _mdf = _avg_a5[_avg_a5["model"] == _model].sort_values("n_turns_int")
        _first = _mdf[_mdf["score"] <= _threshold_a5]
        _drop_n = (
            int(_first["n_turns_int"].iloc[0]) if len(_first) > 0 else _sentinel
        )
        _records_a5.append({"model": _model, "first_drop_n": _drop_n})

    _drop_df = pd.DataFrame(_records_a5)
    _a5_df = prep_benchmark_data(
        _drop_df, "first_drop_n", caps_df_filtered, nudge_labels
    )
    _a5_chart = make_scatter_chart(
        _a5_df,
        "first_drop_n",
        f"First N where IF <= {_threshold_a5}",
        f"First IF Drop (threshold={_threshold_a5}) vs Model Capability",
        log_y=True,
    )
    _a5_chart.save(
        str(plots_dir / "a5_first_if_drop_vs_capability.png"), scale_factor=2
    )
    _a5_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calibration analysis

    - Plot B1: mean calibration error. See notebooks\behavioral_vs_prediction_analysis.qmd for definition of the error. Arrow plot in observable-plot like this one: https://observablehq.com/@observablehq/plot-difference-arrows. Models on the y, IF rate on the x, averaged by condition.
    - Plot B2: calibration error vs capability index: same principle as plot A2
    - Plot B3: calibration error by condition: https://observablehq.com/@observablehq/plot-barley-trellis-arrows similar to plot B1, but faceted vertically by condition
    - Plot B4: distribution of calibration error. Not sure about this one, maybe a box plot per model? I want to show for each model how often it over or under predicts instruction following, giving more granularity than the mean calibration error in Plot B1
    """)
    return


@app.cell
def _(combined_errors):
    combined_errors.model.unique()
    return


@app.cell
def _(Plot, combined_errors_filtered, js, mo):
    # TODO: also produce a sort of count metric where I count for each evaluation setting whether the model under or over predicted instead of using the mean? but that might be messed up by the temperature..
    # on the other hand, I can compare for each sample across protocols 1 and 2 because the question data is the same for a given sample.
    # Plot B1: Arrow plot — mean actual vs predicted IF rate per model
    _b1_agg = (
        combined_errors_filtered.groupby("model")
        .agg(
            actual=("behavioral_score", "mean"),
            predicted=("prediction_predicted_score", "mean"),
        )
        .reset_index()
    )
    _b1_agg["direction"] = _b1_agg.apply(
        lambda r: (
            "over-predicts" if r["predicted"] > r["actual"] else "under-predicts"
        ),
        axis=1,
    )
    _b1_records = _b1_agg.to_dict("records")

    mo.ui.anywidget(
        Plot.plot(
            {
                "x": {"domain": [0, 1], "label": "IF Rate"},
                "y": {"label": None},
                "color": {
                    "domain": ["over-predicts", "under-predicts"],
                    "range": ["#e15759", "#4e79a7"],
                    "legend": True,
                },
                "marks": [
                    Plot.arrow(
                        _b1_records,
                        {
                            "x1": "actual",
                            "x2": "predicted",
                            "y": "model",
                            "stroke": "direction",
                            "strokeWidth": 2,
                            "sort": {"y": "-x1"},
                        },
                    ),
                    Plot.dot(
                        _b1_records,
                        {
                            "x": "actual",
                            "y": "model",
                            "fill": "black",
                            "r": 3,
                            "title": js("d => `Actual: ${d.actual.toFixed(3)}`"),
                        },
                    ),
                    Plot.dot(
                        _b1_records,
                        {
                            "x": "predicted",
                            "y": "model",
                            "fill": "direction",
                            "symbol": "diamond",
                            "r": 4,
                            "title": js(
                                "d => `Predicted: ${d.predicted.toFixed(3)}`"
                            ),
                        },
                    ),
                    Plot.ruleX(
                        [0.5], {"stroke": "#ccc", "strokeDasharray": "4 2"}
                    ),
                ],
                "width": 600,
                "height": 400,
                "marginLeft": 160,
                "title": "Calibration: Actual vs Predicted IF Rate (avg)",
            }
        )
    )
    return


@app.cell
def _(
    caps_df_filtered,
    combined_errors_filtered,
    make_scatter_chart,
    plots_dir,
    prep_benchmark_data,
):
    # Plot B2: Scatter — mean absolute calibration error vs capability (faceted by benchmark)
    combined_errors_filtered_copy = combined_errors_filtered.copy()
    combined_errors_filtered_copy["abs_calibration_error_pct"] = (
        combined_errors_filtered_copy["calibration_error_pct"].abs()
    )
    _b2_agg = (
        combined_errors_filtered_copy.groupby("model")["abs_calibration_error_pct"]
        .mean()
        .reset_index(name="mean_abs_calibration_error")
    )
    _b2_df = prep_benchmark_data(
        _b2_agg,
        "mean_abs_calibration_error",
        caps_df_filtered,
        lambda *a: a[0],
    )
    _b2_chart = make_scatter_chart(
        _b2_df,
        "mean_abs_calibration_error",
        "Mean |Calibration Error| (%)",
        "Absolute Calibration Error vs Model Capability",
    )
    _b2_chart.save(
        str(plots_dir / "b2_calibration_error_vs_capability.png"), scale_factor=2
    )
    _b2_chart
    return


@app.cell
def _(Plot, combined_errors_filtered, js, mo):
    # Plot B3: Trellis arrows — actual vs predicted by condition (faceted)
    _b3_agg = (
        combined_errors_filtered.groupby(["model", "condition"])
        .agg(
            actual=("behavioral_score", "mean"),
            predicted=("prediction_predicted_score", "mean"),
        )
        .reset_index()
    )
    _b3_agg["direction"] = _b3_agg.apply(
        lambda r: (
            "over-predicts" if r["predicted"] > r["actual"] else "under-predicts"
        ),
        axis=1,
    )
    _b3_records = _b3_agg.to_dict("records")

    mo.ui.anywidget(
        Plot.plot(
            {
                "x": {"domain": [0, 1], "label": "IF Rate"},
                "y": {"label": None},
                "fy": {"label": None},
                "color": {
                    "domain": ["over-predicts", "under-predicts"],
                    "range": ["#e15759", "#4e79a7"],
                    "legend": True,
                },
                "marks": [
                    Plot.arrow(
                        _b3_records,
                        {
                            "x1": "actual",
                            "x2": "predicted",
                            "y": "model",
                            "fy": "condition",
                            "stroke": "direction",
                            "strokeWidth": 1.5,
                            "sort": {"y": "-x1"},
                        },
                    ),
                    Plot.dot(
                        _b3_records,
                        {
                            "x": "actual",
                            "y": "model",
                            "fy": "condition",
                            "fill": "black",
                            "r": 2.5,
                            "title": js("d => `Actual: ${d.actual.toFixed(3)}`"),
                        },
                    ),
                    Plot.dot(
                        _b3_records,
                        {
                            "x": "predicted",
                            "y": "model",
                            "fy": "condition",
                            "fill": "direction",
                            "symbol": "diamond",
                            "r": 3,
                            "title": js(
                                "d => `Predicted: ${d.predicted.toFixed(3)}`"
                            ),
                        },
                    ),
                ],
                "width": 700,
                "height": 1400,
                "marginLeft": 120,
                "marginRight": 120,
                "title": "Calibration by Condition: Actual vs Predicted IF Rate",
            }
        )
    )
    return


@app.cell
def _(COLOR_SCHEME, alt, combined_errors_filtered, pd, plots_dir):
    # Plot B4: Box plot — distribution of calibration error per model
    _rule_b4 = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(strokeDash=[4, 2], color="gray")
        .encode(x="x:Q")
    )
    _b4_data = combined_errors_filtered.dropna(subset=["calibration_error_pct"])
    _box_b4 = (
        alt.Chart(_b4_data)
        .mark_boxplot(extent=1.5)
        .encode(
            x=alt.X("calibration_error_pct:Q", title="Calibration Error (%)"),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.Color(
                "model:N", scale=alt.Scale(scheme=COLOR_SCHEME), legend=None
            ),
        )
    )
    _b4_chart = (_box_b4 + _rule_b4).properties(
        width=500, height=400, title="Distribution of Calibration Error by Model"
    )
    _b4_chart.save(
        str(plots_dir / "b4_calibration_error_distribution.png"), scale_factor=2
    )
    _b4_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction analysis
    - Plot C1: avg "predicts instruction following" rate by model and capability, like plots A2 and B2
    - Plot C2: prediction accuracy (did the model's prediction match its actual behavior?) vs capability, same format
    """)
    return


@app.cell
def _(
    caps_df_filtered,
    evals_prediction_filtered,
    make_scatter_chart,
    plots_dir,
    prep_benchmark_data,
):
    # Plot C1: Scatter — avg prediction of IF rate vs capability (faceted by benchmark)
    _c1_agg = (
        evals_prediction_filtered.groupby("model")["score_prediction_instruction"]
        .mean()
        .reset_index(name="prediction_rate")
    )
    _c1_df = prep_benchmark_data(
        _c1_agg,
        "prediction_rate",
        caps_df_filtered,
        lambda *a: a[0],
    )
    _c1_chart = make_scatter_chart(
        _c1_df,
        "prediction_rate",
        "Avg 'Predicts IF' Rate",
        "Prediction of IF Rate vs Model Capability",
        y_domain=[0, 0.55],
    )
    _c1_chart.save(
        str(plots_dir / "c1_prediction_rate_vs_capability.png"), scale_factor=2
    )
    _c1_chart
    return


@app.cell
def _(
    caps_df_filtered,
    evals_prediction_filtered,
    make_scatter_chart,
    plots_dir,
    prep_benchmark_data,
):
    # Plot C2: Scatter — prediction accuracy vs capability (faceted by benchmark)
    _c2_agg = (
        evals_prediction_filtered.groupby("model")["score_prediction_accuracy"]
        .mean()
        .reset_index(name="prediction_accuracy")
    )
    _c2_df = prep_benchmark_data(
        _c2_agg,
        "prediction_accuracy",
        caps_df_filtered,
        lambda *a: a[0],
    )
    _c2_chart = make_scatter_chart(
        _c2_df,
        "prediction_accuracy",
        "Prediction Accuracy",
        "Prediction Accuracy vs Model Capability",
        y_domain=[0, 1],
    )
    _c2_chart.save(
        str(plots_dir / "c2_prediction_accuracy_vs_capability.png"), scale_factor=2
    )
    _c2_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of aligned/misaligned
    - Plot D3: radar plot with IF rate for aligned conditions overlapped with IF rate of unaligned conditions for each model, but faceted by specific condition couple (the cats one and the earth round/flat one): https://observablehq.com/@observablehq/plot-radar-chart https://observablehq.com/@observablehq/plot-radar-chart-faceted
    """)
    return


@app.cell
def _(Plot, evals_filtered, js, make_radar_chart, mo):
    # Plot D3: Radar chart per condition_pair — aligned vs misaligned
    _d3_data = evals_filtered[evals_filtered["instruction_aligned"].notna()].copy()
    _d3_data["alignment"] = _d3_data["instruction_aligned"].map(
        {True: "aligned", False: "misaligned"}
    )
    _d3_agg = (
        _d3_data.groupby(["model", "condition_pair", "alignment"])["score"]
        .mean()
        .reset_index()
    )

    _condition_pairs = sorted(_d3_agg["condition_pair"].unique().tolist())
    _categories = sorted(_d3_agg["model"].unique().tolist())

    _tabs = {}
    for _cp in _condition_pairs:
        _cp_data = _d3_agg[_d3_agg["condition_pair"] == _cp]
        _points = []
        for _, _row in _cp_data.iterrows():
            _points.append(
                {
                    "key": _row["model"],
                    "value": _row["score"],
                    "name": _row["alignment"],
                }
            )
        _tabs[_cp] = mo.ui.anywidget(
            make_radar_chart(
                Plot, js, _points, _categories, f"Aligned vs Misaligned: {_cp}"
            )
        )

    mo.ui.tabs(_tabs)
    return


@app.cell
def _(alt, evals_filtered, plots_dir):
    # Plot D4: Grouped bar — avg IF rate by alignment per model, sorted by alignment gap
    _d4_data = evals_filtered[evals_filtered["instruction_aligned"].notna()].copy()
    _d4_data["alignment"] = _d4_data["instruction_aligned"].map(
        {True: "aligned", False: "misaligned"}
    )
    _d4_agg = (
        _d4_data.groupby(["model", "alignment"])
        .agg(avg_if_rate=("score", "mean"), stderr=("score_stderr", "mean"))
        .reset_index()
    )
    _d4_agg["ci_lo"] = (_d4_agg["avg_if_rate"] - _d4_agg["stderr"]).clip(lower=0)
    _d4_agg["ci_hi"] = (_d4_agg["avg_if_rate"] + _d4_agg["stderr"]).clip(upper=1)
    _d4_wide = _d4_agg.pivot(
        index="model", columns="alignment", values="avg_if_rate"
    ).reset_index()
    _d4_wide["gap"] = _d4_wide.get("aligned", 0) - _d4_wide.get("misaligned", 0)
    _model_order = _d4_wide.sort_values("gap", ascending=False)["model"].tolist()
    _d4_color = alt.Color(
        "alignment:N",
        scale=alt.Scale(
            domain=["aligned", "misaligned"],
            range=["#8dd3c7", "#fb8072"],
        ),
        legend=alt.Legend(
            title="Instruction alignment",
            orient="none",
            legendX=320,
            legendY=410,
            fillColor="white",
            strokeColor="#ccc",
            padding=6,
        ),
    )
    _d4_y = alt.Y("model:N", sort=_model_order, title=None)
    _d4_yoff = alt.YOffset(
        "alignment:N",
        scale=alt.Scale(domain=["aligned", "misaligned"]),
    )
    _d4_bars = (
        alt.Chart(_d4_agg)
        .mark_bar()
        .encode(
            x=alt.X(
                "avg_if_rate:Q",
                title="Avg IF Rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=_d4_y,
            yOffset=_d4_yoff,
            color=_d4_color,
            tooltip=[
                "model",
                "alignment",
                alt.Tooltip("avg_if_rate:Q", format=".3f", title="Avg IF Rate"),
                alt.Tooltip("stderr:Q", format=".3f", title="±SE"),
            ],
        )
    )
    _d4_err = (
        alt.Chart(_d4_agg)
        .mark_errorbar(color="black", thickness=1)
        .encode(
            x=alt.X("ci_lo:Q", title=""),
            x2="ci_hi:Q",
            y=_d4_y,
            yOffset=_d4_yoff,
        )
    )
    _d4_chart = (_d4_bars + _d4_err).properties(
        width=450,
        height=500,
        title="Avg IF Rate: Aligned vs Misaligned Instructions (sorted by gap)",
    )
    _d4_chart.save(
        str(plots_dir / "d4_aligned_misaligned_if_rate.png"), scale_factor=2
    )
    _d4_chart
    return


@app.cell
def _(mo):
    benchmark_dropdown = mo.ui.dropdown(
        options={
            "GPQA": "gpqa",
            "IFBench": "ifbench",
        },
        value="GPQA",
        label="Capability metric (color)",
    )
    benchmark_dropdown
    return (benchmark_dropdown,)


@app.cell
def _(
    alt,
    benchmark_dropdown,
    caps_df_filtered,
    evals_filtered,
    pd,
    plots_dir,
):
    # Plot D6: Quadrant scatter — aligned vs misaligned IF rate, color = capability
    _bm = benchmark_dropdown.value
    _bm_label = {"gpqa": "GPQA", "ifbench": "IFBench"}[_bm]

    _d6_data = evals_filtered[evals_filtered["instruction_aligned"].notna()].copy()
    _d6_data["alignment"] = _d6_data["instruction_aligned"].map(
        {True: "aligned", False: "misaligned"}
    )
    _d6_agg = (
        _d6_data.groupby(["model", "alignment"])["score"]
        .mean()
        .unstack("alignment")
        .reset_index()
    )
    _d6_agg.columns.name = None
    _d6_agg = _d6_agg.rename(
        columns={"aligned": "aligned_if", "misaligned": "misaligned_if"}
    )

    _d6_caps = caps_df_filtered[["model", _bm]].dropna(subset=[_bm])
    _d6_df = _d6_agg.merge(_d6_caps, on="model", how="inner")

    _diag = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    _diag_line = (
        alt.Chart(_diag)
        .mark_line(strokeDash=[4, 2], color="gray")
        .encode(x="x:Q", y="y:Q")
    )
    _pts = (
        alt.Chart(_d6_df)
        .mark_point(filled=True, size=120)
        .encode(
            x=alt.X(
                "aligned_if:Q",
                title="Aligned IF Rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "misaligned_if:Q",
                title="Misaligned IF Rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                f"{_bm}:Q",
                title=_bm_label,
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(title=_bm_label),
            ),
            tooltip=[
                "model",
                alt.Tooltip("aligned_if:Q", format=".3f", title="Aligned IF"),
                alt.Tooltip(
                    "misaligned_if:Q", format=".3f", title="Misaligned IF"
                ),
                alt.Tooltip(f"{_bm}:Q", format=".2f", title=_bm_label),
            ],
        )
    )
    _labels = (
        alt.Chart(_d6_df)
        .mark_text(align="left", dx=6, fontSize=9)
        .encode(
            x="aligned_if:Q",
            y="misaligned_if:Q",
            text="model:N",
        )
    )
    _d6_chart = (_diag_line + _pts + _labels).properties(
        width=450,
        height=450,
        title=f"Aligned vs Misaligned IF Rate (color = {_bm_label})",
    )
    _d6_chart.save(
        str(plots_dir / "d6_aligned_vs_misaligned_quadrant.png"), scale_factor=2
    )
    _d6_chart
    return


@app.cell
def _(all_models, mo):
    alignment_curve_selector = mo.ui.multiselect(
        options=sorted(all_models),
        value=sorted(all_models)[:4],
        label="Models for aligned/misaligned transition curves (pick 3–4)",
    )
    alignment_curve_selector
    return (alignment_curve_selector,)


@app.cell
def _(Plot, alignment_curve_selector, evals_filtered, js, mo):
    # Plot D5: Paired transition curves — aligned vs misaligned per selected model
    # d5_data = evals_filtered[evals_filtered["instruction_aligned"].notna()].copy()
    _d5_data = evals_filtered[evals_filtered["condition_pair"] == "value"].copy()
    _d5_data["alignment"] = _d5_data["instruction_aligned"].map(
        {True: "aligned", False: "misaligned"}
    )
    _d5_data = _d5_data[
        _d5_data["model"].isin(alignment_curve_selector.value)
    ].copy()
    _d5_agg = (
        _d5_data.groupby(["model", "alignment", "n_turns_int"])
        .agg(if_rate=("score", "mean"), stderr=("score_stderr", "mean"))
        .reset_index()
        .sort_values(["model", "alignment", "n_turns_int"])
    )
    _d5_agg["ci_lo"] = (_d5_agg["if_rate"] - _d5_agg["stderr"]).clip(lower=0)
    _d5_agg["ci_hi"] = (_d5_agg["if_rate"] + _d5_agg["stderr"]).clip(upper=1)
    _d5_records = _d5_agg.to_dict("records")

    mo.ui.anywidget(
        Plot.plot(
            {
                "x": {"label": "N turns"},
                "y": {"domain": [0, 1], "label": "IF Rate"},
                "fx": {"label": None},
                "color": {
                    "domain": ["aligned", "misaligned"],
                    "range": ["#8dd3c7", "#fb8072"],
                    "legend": True,
                },
                "marks": [
                    Plot.areaY(
                        _d5_records,
                        {
                            "x": "n_turns_int",
                            "y1": "ci_lo",
                            "y2": "ci_hi",
                            "fill": "alignment",
                            "fx": "model",
                            "fillOpacity": 0.15,
                            "curve": "monotone-x",
                        },
                    ),
                    Plot.lineY(
                        _d5_records,
                        {
                            "x": "n_turns_int",
                            "y": "if_rate",
                            "stroke": "alignment",
                            "fx": "model",
                            "strokeWidth": 2,
                            "curve": "monotone-x",
                        },
                    ),
                    Plot.dot(
                        _d5_records,
                        {
                            "x": "n_turns_int",
                            "y": "if_rate",
                            "fill": "alignment",
                            "fx": "model",
                            "r": 3,
                            "title": js(
                                "d => `N=${d.n_turns_int}, IF=${d.if_rate.toFixed(2)} (${d.alignment})`"
                            ),
                        },
                    ),
                    Plot.ruleY(
                        [0.5], {"stroke": "#ccc", "strokeDasharray": "4 2"}
                    ),
                ],
                "width": 1000,
                "height": 300,
                "marginLeft": 50,
                "title": "Aligned vs Misaligned Instruction: Transition Curves by Model",
            }
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 3.4 — Reasoning models
    - Plot R1: Transition curves (IF rate vs N) for reasoning vs non-reasoning variants of GPT-5.2 and Hermes-4, faceted by aligned vs misaligned condition. Shows both the overall improvement from reasoning and persistent alignment sensitivity.
    """)
    return


@app.cell
def _(all_models, mo):
    reasoning_model_selector = mo.ui.multiselect(
        options=sorted(all_models),
        value=[
            "gpt-5.2",
            "gpt-5.2-medium",
            "hermes-4-70b",
            "hermes-4-70b-reasoning",
        ],
        label="Models for reasoning comparison (select reasoning and non-reasoning variants)",
    )
    reasoning_model_selector
    return (reasoning_model_selector,)


@app.cell
def _(Plot, evals, js, mo, reasoning_model_selector, reasoning_models):
    # Plot R1: Reasoning vs non-reasoning transition curves, faceted by alignment
    _selected = reasoning_model_selector.value
    _r1_data = evals[evals["model"].isin(_selected)].copy()
    _r1_data = _r1_data[_r1_data["instruction_aligned"].notna()].copy()

    if len(_r1_data) == 0:
        _r1_out = mo.md(
            "_No data for selected models with alignment-axis conditions. Select reasoning and non-reasoning variants of the same base model._"
        )
    else:

        def _base_label(m):
            for suffix in [
                " (medium)",
                " (low)",
                " (high)",
                "-reasoning",
                "-medium",
                "-low",
                "-high",
            ]:
                if m.endswith(suffix):
                    return m[: -len(suffix)]
            return m

        _r1_data["reasoning_type"] = _r1_data["model"].apply(
            lambda m: "reasoning" if m in reasoning_models else "non-reasoning"
        )
        _r1_data["base_model"] = _r1_data["model"].apply(_base_label)
        _r1_data["alignment"] = _r1_data["instruction_aligned"].map(
            {True: "aligned", False: "misaligned"}
        )

        _r1_agg = (
            _r1_data.groupby(
                ["base_model", "reasoning_type", "n_turns_int", "alignment"]
            )
            .agg(if_rate=("score", "mean"), stderr=("score_stderr", "mean"))
            .reset_index()
            .sort_values(
                ["base_model", "reasoning_type", "alignment", "n_turns_int"]
            )
        )
        _r1_agg["ci_lo"] = (_r1_agg["if_rate"] - _r1_agg["stderr"]).clip(lower=0)
        _r1_agg["ci_hi"] = (_r1_agg["if_rate"] + _r1_agg["stderr"]).clip(upper=1)
        _r1_records = _r1_agg.to_dict("records")

        _r1_out = mo.ui.anywidget(
            Plot.plot(
                {
                    "x": {"label": "N turns"},
                    "y": {"domain": [0, 1], "label": "IF Rate"},
                    "fx": {"label": None},
                    "fy": {"label": None},
                    "color": {
                        "domain": ["reasoning", "non-reasoning"],
                        "range": ["#80b1d3", "#fb8072"],
                        "legend": True,
                    },
                    "marks": [
                        Plot.areaY(
                            _r1_records,
                            {
                                "x": "n_turns_int",
                                "y1": "ci_lo",
                                "y2": "ci_hi",
                                "fill": "reasoning_type",
                                "fx": "base_model",
                                "fy": "alignment",
                                "fillOpacity": 0.15,
                                "curve": "monotone-x",
                            },
                        ),
                        Plot.lineY(
                            _r1_records,
                            {
                                "x": "n_turns_int",
                                "y": "if_rate",
                                "stroke": "reasoning_type",
                                "fx": "base_model",
                                "fy": "alignment",
                                "strokeWidth": 2,
                                "curve": "monotone-x",
                            },
                        ),
                        Plot.dot(
                            _r1_records,
                            {
                                "x": "n_turns_int",
                                "y": "if_rate",
                                "fill": "reasoning_type",
                                "fx": "base_model",
                                "fy": "alignment",
                                "r": 3,
                                "title": js(
                                    "d => `N=${d.n_turns_int}, IF=${d.if_rate.toFixed(2)} (${d.reasoning_type})`"
                                ),
                            },
                        ),
                        Plot.ruleY(
                            [0.5], {"stroke": "#ccc", "strokeDasharray": "4 2"}
                        ),
                    ],
                    "width": 700,
                    "height": 500,
                    "marginLeft": 60,
                    "marginRight": 80,
                    "title": "Impact of reasoning on instruction following rate",
                }
            )
        )
    _r1_out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 3.5 — Self-prediction analysis

    The arrow plot (B1 above) shows overall mean calibration. Additional analyses below:

    - Plot E1: Calibration bucket plot — for each behavioral regime bucket (IF-dominant ≥0.7, mixed 0.3–0.7, induction-dominant ≤0.3), the mean predicted IF rate vs mean actual IF rate per model. The diagonal represents perfect calibration; points above it mean the model over-predicts its own instruction-following.
    - Plot E2: Prediction-changes-behavior — difference in IF rate between Protocol 2 (actual output after making a self-prediction) and Protocol 1 (behavioral baseline), restricted to transition-adjacent N values (behavioral score 0.2–0.8). A bar above zero means prediction boosted instruction-following.
    """)
    return


@app.cell
def _(COLOR_SCHEME, alt, combined_errors_filtered, pd, plots_dir):
    # Plot E1: Calibration bucket plot — predicted vs actual T rate by behavioral regime
    _e1_data = combined_errors_filtered.dropna(
        subset=["behavioral_score", "prediction_predicted_score"]
    ).copy()


    def _classify_regime(s):
        if s >= 0.7:
            return "Instruction-dominant"
        elif s <= 0.3:
            return "Pattern-dominant"
        else:
            return "Mixed"


    _e1_data["regime"] = _e1_data["behavioral_score"].apply(_classify_regime)

    _e1_agg = (
        _e1_data.groupby(["model", "regime"])
        .agg(
            actual_t_rate=("behavioral_score", "mean"),
            predicted_t_rate=("prediction_predicted_score", "mean"),
            n=("behavioral_score", "count"),
        )
        .reset_index()
    )

    _diag = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    _diag_line = (
        alt.Chart(_diag)
        .mark_line(strokeDash=[4, 2], color="gray")
        .encode(x="x:Q", y="y:Q")
    )
    _e1_scatter = (
        alt.Chart(_e1_agg)
        .mark_point(size=250, filled=True, opacity=0.85)
        .encode(
            x=alt.X(
                "actual_t_rate:Q",
                title="Actual IF Rate (behavioral)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "predicted_t_rate:Q",
                title="Predicted IF Rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("model:N", scale=alt.Scale(scheme=COLOR_SCHEME)),
            shape=alt.Shape(
                "regime:N",
                scale=alt.Scale(
                    domain=["Instruction-dominant", "Mixed", "Pattern-dominant"],
                    range=["circle", "square", "cross"],
                ),
                title="Behavioral Regime",
            ),
            tooltip=[
                "model",
                "regime",
                alt.Tooltip("actual_t_rate:Q", format=".3f", title="Actual IF"),
                alt.Tooltip(
                    "predicted_t_rate:Q", format=".3f", title="Predicted IF"
                ),
                alt.Tooltip("n:Q", title="N samples"),
            ],
        )
    )
    _over_annot = (
        alt.Chart(
            pd.DataFrame({"x": [0.2], "y": [0.72], "text": ["▲ over-predicts IF"]})
        )
        .mark_text(color="gray", fontSize=14, align="left")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    _under_annot = (
        alt.Chart(
            pd.DataFrame(
                {"x": [0.6], "y": [0.22], "text": ["▼ under-predicts IF"]}
            )
        )
        .mark_text(color="gray", fontSize=14, align="left")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )
    _e1_chart = (_diag_line + _e1_scatter + _over_annot + _under_annot).properties(
        width=500,
        height=450,
        title="Self-Prediction Calibration by Behavioral Regime",
    )
    _e1_chart.save(str(plots_dir / "e1_calibration_bucket.png"), scale_factor=2)
    _e1_chart
    return


@app.cell
def _(alt, combined_errors_filtered, pd, plots_dir):
    # Plot E2: Paired difference plot — Protocol 2 vs Protocol 1 IF rate at transition N values
    _e2_data = combined_errors_filtered.dropna(
        subset=["behavioral_score", "prediction_actual_score"]
    ).copy()
    _e2_data["if_diff"] = (
        _e2_data["prediction_actual_score"] - _e2_data["behavioral_score"]
    )

    # Transition-adjacent N: behavioral score between 0.2 and 0.8 (model not clearly IF- or PF-dominant)
    _e2_transition = _e2_data[
        (_e2_data["behavioral_score"] > 0.2) & (_e2_data["behavioral_score"] < 0.8)
    ].copy()
    _e2_transition


    _e2_agg = (
        _e2_transition.groupby("model")["if_diff"]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(columns={"mean": "mean_diff", "sem": "stderr"})
    )
    _e2_agg["ci_lo"] = _e2_agg["mean_diff"] - 1.96 * _e2_agg["stderr"]
    _e2_agg["ci_hi"] = _e2_agg["mean_diff"] + 1.96 * _e2_agg["stderr"]

    _e2_bars = (
        alt.Chart(_e2_agg)
        .mark_bar()
        .encode(
            x=alt.X(
                "mean_diff:Q",
                title="ΔIF Rate (Protocol 2 − Protocol 1)",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.mean_diff > 0,
                alt.value("#2ca02c"),
                alt.value("#d62728"),
            ),
            tooltip=[
                "model",
                alt.Tooltip("mean_diff:Q", format=".3f", title="Mean ΔIF"),
                alt.Tooltip("ci_lo:Q", format=".3f", title="95% CI lo"),
                alt.Tooltip("ci_hi:Q", format=".3f", title="95% CI hi"),
            ],
        )
    )
    _e2_error = (
        alt.Chart(_e2_agg)
        .mark_errorbar()
        .encode(
            x=alt.X("ci_lo:Q", title=""),
            x2="ci_hi:Q",
            y=alt.Y("model:N", sort="-x"),
        )
    )
    _e2_rule = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(strokeDash=[4, 2], color="gray")
        .encode(x="x:Q")
    )
    _e2_chart = (_e2_bars + _e2_error + _e2_rule).properties(
        width=500,
        height=400,
        title="Effect of Self-Prediction on IF Rate at Transition N Values (95% CI)",
    )
    _e2_chart.save(
        str(plots_dir / "e2_prediction_changes_behavior.png"), scale_factor=2
    )
    _e2_chart
    return


@app.cell
def _(alt, evals_filtered, pd, plots_dir):
    # D5: aggregate transition curves — aligned vs misaligned instructions vs N,
    # averaged over models and the value/factual condition pairs (fixed-output).
    # Compact main-text figure beside the per-model alignment-gap table.
    _d5 = evals_filtered[evals_filtered["instruction_aligned"].notna()].copy()
    _d5["alignment"] = _d5["instruction_aligned"].map(
        {True: "aligned", False: "misaligned"}
    )
    # Per-model mean per (alignment, N), then mean and +/- 1 SE band across models.
    _d5_pm = (
        _d5.groupby(["model", "alignment", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="if_rate")
    )
    _d5_agg = (
        _d5_pm.groupby(["alignment", "n_turns_int"])["if_rate"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    _d5_agg["se"] = _d5_agg["std"] / _d5_agg["n"] ** 0.5
    _d5_agg["lo"] = (_d5_agg["mean"] - _d5_agg["se"]).clip(0, 1)
    _d5_agg["hi"] = (_d5_agg["mean"] + _d5_agg["se"]).clip(0, 1)
    _d5_color = alt.Color(
        "alignment:N",
        scale=alt.Scale(
            domain=["aligned", "misaligned"],
            range=["#4daf4a", "#e41a1c"],
        ),
        legend=alt.Legend(title=None),
    )
    _d5_base = alt.Chart(_d5_agg)
    _d5_band = _d5_base.mark_area(opacity=0.15).encode(
        x=alt.X("n_turns_int:Q", title="N turns"),
        y=alt.Y("lo:Q", title="Mean IF rate", scale=alt.Scale(domain=[0, 1])),
        y2="hi:Q",
        color=_d5_color,
    )
    _d5_lines = _d5_base.mark_line(point=True).encode(
        x="n_turns_int:Q",
        y=alt.Y("mean:Q", scale=alt.Scale(domain=[0, 1])),
        color=_d5_color,
    )
    _d5_rule = (
        alt.Chart(pd.DataFrame({"y": [0.5]}))
        .mark_rule(color="#ccc", strokeDash=[4, 2])
        .encode(y="y:Q")
    )
    _d5_chart = (_d5_band + _d5_lines + _d5_rule).properties(width=320, height=240)
    _d5_chart.save(str(plots_dir / "d5_alignment_curves.png"), scale_factor=2)
    _d5_chart
    return


if __name__ == "__main__":
    app.run()
