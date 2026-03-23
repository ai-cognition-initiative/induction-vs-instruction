import marimo

__generated_with = "0.21.1"
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
    # Analysis of induction vs prediction experiments - Task-based conditions

    Your job is to implement all plots in this notebook. Unless specified otherwise, the plotting library should be Altair. You might be provided with links to specific plot examples, if so, use that implementation. Familiarize yourself with the experiment first. Code cells marked as todo are where the code is going to go. Add descriptions to markdown cells where appropriate.

    Useful skill/MCPs: marimo, observable-plot, napkin (for info about data wrangling and plotting)

    Acronyms:
    - IF: instruction-following
    - PF: pattern-following (cases where the models falls prey to induction)
    - IT: instruction template
    - N: number of turns, goes from 1 to 50 but it does not cover all values. Be careful with the data format.

    Note: only behavioral data is available for these conditions (no self-prediction protocol).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data loading

    - Load data from outputs\viz\dynamic. Different files will be needed for different plots.
    - Load model capability scores from data\model_capability_scores.json. the key score is intelligence_index
    """)
    return


@app.cell
def _(Path, json, pd):
    _root = Path(__file__).resolve().parent.parent.parent
    _task_based = _root / "outputs" / "viz" / "dynamic"

    evals_all = pd.read_parquet(_task_based / "evals.parquet")
    evals = evals_all[evals_all["instruction"] == "instruction_no_hint"].copy()
    evals["n_turns_int"] = evals["n_turns"].astype(int)

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
        evals,
        evals_all,
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
            # "llama-3.1-70b-instruct"
        ],
        label="Models to exclude",
    )
    model_exclusion
    return (model_exclusion,)


@app.cell
def _(DISPLAY_NAMES, caps_df, evals, evals_all, model_exclusion):
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
    caps_df_filtered = _rename(
        caps_df[~caps_df["model"].isin(_excluded)].copy()
        if _excluded
        else caps_df.copy()
    )
    return caps_df_filtered, evals_all_filtered, evals_filtered


@app.cell
def _(Path):
    plots_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "plots"
        / "dynamic"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    return (plots_dir,)


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
        start=0.5, stop=1.0, value=0.8, step=0.05, label="IF/PF Threshold"
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
    _counts = _all_combos.merge(_counts, on=["model", "category"], how="left").fillna({"count": 0})
    _counts["count"] = _counts["count"].astype(int)

    # Sort models by IF-dominant count ascending (most IF-dominant on top)
    _if_counts = _counts[_counts["category"] == "IF-dominant"].set_index("model")["count"]
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
    _a1_chart.save(str(plots_dir / "a1_model_behavior_stacked.png"), scale_factor=2)
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

    _if_counts_d = _counts_d[_counts_d["category"] == "IF-dominant"].set_index("model")["count"]
    _model_order_d = _if_counts_d.sort_values(ascending=True).index.tolist()

    (
        alt.Chart(_counts_d)
        .mark_bar(cornerRadius=4)
        .encode(
            y=alt.Y("model:N", sort=_model_order_d, title=None),
            x=alt.X(
                "count:Q",
                stack=True,
                title=f"N values (out of {_n_total_d})",
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
            title=f"{a1_detail_condition.value} (threshold={_threshold_d})",
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
        .properties(title=f"IF Rate by Model and Condition ({_sel})")
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
    # Per-condition IF rate averaged across models and N turns
    # SE combines between-cell variance and mean within-cell resampling variance
    _sel_cond = instruction_dropdown.value
    _cond_data = evals_all_filtered[
        evals_all_filtered["instruction"] == _sel_cond
    ].copy()
    _cond_records = []
    for _cond, _grp in _cond_data.groupby("condition"):
        _cell_means = _grp["score"]
        _cell_ses = _grp["score_stderr"]
        _k = len(_cell_means)
        _mean_if = float(_cell_means.mean())
        _between_var = float(_cell_means.var(ddof=1)) if _k > 1 else 0.0
        _within_var = float((_cell_ses**2).mean())
        _se = ((_between_var + _within_var) / _k) ** 0.5
        _cond_records.append(
            {
                "condition": _cond,
                "mean_if": _mean_if,
                "se": _se,
                "k": _k,
                "ci_lo": max(0.0, _mean_if - _se),
                "ci_hi": min(1.0, _mean_if + _se),
            }
        )
    _cond_df = pd.DataFrame(_cond_records)
    _cond_bars = (
        alt.Chart(_cond_df)
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
    _cond_err = (
        alt.Chart(_cond_df)
        .mark_errorbar(color="black", ticks=True)
        .encode(
            x=alt.X("ci_lo:Q", title=""),
            x2="ci_hi:Q",
            y=alt.Y("condition:N", sort="-x"),
        )
    )
    _cond_chart = (_cond_bars + _cond_err).properties(
        width=500,
        height=alt.Step(20),
        title=f"IF Rate per Condition — average over models & N",
    ).configure_axisY(labelLimit=300)
    _cond_chart.save(
        str(plots_dir / "a6_per_condition_if_rate.png"), scale_factor=2
    )
    _cond_chart
    return


@app.cell
def _(Plot, evals_filtered, js, mo):
    # Plot A4: Stacked unit chart — colored stripes of IF rate by N
    _a4_agg = (
        evals_filtered.groupby(["model", "n_turns", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="behavioral_score")
        .sort_values(["model", "n_turns_int"])
    )
    _a4_records = _a4_agg.to_dict("records")

    mo.ui.anywidget(
        Plot.plot(
            {
                "x": {"axis": None},
                "y": {"label": "Model"},
                "color": {
                    "type": "linear",
                    "scheme": "RdYlGn",
                    "domain": [0, 1],
                    "label": "IF Rate",
                    "legend": True,
                },
                "marks": [
                    Plot.barX(
                        _a4_records,
                        {
                            "x": 1,
                            "y": "model",
                            "fill": "behavioral_score",
                            "sort": {"y": "-x", "reduce": "sum"},
                            "order": "n_turns_int",
                            "title": js(
                                "d => `N=${d.n_turns}, IF=${d.behavioral_score.toFixed(2)}`"
                            ),
                        },
                    ),
                    Plot.text(
                        _a4_records,
                        Plot.stackX(
                            {
                                "x": 1,
                                "y": "model",
                                "order": "n_turns_int",
                                "text": "n_turns",
                                "fill": "white",
                                "fontSize": 8,
                            }
                        ),
                    ),
                ],
                "width": 800,
                "height": 550,
                "marginLeft": 160,
            }
        )
    )
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
        .mark_text(fontSize=7)
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
    _hm_chart = (_rects + _texts).properties(
        width=max(500, len(_n_order) * 28),
        height=max(200, _hm["model"].nunique() * 28),
        title="IF Rate Heatmap: Model × N turns (Altair)",
    )
    _hm_chart.save(str(plots_dir / "a4_if_rate_heatmap.png"), scale_factor=2)
    _hm_chart
    return


@app.cell
def _(all_models, mo):
    archetype_selector = mo.ui.multiselect(
        options=sorted(all_models),
        value=sorted(all_models)[:4],
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
                    Plot.text(
                        _tc_records,
                        Plot.selectLast(
                            {
                                "x": "n_turns_int",
                                "y": "if_rate",
                                "z": "model",
                                "text": "model",
                                "textAnchor": "start",
                                "dx": 6,
                                "fontSize": 9,
                                "fill": "model",
                            }
                        ),
                    ),
                    Plot.ruleY(
                        [0.5], {"stroke": "#ccc", "strokeDasharray": "4 2"}
                    ),
                ],
                "width": 750,
                "height": 400,
                "marginRight": 200,
                "title": "Archetype Transition Curves: P(output=T) vs N",
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
    ## Effect of hint
    - Plot D1: radar plot with non-hint IF rate overlapped with IF rate for each model: https://observablehq.com/@observablehq/plot-radar-chart
    """)
    return


@app.cell
def _(Plot, evals_all_filtered, js, make_radar_chart, mo, pd):
    # Plot D1: Radar chart — hint vs no-hint IF rate per model
    _d1_agg = (
        evals_all_filtered.groupby(["model", "instruction"])["score"]
        .mean()
        .reset_index()
    )
    _d1_hint = (
        _d1_agg[_d1_agg["instruction"] == "instruction_hint"]
        .rename(columns={"score": "value"})
        .assign(name="with hint")
    )
    _d1_nohint = (
        _d1_agg[_d1_agg["instruction"] == "instruction_no_hint"]
        .rename(columns={"score": "value"})
        .assign(name="no hint")
    )

    _categories = sorted(_d1_agg["model"].unique().tolist())
    _points = []
    for _, _row in pd.concat([_d1_nohint, _d1_hint]).iterrows():
        _points.append(
            {"key": _row["model"], "value": _row["value"], "name": _row["name"]}
        )

    mo.ui.anywidget(
        make_radar_chart(
            Plot, js, _points, _categories, "Effect of Hint on IF Rate"
        )
    )
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
        legend=alt.Legend(title="Instruction alignment"),
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
def _(evals_filtered):
    evals_filtered
    return


@app.cell
def _(Plot, alignment_curve_selector, evals_filtered, js, mo):
    # Plot D5: Paired transition curves — aligned vs misaligned per selected model
    _d5_data = evals_filtered[
        evals_filtered["condition_pair"] == "preference"
    ].copy()
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
                "title": "Aligned vs Misaligned: Transition Curves by Model",
            }
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Section 3.4 — Reasoning models
    - Plot R1: Transition curves (IF rate vs N) for reasoning vs non-reasoning variants, faceted by aligned vs misaligned condition. Shows both the overall improvement from reasoning and persistent alignment sensitivity.
    """)
    return


@app.cell
def _(all_models, mo):
    reasoning_model_selector = mo.ui.multiselect(
        options=sorted(all_models),
        value=[
            "gpt-5.2-medium",
            "gpt-5.2",
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
        .sort_values(["base_model", "reasoning_type", "alignment", "n_turns_int"])
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Follow-up conditions: variety and classification

    Three new conditions test what drives the task-based-condition advantage:
    - **Variety** (2 conditions): diverse 1–3 sentence responses on a fixed topic, ignoring the question. Tests whether response diversity alone confers resistance.
    - **Classification** (1 condition): single-token output (`science` or `humanities`), but requires reading the question. Tests whether question engagement alone confers resistance.

    Compared against baselines from the same 3 models:
    - **Fixed-output baseline**: `neutral` (single-token, no engagement)
    - **Task-based baseline**: avg of all task-based conditions (engagement + diversity)

    Data: 3 models (gemma-3-12b-it, gpt-5.2, llama-3.3-70b-instruct), no-hint only.
    """)
    return


@app.cell
def _(Path, pd):
    _root = Path(__file__).resolve().parent.parent.parent
    _followup_dir = _root / "outputs" / "viz" / "additional-dynamic"
    _static_dir = _root / "outputs" / "viz" / "static"
    _dynamic_dir = _root / "outputs" / "viz" / "dynamic"

    _followup_models = {"gemma-3-27b-it", "gpt-5.2", "llama-3.3-70b-instruct"}

    # Load followup data
    followup_evals = pd.read_parquet(_followup_dir / "evals.parquet")
    followup_evals = followup_evals[
        followup_evals["instruction"] == "instruction_no_hint"
    ].copy()
    followup_evals["n_turns_int"] = followup_evals["n_turns"].astype(int)

    # Load fixed-output baseline (neutral only, same 3 models)
    _static_all = pd.read_parquet(_static_dir / "evals.parquet")
    _static_baseline = _static_all[
        (_static_all["condition"] == "neutral")
        & (_static_all["instruction"] == "instruction_no_hint")
        & (_static_all["model"].isin(_followup_models))
    ].copy()
    _static_baseline["n_turns_int"] = _static_baseline["n_turns"].astype(int)

    # Load task-based baseline (all task-based conditions, same 3 models)
    _dynamic_all = pd.read_parquet(_dynamic_dir / "evals.parquet")
    _dynamic_baseline = _dynamic_all[
        (_dynamic_all["instruction"] == "instruction_no_hint")
        & (_dynamic_all["model"].isin(_followup_models))
    ].copy()
    _dynamic_baseline["n_turns_int"] = _dynamic_baseline["n_turns"].astype(int)

    # Build combined comparison dataframe with condition_group labels
    # Static baseline: avg across models per N → single "fixed-output (neutral)" line
    _sb_agg = (
        _static_baseline.groupby("n_turns_int")["score"]
        .mean()
        .reset_index(name="if_rate")
    )
    _sb_agg["condition_group"] = "fixed-output (neutral)"

    # Dynamic baseline: avg across conditions and models per N
    _db_agg = (
        _dynamic_baseline.groupby("n_turns_int")["score"]
        .mean()
        .reset_index(name="if_rate")
    )
    _db_agg["condition_group"] = "task-based (avg)"

    # Followup: avg across models per N, one line per condition
    _fu_agg = (
        followup_evals.groupby(["condition", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="if_rate")
    )

    # Map followup conditions to readable group labels
    _fu_label_map = {
        "classify_sh_economics": "classify-question",
        "variety_geography_animals": "random-facts-geography",
        "variety_animals_geography": "random-facts-animals",
    }
    _fu_agg["condition_group"] = _fu_agg["condition"].map(_fu_label_map)
    _fu_agg = _fu_agg[["n_turns_int", "if_rate", "condition_group"]]

    comparison_df = pd.concat(
        [_sb_agg, _db_agg, _fu_agg], ignore_index=True
    ).sort_values(["condition_group", "n_turns_int"])

    # Per-model version (for faceted F3)
    _sb_by_model = (
        _static_baseline.groupby(["model", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="if_rate")
    )
    _sb_by_model["condition_group"] = "fixed-output (neutral)"

    _db_by_model = (
        _dynamic_baseline.groupby(["model", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="if_rate")
    )
    _db_by_model["condition_group"] = "task-based (avg)"

    _fu_by_model = (
        followup_evals.groupby(["model", "condition", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="if_rate")
    )
    _fu_by_model["condition_group"] = _fu_by_model["condition"].map(_fu_label_map)
    _fu_by_model = _fu_by_model[
        ["model", "n_turns_int", "if_rate", "condition_group"]
    ]

    comparison_df_by_model = pd.concat(
        [_sb_by_model, _db_by_model, _fu_by_model], ignore_index=True
    ).sort_values(["model", "condition_group", "n_turns_int"])

    followup_boundaries = pd.read_parquet(_followup_dir / "_boundaries.parquet")
    return comparison_df, comparison_df_by_model, followup_evals


@app.cell
def _(Plot, comparison_df, js, mo):
    # F1: All conditions compared — followup vs fixed-output/task-based baselines (avg across models)
    _cmp_records = comparison_df.to_dict("records")

    _group_order = [
        "fixed-output (neutral)",
        "classify-question",
        # "random-facts-geography",
        "random-facts-animals",
        "task-based (avg)",
    ]
    _group_colors = [
        "#e41a1c",  # red — fixed-output
        "#ff7f00",  # orange — classify
        "#984ea3",  # purple — variety
        #   "#a65628",  # brown — variety
        "#4daf4a",  # green — task-based
    ]

    mo.ui.anywidget(
        Plot.plot(
            {
                "x": {"label": "N turns"},
                "y": {"domain": [0, 1], "label": "IF Rate (avg across 3 models)"},
                "color": {
                    "domain": _group_order,
                    "range": _group_colors,
                    "legend": True,
                },
                "marks": [
                    Plot.lineY(
                        _cmp_records,
                        {
                            "x": "n_turns_int",
                            "y": "if_rate",
                            "stroke": "condition_group",
                            "strokeWidth": 2.5,
                            "curve": "monotone-x",
                        },
                    ),
                    Plot.dot(
                        _cmp_records,
                        {
                            "x": "n_turns_int",
                            "y": "if_rate",
                            "fill": "condition_group",
                            "r": 3.5,
                            "title": js(
                                "d => `${d.condition_group}: N=${d.n_turns_int}, IF=${d.if_rate.toFixed(2)}`"
                            ),
                        },
                    ),
                    Plot.text(
                        _cmp_records,
                        Plot.selectLast(
                            {
                                "x": "n_turns_int",
                                "y": "if_rate",
                                "z": "condition_group",
                                "text": "condition_group",
                                "textAnchor": "start",
                                "dx": 6,
                                "fontSize": 9,
                                "fill": "condition_group",
                            }
                        ),
                    ),
                    Plot.ruleY(
                        [0.5], {"stroke": "#ccc", "strokeDasharray": "4 2"}
                    ),
                ],
                "width": 800,
                "height": 420,
                "marginRight": 200,
                "title": "Follow-up vs baselines: where do variety and classify fall?",
            }
        )
    )
    return


@app.cell
def _(Path, Plot, followup_evals, js, mo, pd):
    # F2: Per-model faceted view — followup conditions vs fixed-output & task-based baselines
    _root2 = Path(__file__).resolve().parent.parent.parent
    _followup_models2 = {"gemma-3-27b-it", "gpt-5.2", "llama-3.3-70b-instruct"}

    _static2 = pd.read_parquet(
        _root2 / "outputs" / "viz" / "static" / "evals.parquet"
    )
    _static2 = _static2[
        (_static2["condition"] == "neutral")
        & (_static2["instruction"] == "instruction_no_hint")
        & (_static2["model"].isin(_followup_models2))
    ].copy()
    _static2["n_turns_int"] = _static2["n_turns"].astype(int)
    _static2["condition_group"] = "fixed-output (neutral)"

    _dynamic2 = pd.read_parquet(
        _root2 / "outputs" / "viz" / "dynamic" / "evals.parquet"
    )
    _dynamic2 = _dynamic2[
        (_dynamic2["instruction"] == "instruction_no_hint")
        & (_dynamic2["model"].isin(_followup_models2))
    ].copy()
    _dynamic2["n_turns_int"] = _dynamic2["n_turns"].astype(int)
    # Average across task-based conditions per model per N
    _dyn_avg = (
        _dynamic2.groupby(["model", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="score")
    )
    _dyn_avg["condition_group"] = "task-based (avg)"

    _fu2 = followup_evals.copy()
    _fu_label_map2 = {
        "classify_sh_economics": "classify-question",
        "variety_geography_animals": "random-facts-geography",
        "variety_animals_geography": "random-facts-animals",
    }
    _fu2["condition_group"] = _fu2["condition"].map(_fu_label_map2)

    _combined = pd.concat(
        [
            _static2[["model", "n_turns_int", "score", "condition_group"]],
            _dyn_avg[["model", "n_turns_int", "score", "condition_group"]],
            _fu2[["model", "n_turns_int", "score", "condition_group"]],
        ],
        ignore_index=True,
    )

    _f2_agg = (
        _combined.groupby(["model", "condition_group", "n_turns_int"])["score"]
        .mean()
        .reset_index(name="if_rate")
        .sort_values(["model", "condition_group", "n_turns_int"])
    )
    _f2_records = _f2_agg.to_dict("records")

    _grp_order2 = [
        "fixed-output (neutral)",
        "classify-question",
        "random-facts-geography",
        "random-facts-animals",
        "task-based (avg)",
    ]
    _grp_colors2 = ["#e41a1c", "#ff7f00", "#984ea3", "#a65628", "#4daf4a"]

    mo.ui.anywidget(
        Plot.plot(
            {
                "x": {"label": "N turns"},
                "y": {"domain": [0, 1], "label": "IF Rate"},
                "fx": {"label": None},
                "color": {
                    "domain": _grp_order2,
                    "range": _grp_colors2,
                    "legend": True,
                },
                "marks": [
                    Plot.lineY(
                        _f2_records,
                        {
                            "x": "n_turns_int",
                            "y": "if_rate",
                            "stroke": "condition_group",
                            "fx": "model",
                            "strokeWidth": 2,
                            "opacity": 0.5,
                            "curve": "monotone-x",
                        },
                    ),
                    Plot.dot(
                        _f2_records,
                        {
                            "x": "n_turns_int",
                            "y": "if_rate",
                            "fill": "condition_group",
                            "fx": "model",
                            "r": 3,
                            "title": js(
                                "d => `${d.condition_group}: N=${d.n_turns_int}, IF=${d.if_rate.toFixed(2)}`"
                            ),
                        },
                    ),
                    Plot.ruleY(
                        [0.5], {"stroke": "#ccc", "strokeDasharray": "4 2"}
                    ),
                ],
                "width": 1050,
                "height": 350,
                "marginLeft": 50,
                "title": "Follow-up vs baselines per model",
            }
        )
    )
    return


@app.cell
def _(alt, comparison_df_by_model, plots_dir):
    # F3: Bar chart — avg IF rate per condition group, faceted by model
    _f3_agg = (
        comparison_df_by_model.groupby(["model", "condition_group"])["if_rate"]
        .mean()
        .reset_index(name="avg_if")
    )
    _f3_agg["group_type"] = _f3_agg["condition_group"].apply(
        lambda g: (
            "baseline" if "fixed-output" in g or "task-based" in g else "followup"
        )
    )
    _f3_order = [
        "fixed-output (neutral)",
        "classify (engagement only)",
        "variety: geo→animals",
        "variety: animals→geo",
        "task-based (avg)",
    ]

    _bars = (
        alt.Chart(_f3_agg)
        .mark_bar()
        .encode(
            x=alt.X(
                "avg_if:Q",
                title="Mean IF Rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("condition_group:N", sort=_f3_order, title=None),
            color=alt.Color(
                "group_type:N",
                scale=alt.Scale(
                    domain=["baseline", "followup"],
                    range=["#999999", "#ff7f00"],
                ),
                legend=alt.Legend(title="Type"),
            ),
            tooltip=[
                "model",
                "condition_group",
                alt.Tooltip("avg_if:Q", format=".3f", title="Mean IF Rate"),
            ],
        )
        .facet(
            facet=alt.Facet("model:N", title=None),
            columns=3,
        )
        .properties(title="Mean IF rate by condition group, per model")
    )
    _bars.save(str(plots_dir / "f3_followup_by_model.png"), scale_factor=2)
    _bars
    return


if __name__ == "__main__":
    app.run()
