import marimo

__generated_with = "0.20.2"
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
        LIKERT_OFFSET_JS,
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
    # Analysis of induction vs prediction experiments - Dynamic condition

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
    _dynamic = _root / "outputs" / "viz" / "dynamic"

    evals_all = pd.read_parquet(_dynamic / "evals.parquet")
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
            "intelligence_index": scores.get("intelligence_index"),
            "mmlu_pro": scores.get("mmlu_pro"),
            "gpqa": scores.get("gpqa"),
            "ifbench": scores.get("ifbench"),
        }


    caps_df = pd.DataFrame([_get_caps(k, v) for k, v in _caps_raw.items()])
    caps_df = caps_df.dropna(
        how="all", subset=["intelligence_index", "mmlu_pro", "gpqa", "ifbench"]
    )

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
        value=["hermes-4-70b-reasoning", "gpt-5.2-medium"],
        label="Models to exclude",
    )
    model_exclusion
    return (model_exclusion,)


@app.cell
def _(caps_df, evals, evals_all, model_exclusion):
    _excluded = model_exclusion.value
    evals_filtered = (
        evals[~evals["model"].isin(_excluded)].copy()
        if _excluded
        else evals.copy()
    )
    evals_all_filtered = (
        evals_all[~evals_all["model"].isin(_excluded)].copy()
        if _excluded
        else evals_all.copy()
    )
    caps_df_filtered = (
        caps_df[~caps_df["model"].isin(_excluded)].copy()
        if _excluded
        else caps_df.copy()
    )
    return caps_df_filtered, evals_all_filtered, evals_filtered


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
    LIKERT_OFFSET_JS,
    Plot,
    evals_filtered,
    js,
    mo,
    n_values_sorted,
    threshold_slider,
):
    # Plot A1: Diverging stacked bar — N values classified as IF/PF/Mixed
    _threshold = threshold_slider.value
    _avg = (
        evals_filtered.groupby(["model", "n_turns"])["score"].mean().reset_index()
    )

    # Expand each N value into a unit row with its classification
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

    mo.ui.anywidget(
        Plot.plot(
            {
                "x": {
                    "label": f"← PF-dominant · Number of N values (out of {_n_total}) · IF-dominant →",
                    "tickFormat": js("Math.abs"),
                },
                "y": {"label": None},
                "color": {
                    "domain": CATEGORY_ORDER,
                    "range": CATEGORY_COLORS,
                    "legend": True,
                },
                "marks": [
                    Plot.barX(
                        _unit_records,
                        Plot.groupY(
                            {"x": "count"},
                            {
                                "y": "model",
                                "fill": "category",
                                "order": CATEGORY_ORDER,
                                "offset": js(LIKERT_OFFSET_JS),
                                "sort": {"y": "-x"},
                            },
                        ),
                    ),
                    Plot.ruleX([0]),
                ],
                "width": 700,
                "height": 450,
                "marginLeft": 160,
                "title": f"N-value classification (threshold={_threshold})",
            }
        )
    )
    return


@app.cell
def _(
    caps_df_filtered,
    evals_filtered,
    make_scatter_chart,
    prep_benchmark_data,
    reasoning_models,
):
    # Plot A2: Scatter — avg IF rate vs capability (faceted by benchmark)
    _avg_if = (
        evals_filtered.groupby("model")["score"]
        .mean()
        .reset_index(name="avg_if_rate")
    )
    _a2_df = prep_benchmark_data(
        _avg_if, "avg_if_rate", caps_df_filtered, reasoning_models, lambda *a: a[0]
    )
    make_scatter_chart(
        _a2_df, "avg_if_rate", "Avg IF Rate", "Avg IF Rate vs Model Capability"
    )
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
def _(COLOR_SCHEME, alt, evals_all_filtered, instruction_dropdown):
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
    (
        (_bars + _error)
        .facet(facet=alt.Facet("condition:N", title="Condition"), columns=2)
        .resolve_scale(y="independent")
        .properties(title=f"IF Rate by Model and Condition ({_sel})")
    )
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
    prep_benchmark_data,
    reasoning_models,
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
        _drop_df, "first_drop_n", caps_df_filtered, reasoning_models, nudge_labels
    )
    make_scatter_chart(
        _a5_df,
        "first_drop_n",
        f"First N where IF <= {_threshold_a5}",
        f"First IF Drop (threshold={_threshold_a5}) vs Model Capability",
        log_y=True,
    )
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
def _(alt, evals_filtered):
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
    (_d4_bars + _d4_err).properties(
        width=450,
        height=500,
        title="Avg IF Rate: Aligned vs Misaligned Instructions (sorted by gap)",
    )
    return


@app.cell
def _(mo):
    benchmark_dropdown = mo.ui.dropdown(
        options={
            "Intelligence Index": "intelligence_index",
            "MMLU-Pro": "mmlu_pro",
            "GPQA": "gpqa",
            "IFBench": "ifbench",
        },
        value="Intelligence Index",
        label="Capability metric (color)",
    )
    benchmark_dropdown
    return (benchmark_dropdown,)


@app.cell
def _(alt, benchmark_dropdown, caps_df_filtered, evals_filtered, pd):
    # Plot D6: Quadrant scatter — aligned vs misaligned IF rate, color = capability
    _bm = benchmark_dropdown.value
    _bm_label = {
        "intelligence_index": "Intelligence Index",
        "mmlu_pro": "MMLU-Pro",
        "gpqa": "GPQA",
        "ifbench": "IFBench",
    }[_bm]

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
    (_diag_line + _pts + _labels).properties(
        width=450,
        height=450,
        title=f"Aligned vs Misaligned IF Rate (color = {_bm_label})",
    )
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
def _(
    Plot,
    evals_filtered,
    js,
    mo,
    reasoning_model_selector,
    reasoning_models,
):
    # Plot R1: Reasoning vs non-reasoning transition curves, faceted by alignment
    _selected = reasoning_model_selector.value
    _r1_data = evals_filtered[evals_filtered["model"].isin(_selected)].copy()
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


if __name__ == "__main__":
    app.run()
