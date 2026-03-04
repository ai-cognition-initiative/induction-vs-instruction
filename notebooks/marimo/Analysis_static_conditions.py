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
        BENCHMARKS,
        SHAPE_SCALE,
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
    # Analysis of induction vs prediction experiments - Static condition

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
    _static = _root / "outputs" / "viz" / "static"

    _exclude_conditions = []
    # ["value_aligned_helpful", "value_misaligned_helpful"]

    evals_all = pd.read_parquet(_static / "evals.parquet")
    evals_all = evals_all[~evals_all["condition"].isin(_exclude_conditions)].copy()
    evals = evals_all[evals_all["instruction"] == "instruction_no_hint"].copy()
    evals["n_turns_int"] = evals["n_turns"].astype(int)

    combined_errors_all = pd.read_parquet(_static / "_combined_errors.parquet")
    combined_errors_all = combined_errors_all[
        ~combined_errors_all["condition"].isin(_exclude_conditions)
    ].copy()
    combined_errors = combined_errors_all[
        combined_errors_all["instruction"] == "instruction_no_hint"
    ].copy()
    combined_errors["n_turns_int"] = combined_errors["n_turns"].astype(int)

    evals_prediction_all = pd.read_parquet(_static / "evals_prediction.parquet")
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
        #  "gpt-5.2",
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
        value=[],
        label="Models to exclude",
    )
    model_exclusion
    return (model_exclusion,)


@app.cell
def _(
    caps_df,
    combined_errors,
    evals,
    evals_all,
    evals_prediction,
    model_exclusion,
):
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
    combined_errors_filtered = (
        combined_errors[~combined_errors["model"].isin(_excluded)].copy()
        if _excluded
        else combined_errors.copy()
    )
    evals_prediction_filtered = (
        evals_prediction[~evals_prediction["model"].isin(_excluded)].copy()
        if _excluded
        else evals_prediction.copy()
    )
    caps_df_filtered = (
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
                "marginLeft": 250,
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
def _(alt, evals_all_filtered, instruction_dropdown):
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
            color=alt.Color("model:N", legend=None),
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
    )
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
                "title": "Mean Calibration: Actual vs Predicted IF Rate",
            }
        )
    )
    return


@app.cell
def _(
    caps_df_filtered,
    combined_errors_filtered,
    make_scatter_chart,
    prep_benchmark_data,
    reasoning_models,
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
        reasoning_models,
        lambda *a: a[0],
    )
    make_scatter_chart(
        _b2_df,
        "mean_abs_calibration_error",
        "Mean |Calibration Error| (%)",
        "Absolute Calibration Error vs Model Capability",
    )
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
def _(alt, combined_errors_filtered, pd):
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
            color=alt.Color("model:N", legend=None),
        )
    )
    (_box_b4 + _rule_b4).properties(
        width=500, height=400, title="Distribution of Calibration Error by Model"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction analysis
    - Plot C1: avg "predicts instruction following" rate by model and capability, like plots A2 and B2
    """)
    return


@app.cell
def _(
    caps_df_filtered,
    evals_prediction_filtered,
    make_scatter_chart,
    prep_benchmark_data,
    reasoning_models,
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
        reasoning_models,
        lambda *a: a[0],
    )
    make_scatter_chart(
        _c1_df,
        "prediction_rate",
        "Avg 'Predicts IF' Rate",
        "Prediction of IF Rate vs Model Capability",
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
    ## Effect of prediction
    - Plot D2: radar plot with IF rate without prediction overlapped with IF rate AFTER prediction for each model: https://observablehq.com/@observablehq/plot-radar-chart
    """)
    return


@app.cell
def _(Plot, combined_errors_filtered, js, make_radar_chart, mo):
    # Plot D2: Radar chart — IF rate without prediction vs after prediction
    _d2_agg = (
        combined_errors_filtered.groupby("model")
        .agg(
            behavioral=("behavioral_score", "mean"),
            after_prediction=("prediction_actual_score", "mean"),
        )
        .reset_index()
    )

    _categories = sorted(_d2_agg["model"].unique().tolist())
    _points = []
    for _, _row in _d2_agg.iterrows():
        _points.append(
            {
                "key": _row["model"],
                "value": _row["behavioral"],
                "name": "behavioral (no prediction)",
            }
        )
        _points.append(
            {
                "key": _row["model"],
                "value": _row["after_prediction"],
                "name": "after prediction",
            }
        )

    mo.ui.anywidget(
        make_radar_chart(
            Plot, js, _points, _categories, "Effect of Prediction on IF Rate"
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


if __name__ == "__main__":
    app.run()
