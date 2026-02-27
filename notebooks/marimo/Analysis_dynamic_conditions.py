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
        make_scatter_chart,
        mo,
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

    combined_errors_all = pd.read_parquet(_dynamic / "_combined_errors.parquet")
    combined_errors = combined_errors_all[
        combined_errors_all["instruction"] == "instruction_no_hint"
    ].copy()
    combined_errors["n_turns_int"] = combined_errors["n_turns"].astype(int)

    with open(_root / "data" / "model_capability_scores.json") as _f:
        _caps_raw = json.load(_f)

    reasoning_models = {
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
      #  "gpt-5.2", not in this case
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
    return (
        caps_df,
        combined_errors,
        evals,
        evals_all,
        n_values_sorted,
        reasoning_models,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavioral analysis
    - Plot A1 - stacked bar plot (Altair). For each model on the x axis, show three stacks: number of N values with IF >= THRESHOLD, number of N values with PF >= THRESHOLD, remaining N values in the dataset. Average across conditions.THRESHOLD should be set interactively in the notebook, with a default of 0.8.
    - Plot A2 - avg IF rate vs capability. Compute average IF rate and make a scatterplot of (IF rate, intelligence_index). Color is model.
    - Plot A3 - horizontal bar chart with avg IF rate + error bars to show uncertainty. Error bars come from the stderr metric in the dataframe. Faceted by condition (col size 3 or 4), y shows models. x axis should be fixed from 0 to 1. Add dropdown to select which instruction_template to show
    - Plot A4: vertical colored stripes of IF rate for increasing N: similar to https://observablehq.com/@observablehq/plot-stacked-unit-chart. But here the data is sorted by N, not by the value of the data itself. The quantity to plot is again the mean IF rate across condition per model (models on the y axis). Each stacked unit is the IF rate for one N value per increasing N, with color showing the average IF rate.
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
    evals,
    js,
    mo,
    n_values_sorted,
    threshold_slider,
):
    # Plot A1: Diverging stacked bar — N values classified as IF/PF/Mixed
    _threshold = threshold_slider.value
    _avg = evals.groupby(["model", "n_turns"])["score"].mean().reset_index()

    # Expand each N value into a unit row with its classification
    _unit_records = []
    for _model in _avg["model"].unique():
        _model_df = _avg[_avg["model"] == _model].sort_values("n_turns", key=lambda s: s.astype(int))
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
        Plot.plot({
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
        })
    )
    return


@app.cell
def _(
    caps_df,
    evals,
    make_scatter_chart,
    prep_benchmark_data,
    reasoning_models,
):
    # Plot A2: Scatter — avg IF rate vs capability (faceted by benchmark)
    _avg_if = evals.groupby("model")["score"].mean().reset_index(name="avg_if_rate")
    _a2_df = prep_benchmark_data(
        _avg_if, "avg_if_rate", caps_df, reasoning_models, lambda *a: a[0]
    )
    make_scatter_chart(
        _a2_df, "avg_if_rate", "Avg IF Rate", "Avg IF Rate vs Model Capability"
    )
    return


@app.cell
def _(caps_df):
    caps_df
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
def _(alt, evals_all, instruction_dropdown):
    # Plot A3: Horizontal bar + error bars, faceted by condition
    _sel = instruction_dropdown.value
    _a3_data = evals_all[evals_all["instruction"] == _sel].copy()
    _a3_agg = (
        _a3_data.groupby(["model", "condition"])
        .agg(mean_score=("score", "mean"), mean_stderr=("score_stderr", "mean"))
        .reset_index()
    )
    _a3_agg["ci_lo"] = (_a3_agg["mean_score"] - _a3_agg["mean_stderr"]).clip(lower=0)
    _a3_agg["ci_hi"] = (_a3_agg["mean_score"] + _a3_agg["mean_stderr"]).clip(upper=1)

    _bars = (
        alt.Chart(_a3_agg)
        .mark_bar()
        .encode(
            x=alt.X("mean_score:Q", title="IF Rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", sort="-x", title=None),
            color=alt.Color("model:N", legend=None),
            tooltip=["model", "condition", "mean_score:Q", "mean_stderr:Q"],
        )
    )
    _error = (
        alt.Chart(_a3_agg)
        .mark_errorbar()
        .encode(
            x=alt.X("ci_lo:Q", title=""), x2="ci_hi:Q", y=alt.Y("model:N", sort="-x")
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
def _(Plot, evals, js, mo):
    # Plot A4: Stacked unit chart — colored stripes of IF rate by N
    _a4_agg = (
        evals.groupby(["model", "n_turns", "n_turns_int"])["score"]
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
                "width": 700,
                "height": 450,
                "marginLeft": 160,
            }
        )
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
def _(Plot, combined_errors, js, mo):
    # Plot B1: Arrow plot — mean actual vs predicted IF rate per model
    _b1_agg = (
        combined_errors.groupby("model")
        .agg(
            actual=("behavioral_score", "mean"),
            predicted=("prediction_predicted_score", "mean"),
        )
        .reset_index()
    )
    _b1_agg["direction"] = _b1_agg.apply(
        lambda r: "over-predicts" if r["predicted"] > r["actual"] else "under-predicts",
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
                            "title": js("d => `Predicted: ${d.predicted.toFixed(3)}`"),
                        },
                    ),
                    Plot.ruleX([0.5], {"stroke": "#ccc", "strokeDasharray": "4 2"}),
                ],
                "width": 600,
                "height": 200,
                "marginLeft": 160,
                "title": "Mean Calibration: Actual vs Predicted IF Rate",
            }
        )
    )
    return


@app.cell
def _(
    caps_df,
    combined_errors,
    make_scatter_chart,
    prep_benchmark_data,
    reasoning_models,
):
    # Plot B2: Scatter — mean calibration error vs capability (faceted by benchmark)
    _b2_agg = (
        combined_errors.groupby("model")["calibration_error_pct"]
        .mean()
        .reset_index(name="mean_calibration_error")
    )
    _b2_df = prep_benchmark_data(
        _b2_agg, "mean_calibration_error", caps_df, reasoning_models, lambda *a: a[0]
    )
    make_scatter_chart(
        _b2_df,
        "mean_calibration_error",
        "Mean Calibration Error (%)",
        "Calibration Error vs Model Capability",
    )
    return


@app.cell
def _(Plot, combined_errors, js, mo):
    # Plot B3: Trellis arrows — actual vs predicted by condition (faceted)
    _b3_agg = (
        combined_errors.groupby(["model", "condition"])
        .agg(
            actual=("behavioral_score", "mean"),
            predicted=("prediction_predicted_score", "mean"),
        )
        .reset_index()
    )
    _b3_agg["direction"] = _b3_agg.apply(
        lambda r: "over-predicts" if r["predicted"] > r["actual"] else "under-predicts",
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
                            "title": js("d => `Predicted: ${d.predicted.toFixed(3)}`"),
                        },
                    ),
                ],
                "width": 600,
                "height": 800,
                "marginLeft": 160,
                "marginRight": 160,
                "title": "Calibration by Condition: Actual vs Predicted IF Rate",
            }
        )
    )
    return


@app.cell
def _(alt, combined_errors, pd):
    # Plot B4: Box plot — distribution of calibration error per model
    _rule_b4 = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(strokeDash=[4, 2], color="gray")
        .encode(x="x:Q")
    )
    _b4_data = combined_errors.dropna(subset=["calibration_error_pct"])
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


if __name__ == "__main__":
    app.run()
