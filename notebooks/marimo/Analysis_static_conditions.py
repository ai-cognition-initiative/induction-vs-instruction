import marimo

__generated_with = "0.20.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/Analysis_static_conditions.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    import json
    from pathlib import Path
    from pyobsplot import Plot, js

    return Path, Plot, alt, js, json, mo, pd


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
def _(Path, Plot, alt, js, json, pd):
    _root = Path(__file__).resolve().parent.parent.parent
    _static = _root / "outputs" / "viz" / "static"

    evals_all = pd.read_parquet(_static / "evals.parquet")
    evals = evals_all[evals_all["instruction"] == "instruction_no_hint"].copy()
    evals["n_turns_int"] = evals["n_turns"].astype(int)

    combined_errors_all = pd.read_parquet(_static / "_combined_errors.parquet")
    combined_errors = combined_errors_all[
        combined_errors_all["instruction"] == "instruction_no_hint"
    ].copy()
    combined_errors["n_turns_int"] = combined_errors["n_turns"].astype(int)

    evals_prediction_all = pd.read_parquet(_static / "evals_prediction.parquet")
    evals_prediction = evals_prediction_all[
        evals_prediction_all["instruction"] == "instruction_no_hint"
    ].copy()
    evals_prediction["n_turns_int"] = evals_prediction["n_turns"].astype(int)

    with open(_root / "data" / "model_capability_scores.json") as _f:
        _caps_raw = json.load(_f)

    reasoning_models = {
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
        "gpt-5.2",
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


    def nudge_labels(
        df, x_col, y_col, x_range, y_range, pad_x=0.02, pad_y=0.03, iters=50
    ):
        """Iterative repulsion to separate overlapping text labels."""
        out = df.copy()
        x_span = x_range[1] - x_range[0] or 1
        y_span = y_range[1] - y_range[0] or 1
        lx = ((out[x_col] - x_range[0]) / x_span).values.astype(float)
        ly = ((out[y_col] - y_range[0]) / y_span).values.astype(float)
        lx = lx + pad_x
        n = len(lx)
        for _ in range(iters):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = lx[i] - lx[j]
                    dy = ly[i] - ly[j]
                    dist = (dx**2 + dy**2) ** 0.5
                    if dist < pad_y:
                        force = (pad_y - dist) / 2
                        if dist > 0:
                            lx[i] += force * dx / dist * 0.3
                            ly[i] += force * dy / dist
                            lx[j] -= force * dx / dist * 0.3
                            ly[j] -= force * dy / dist
                        else:
                            ly[i] += force
                            ly[j] -= force
        out["label_x"] = lx * x_span + x_range[0]
        out["label_y"] = ly * y_span + y_range[0]
        return out


    BENCHMARKS = {
        "intelligence_index": "Intelligence Index",
        "mmlu_pro": "MMLU Pro",
        "gpqa": "GPQA",
        "ifbench": "IFBench",
    }
    SHAPE_SCALE = alt.Scale(
        domain=["non-reasoning", "reasoning"], range=["circle", "diamond"]
    )


    def prep_benchmark_data(df, y_col, caps_df, reasoning_models, nudge_labels):
        """Prepare data for benchmark scatter plots."""
        _wide = df.merge(caps_df, on="model", how="inner")
        _wide["reasoning"] = (
            _wide["model"]
            .isin(reasoning_models)
            .map({True: "reasoning", False: "non-reasoning"})
        )
        parts = []
        for _col, _label in BENCHMARKS.items():
            _part = (
                _wide[["model", y_col, "reasoning", _col]]
                .dropna(subset=[_col])
                .copy()
            )
            if len(_part) == 0:
                continue
            _part = _part.rename(columns={_col: "benchmark_value"})
            _part["benchmark"] = _label
            _xr = [
                _part["benchmark_value"].min() * 0.95,
                _part["benchmark_value"].max() * 1.05,
            ]
            _yr = [0, 1] if y_col in ["avg_if_rate", "prediction_rate"] else None
            if _yr:
                _part = nudge_labels(_part, "benchmark_value", y_col, _xr, _yr)
            else:
                _yr = [_part[y_col].min() - 5, _part[y_col].max() + 5]
                _part = nudge_labels(_part, "benchmark_value", y_col, _xr, _yr)
            parts.append(_part)
        result = pd.concat(parts, ignore_index=True)
        return result.where(pd.notnull(result), None)


    def make_scatter_chart(df, y_col, y_title, title):
        """Create a faceted scatter chart with labels and trendline."""
        points = (
            alt.Chart(df)
            .mark_point(size=100, filled=True)
            .encode(
                x=alt.X(
                    "benchmark_value:Q",
                    title="Benchmark Score",
                    scale=alt.Scale(zero=False),
                ),
                y=alt.Y(f"{y_col}:Q", title=y_title),
                color=alt.Color("model:N", legend=None),
                shape=alt.Shape(
                    "reasoning:N", scale=SHAPE_SCALE, title="Model type"
                ),
                tooltip=[
                    "model",
                    f"{y_col}:Q",
                    "benchmark_value:Q",
                    "benchmark:N",
                    "reasoning:N",
                ],
            )
        )
        trendline = (
            alt.Chart(df)
            .transform_regression("benchmark_value", y_col)
            .mark_line(color="gray", strokeDash=[4, 2])
            .encode(x="benchmark_value:Q", y=f"{y_col}:Q")
        )
        leaders = (
            alt.Chart(df)
            .mark_rule(strokeWidth=0.5, color="#999")
            .encode(
                x="benchmark_value:Q",
                y=f"{y_col}:Q",
                x2="label_x:Q",
                y2="label_y:Q",
            )
        )
        text = (
            alt.Chart(df)
            .mark_text(align="left", dx=3, fontSize=9)
            .encode(x="label_x:Q", y="label_y:Q", text="model:N")
        )
        layers = points + trendline + leaders + text
        if y_col == "mean_calibration_error":
            rule = (
                alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(strokeDash=[4, 2], color="gray")
                .encode(y="y:Q")
            )
            layers = layers + rule
        return (
            layers.facet(facet=alt.Facet("benchmark:N", title=None), columns=2)
            .resolve_scale(x="independent")
            .properties(title=title)
        )

    def make_radar_chart(points, categories, title, width=500, height=500):
        """Create a radar chart using pyobsplot (Observable Plot).

        Args:
            points: list of dicts with "key" (category/model), "value" (0-1 IF rate), "name" (group label)
            categories: list of axis labels (model names)
            title: chart title
            width: chart width
            height: chart height

        Returns:
            A Plot.plot(...) widget (caller wraps in mo.ui.anywidget).
        """
        _cats_json = json.dumps(categories)
        _lon_js = f"d3.scalePoint({_cats_json}, [180, -180]).padding(0.5).align(1)"

        return Plot.plot({
            "projection": {
                "type": "azimuthal-equidistant",
                "rotate": [0, -90],
                "domain": js("d3.geoCircle().center([0, 90]).radius(0.625)()"),
            },
            "marks": [
                # Grid circles at radii mapping to 0%, 20%, 40%, 60%, 80%, 100%
                Plot.geo([0.5, 0.4, 0.3, 0.2, 0.1], {
                    "geometry": js("(r) => d3.geoCircle().center([0, 90]).radius(r)()"),
                    "stroke": "currentColor",
                    "strokeOpacity": 0.2,
                }),
                # Spokes
                Plot.link(js(f"({_lon_js}).domain()"), {
                    "x1": js(f"(d) => ({_lon_js})(d)"),
                    "y1": 90,
                    "x2": 0,
                    "y2": 90,
                    "stroke": "currentColor",
                    "strokeOpacity": 0.2,
                }),
                # Axis labels (model names)
                Plot.text(js(f"({_lon_js}).domain()"), {
                    "x": js(f"(d) => ({_lon_js})(d)"),
                    "y": 89.4,
                    "text": js("Plot.identity"),
                    "lineWidth": 5,
                }),
                # Grid labels (percentages)
                Plot.text([0.5, 0.4, 0.3, 0.2, 0.1], {
                    "x": 180,
                    "y": js("(r) => 90 - r"),
                    "dx": 2,
                    "textAnchor": "start",
                    "text": js("(r) => `${r * 200}%`"),
                    "fill": "currentColor",
                    "stroke": "white",
                    "fontSize": 8,
                }),
                # Filled area per group
                Plot.area(points, {
                    "x1": js(f"(d) => ({_lon_js})(d.key)"),
                    "y1": js("(d) => 90 - d.value * 0.5"),
                    "fill": "name",
                    "stroke": "name",
                    "curve": "cardinal-closed",
                    "fillOpacity": 0.2,
                    "strokeWidth": 1.5,
                }),
                # Points
                Plot.dot(points, {
                    "x": js(f"(d) => ({_lon_js})(d.key)"),
                    "y": js("(d) => 90 - d.value * 0.5"),
                    "fill": "name",
                    "stroke": "white",
                    "r": 3,
                }),
                # Tooltips
                Plot.text(points, Plot.pointer({
                    "x": js(f"(d) => ({_lon_js})(d.key)"),
                    "y": js("(d) => 90 - d.value * 0.5"),
                    "text": js("(d) => `${d.name}: ${(d.value * 100).toFixed(1)}%`"),
                    "fill": "currentColor",
                    "stroke": "white",
                    "fontSize": 10,
                    "dy": -10,
                })),
            ],
            "width": width,
            "height": height,
            "title": title,
            "color": {"legend": True},
        })

    return (
        caps_df,
        combined_errors,
        evals,
        evals_all,
        evals_prediction,
        make_radar_chart,
        make_scatter_chart,
        n_values_sorted,
        nudge_labels,
        prep_benchmark_data,
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
def _(Plot, evals, js, mo, n_values_sorted, threshold_slider):
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

    # Likert-style mapping: IF=1 (right), Mixed=0 (center), PF=-1 (left)
    _order = ["PF-dominant", "Mixed", "IF-dominant"]
    _likert_map = {"PF-dominant": -1, "Mixed": 0, "IF-dominant": 1}
    _order_json = js(repr(_order))
    _likert_map_json = js(f"new Map({list(_likert_map.items())})")

    _n_total = len(n_values_sorted)

    mo.ui.anywidget(
        Plot.plot({
            "x": {
                "label": f"← PF-dominant · Number of N values (out of {_n_total}) · IF-dominant →",
                "tickFormat": js("Math.abs"),
            },
            "y": {"label": None},
            "color": {
                "domain": _order,
                "range": ["#d62728", "#999999", "#2ca02c"],
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
                            "order": _order,
                            "offset": js("""(faceI, X1, X2, Z) => {
                                const map = new Map([["PF-dominant", -1], ["Mixed", 0], ["IF-dominant", 1]]);
                                for (const stacks of faceI) {
                                    for (const stack of stacks) {
                                        const k = d3.sum(stack, (i) => (X2[i] - X1[i]) * (1 - map.get(Z[i]))) / 2;
                                        for (const i of stack) {
                                            X1[i] -= k;
                                            X2[i] -= k;
                                        }
                                    }
                                }
                            }"""),
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
    _avg_if = (
        evals.groupby("model")["score"].mean().reset_index(name="avg_if_rate")
    )
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


@app.cell
def _(mo):
    threshold_slider_a5 = mo.ui.slider(
        start=0.1, stop=0.9, value=0.5, step=0.05, label="IF drop threshold (A5)"
    )
    threshold_slider_a5
    return (threshold_slider_a5,)


@app.cell
def _(
    caps_df,
    evals,
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
        evals.groupby(["model", "n_turns_int"])["score"].mean().reset_index()
    )
    _max_n = max(n_values_sorted, key=int)
    _sentinel = int(_max_n) + 5

    _records_a5 = []
    for _model in _avg_a5["model"].unique():
        _mdf = _avg_a5[_avg_a5["model"] == _model].sort_values("n_turns_int")
        _first = _mdf[_mdf["score"] <= _threshold_a5]
        _drop_n = int(_first["n_turns_int"].iloc[0]) if len(_first) > 0 else _sentinel
        _records_a5.append({"model": _model, "first_drop_n": _drop_n})

    _drop_df = pd.DataFrame(_records_a5)
    _a5_df = prep_benchmark_data(
        _drop_df, "first_drop_n", caps_df, reasoning_models, nudge_labels
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
        _b2_agg,
        "mean_calibration_error",
        caps_df,
        reasoning_models,
        lambda *a: a[0],
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
                "width": 800,
                "height": 1300,
                "marginLeft": 120,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction analysis
    - Plot C1: avg "predicts instruction following" rate by model and capability, like plots A2 and B2
    """)
    return


@app.cell
def _(
    caps_df,
    evals_prediction,
    make_scatter_chart,
    prep_benchmark_data,
    reasoning_models,
):
    # Plot C1: Scatter — avg prediction of IF rate vs capability (faceted by benchmark)
    _c1_agg = (
        evals_prediction.groupby("model")["score_prediction_instruction"]
        .mean()
        .reset_index(name="prediction_rate")
    )
    _c1_df = prep_benchmark_data(
        _c1_agg, "prediction_rate", caps_df, reasoning_models, lambda *a: a[0]
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
def _(evals_all, make_radar_chart, mo, pd):
    # Plot D1: Radar chart — hint vs no-hint IF rate per model
    _d1_agg = (
        evals_all.groupby(["model", "instruction"])["score"].mean().reset_index()
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
        _points.append({"key": _row["model"], "value": _row["value"], "name": _row["name"]})

    mo.ui.anywidget(make_radar_chart(_points, _categories, "Effect of Hint on IF Rate"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of prediction
    - Plot D2: radar plot with IF rate without prediction overlapped with IF rate AFTER prediction for each model: https://observablehq.com/@observablehq/plot-radar-chart
    """)
    return


@app.cell
def _(combined_errors, make_radar_chart, mo):
    # Plot D2: Radar chart — IF rate without prediction vs after prediction
    _d2_agg = (
        combined_errors.groupby("model")
        .agg(
            behavioral=("behavioral_score", "mean"),
            after_prediction=("prediction_actual_score", "mean"),
        )
        .reset_index()
    )

    _categories = sorted(_d2_agg["model"].unique().tolist())
    _points = []
    for _, _row in _d2_agg.iterrows():
        _points.append({"key": _row["model"], "value": _row["behavioral"], "name": "behavioral (no prediction)"})
        _points.append({"key": _row["model"], "value": _row["after_prediction"], "name": "after prediction"})

    mo.ui.anywidget(make_radar_chart(_points, _categories, "Effect of Prediction on IF Rate"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of aligned/misaligned
    - Plot D3: radar plot with IF rate for aligned conditions overlapped with IF rate of unaligned conditions for each model, but faceted by specific condition couple (the cats one and the earth round/flat one): https://observablehq.com/@observablehq/plot-radar-chart https://observablehq.com/@observablehq/plot-radar-chart-faceted
    """)
    return


@app.cell
def _(evals, make_radar_chart, mo):
    # Plot D3: Radar chart per condition_pair — aligned vs misaligned
    _d3_data = evals[evals["instruction_aligned"].notna()].copy()
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
            _points.append({"key": _row["model"], "value": _row["score"], "name": _row["alignment"]})
        _tabs[_cp] = mo.ui.anywidget(make_radar_chart(_points, _categories, f"Aligned vs Misaligned: {_cp}"))

    mo.ui.tabs(_tabs)
    return


if __name__ == "__main__":
    app.run()
