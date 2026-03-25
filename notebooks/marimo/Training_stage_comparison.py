import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    from pathlib import Path

    return Path, alt, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training stage comparison

    Transition curves (P(output=T) vs N) comparing training stages.

    - **OLMo 3.1 32B**: three stages — SFT only, SFT+DPO, SFT+DPO+RLVR (= final instruct model)
    - **Llama 3.1 vs 3.3** (both 70B instruct): effect of one generation of training improvements

    Data filtered to `instruction_no_hint` template. Error bands show ±1 SE.
    """)
    return


@app.cell
def _(Path):
    plots_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "plots"
        / "training-comparison"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    return (plots_dir,)


@app.cell
def _(Path, pd):
    _root = Path(__file__).resolve().parent.parent.parent
    _data_dir = _root / "outputs" / "viz" / "training-comparison"

    # --- Behavioral data ---
    _df = pd.read_parquet(_data_dir / "evals.parquet")
    _df = _df[_df["instruction"] == "instruction_no_hint"].copy()
    _df["n_turns_int"] = _df["n_turns"].astype(int)

    _MODEL_LABELS = {
        "olmo-3.1-32b-instruct-sft": "SFT",
        "olmo-3.1-32b-instruct-dpo": "SFT + DPO",
        "olmo-3.1-32b-instruct": "SFT + DPO + RLVR",
        "llama-3.1-70b-instruct": "Llama 3.1 70B",
        "llama-3.3-70b-instruct": "Llama 3.3 70B",
    }
    _df["model_label"] = _df["model"].map(_MODEL_LABELS).fillna(_df["model"])

    # Static conditions (exclude token-pattern conditions)
    _static_conds = {
        "neutral",
        "factual_aligned_earth",
        "factual_misaligned_earth",
        "value_aligned_helpful",
        "value_misaligned_helpful",
    }
    # Dynamic (task-based) conditions
    _dynamic_conds = {
        "language_fr_ru",
        "language_ru_fr",
        "persona_casual_formal",
        "persona_formal_casual",
        "preference_aligned_helpful",
        "preference_misaligned_helpful",
        "style_javascript_python",
        "style_python_javascript",
    }

    static_df = _df[_df["condition"].isin(_static_conds)].copy()
    dynamic_df = _df[_df["condition"].isin(_dynamic_conds)].copy()

    # Aggregate across conditions within each group
    def _agg_group(df):
        agg = (
            df.groupby(["model", "model_label", "n_turns_int"])
            .agg(score=("score", "mean"), stderr=("score_stderr", "mean"))
            .reset_index()
        )
        agg["ci_lo"] = (agg["score"] - agg["stderr"]).clip(lower=0)
        agg["ci_hi"] = (agg["score"] + agg["stderr"]).clip(upper=1)
        return agg

    static_agg = _agg_group(static_df)
    dynamic_agg = _agg_group(dynamic_df)

    olmo_static = static_agg[static_agg["model"].str.startswith("olmo")].copy()
    llama_static = static_agg[static_agg["model"].str.startswith("llama")].copy()
    olmo_dynamic = dynamic_agg[dynamic_agg["model"].str.startswith("olmo")].copy()
    llama_dynamic = dynamic_agg[dynamic_agg["model"].str.startswith("llama")].copy()

    # --- Prediction data ---
    _dfp = pd.read_parquet(_data_dir / "evals_prediction.parquet")
    _dfp = _dfp[_dfp["instruction"] == "instruction_no_hint"].copy()
    _dfp["n_turns_int"] = _dfp["n_turns"].astype(int)
    _dfp["model_label"] = _dfp["model"].map(_MODEL_LABELS).fillna(_dfp["model"])

    # Exclude token conditions from prediction too
    _dfp = _dfp[~_dfp["condition"].str.startswith("token_")].copy()

    pred_agg = (
        _dfp.groupby(["model", "model_label", "n_turns_int"])
        .agg(
            if_rate=("score_instruction_following", "mean"),
            stderr_if=("stderr_instruction_following", "mean"),
            pred_accuracy=("score_prediction_accuracy", "mean"),
            stderr_pa=("stderr_prediction_accuracy", "mean"),
            pred_instruction=("score_prediction_instruction", "mean"),
            stderr_pi=("stderr_prediction_instruction", "mean"),
        )
        .reset_index()
    )
    pred_agg["ci_lo_if"] = (pred_agg["if_rate"] - pred_agg["stderr_if"]).clip(lower=0)
    pred_agg["ci_hi_if"] = (pred_agg["if_rate"] + pred_agg["stderr_if"]).clip(upper=1)
    pred_agg["ci_lo_pa"] = (pred_agg["pred_accuracy"] - pred_agg["stderr_pa"]).clip(lower=0)
    pred_agg["ci_hi_pa"] = (pred_agg["pred_accuracy"] + pred_agg["stderr_pa"]).clip(upper=1)
    pred_agg["ci_lo_pi"] = (pred_agg["pred_instruction"] - pred_agg["stderr_pi"]).clip(lower=0)
    pred_agg["ci_hi_pi"] = (pred_agg["pred_instruction"] + pred_agg["stderr_pi"]).clip(upper=1)

    olmo_pred = pred_agg[pred_agg["model"].str.startswith("olmo")].copy()
    llama_pred = pred_agg[pred_agg["model"].str.startswith("llama")].copy()
    return (
        llama_dynamic,
        llama_pred,
        llama_static,
        olmo_dynamic,
        olmo_pred,
        olmo_static,
    )


@app.cell
def _(alt):
    def make_transition_panel(data, title, domain, range_colors, y_col="score", y_title="Instruction following rate"):
        """Transition curve panel: band + line + dots + 0.5 reference rule."""
        _band = (
            alt.Chart(data)
            .mark_area(opacity=0.2, interpolate="monotone")
            .encode(
                x=alt.X("n_turns_int:Q", title="Number of turns"),
                y=alt.Y("ci_lo:Q", scale=alt.Scale(domain=[0, 1]), title=y_title),
                y2="ci_hi:Q",
                color=alt.Color(
                    "model_label:N",
                    scale=alt.Scale(domain=domain, range=range_colors),
                ),
            )
        )
        _line = (
            alt.Chart(data)
            .mark_line(interpolate="monotone", strokeWidth=1.6)
            .encode(
                x="n_turns_int:Q",
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "model_label:N",
                    scale=alt.Scale(domain=domain, range=range_colors),
                    legend=alt.Legend(title=None),
                ),
            )
        )
        _dots = (
            alt.Chart(data)
            .mark_point(size=28, filled=True)
            .encode(
                x="n_turns_int:Q",
                y=alt.Y(f"{y_col}:Q", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    "model_label:N",
                    scale=alt.Scale(domain=domain, range=range_colors),
                ),
                tooltip=[
                    alt.Tooltip("model_label:N", title="Model"),
                    alt.Tooltip("n_turns_int:Q", title="N"),
                    alt.Tooltip(f"{y_col}:Q", format=".3f", title="IF Rate"),
                ],
            )
        )
        _rule = (
            alt.Chart(alt.InlineData(values=[{"y": 0.5}]))
            .mark_rule(color="#aaa", strokeDash=[3, 3], strokeWidth=0.8)
            .encode(y=alt.Y("y:Q"))
        )
        return (_band + _line + _dots + _rule).properties(
            width=500, height=270, title=title,
        )

    return (make_transition_panel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavioral — Static conditions

    Fixed-output conditions (neutral, factual, value), excluding token-pattern conditions.
    Averaged across conditions within the group.
    """)
    return


@app.cell
def _(make_transition_panel, olmo_static, plots_dir):
    _olmo_static_chart = make_transition_panel(
        olmo_static,
        "OLMo 3.1 32B — Training stages (fixed-output conditions)",
        ["SFT", "SFT + DPO", "SFT + DPO + RLVR"],
        ["#8dd3c7", "#bebada", "#80b1d3"],
    )
    _olmo_static_chart.save(str(plots_dir / "olmo_static_behavioral.png"), scale_factor=2)
    _olmo_static_chart
    return


@app.cell
def _(llama_static, make_transition_panel, plots_dir):
    _llama_static_chart = make_transition_panel(
        llama_static,
        "Llama 70B — Version comparison (fixed-output conditions)",
        ["Llama 3.1 70B", "Llama 3.3 70B"],
        ["#fb8072", "#fdb462"],
    )
    _llama_static_chart.save(str(plots_dir / "llama_static_behavioral.png"), scale_factor=2)
    _llama_static_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavioral — Dynamic (task-based) conditions

    Language, persona, style, preference, and code conditions.
    These require LLM-judge or format-check scoring rather than exact match.
    Averaged across conditions within the group.
    """)
    return


@app.cell
def _(make_transition_panel, olmo_dynamic, plots_dir):
    _olmo_dyn_chart = make_transition_panel(
        olmo_dynamic,
        "OLMo 3.1 32B — Training stages (task-based conditions)",
        ["SFT", "SFT + DPO", "SFT + DPO + RLVR"],
        ["#8dd3c7", "#bebada", "#80b1d3"],
    )
    _olmo_dyn_chart.save(str(plots_dir / "olmo_dynamic_behavioral.png"), scale_factor=2)
    _olmo_dyn_chart
    return


@app.cell
def _(llama_dynamic, make_transition_panel, plots_dir):
    _llama_dyn_chart = make_transition_panel(
        llama_dynamic,
        "Llama 70B — Version comparison (task-based conditions)",
        ["Llama 3.1 70B", "Llama 3.3 70B"],
        ["#fb8072", "#fdb462"],
    )
    _llama_dyn_chart.save(str(plots_dir / "llama_dynamic_behavioral.png"), scale_factor=2)
    _llama_dyn_chart
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Self-prediction (Protocol 2) — Static conditions

    Three metrics from the self-prediction protocol:
    - **IF Rate**: instruction following rate (actual output after predicting)
    - **Prediction Accuracy**: how often the model correctly predicted its own output
    - **Prediction = Instruction**: how often the model predicted it would follow the instruction
    """)
    return


@app.cell
def _(alt, olmo_pred, pd, plots_dir):
    # OLMo prediction — melt to long form for faceting by metric
    _metrics = [
        ("if_rate", "ci_lo_if", "ci_hi_if", "IF Rate (actual)"),
        ("pred_accuracy", "ci_lo_pa", "ci_hi_pa", "Prediction Accuracy"),
        ("pred_instruction", "ci_lo_pi", "ci_hi_pi", "Predicted = Instruction"),
    ]
    _rows = []
    for _, _r in olmo_pred.iterrows():
        for _col, _lo, _hi, _label in _metrics:
            _rows.append({
                "model_label": _r["model_label"],
                "n_turns_int": _r["n_turns_int"],
                "value": _r[_col],
                "ci_lo": _r[_lo],
                "ci_hi": _r[_hi],
                "metric": _label,
            })
    _olmo_pred_long = pd.DataFrame(_rows)

    _domain = ["SFT", "SFT + DPO", "SFT + DPO + RLVR"]
    _range = ["#8dd3c7", "#bebada", "#80b1d3"]
    _band = (
        alt.Chart(_olmo_pred_long)
        .mark_area(opacity=0.2, interpolate="monotone")
        .encode(
            x=alt.X("n_turns_int:Q", title="Number of turns"),
            y=alt.Y("ci_lo:Q", scale=alt.Scale(domain=[0, 1]), title="Rate"),
            y2="ci_hi:Q",
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range)),
        )
    )
    _line = (
        alt.Chart(_olmo_pred_long)
        .mark_line(interpolate="monotone", strokeWidth=1.6)
        .encode(
            x="n_turns_int:Q",
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range), legend=alt.Legend(title=None)),
        )
    )
    _dots = (
        alt.Chart(_olmo_pred_long)
        .mark_point(size=28, filled=True)
        .encode(
            x="n_turns_int:Q",
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range)),
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("n_turns_int:Q", title="N"),
                alt.Tooltip("value:Q", format=".3f", title="Rate"),
                alt.Tooltip("metric:N", title="Metric"),
            ],
        )
    )
    _rule = (
        alt.Chart(alt.InlineData(values=[{"y": 0.5}]))
        .mark_rule(color="#aaa", strokeDash=[3, 3], strokeWidth=0.8)
        .encode(y=alt.Y("y:Q"))
    )
    _olmo_pred_chart = (
        (_band + _line + _dots + _rule)
        .facet(facet=alt.Facet("metric:N", title=None), columns=3)
        .resolve_scale(y="shared")
        .properties(title="OLMo 3.1 32B — Self-prediction by training stage")
    )
    _olmo_pred_chart.save(str(plots_dir / "olmo_prediction.png"), scale_factor=2)
    _olmo_pred_chart
    return


@app.cell
def _(alt, llama_pred, pd, plots_dir):
    # Llama prediction — melt to long form for faceting by metric
    _metrics = [
        ("if_rate", "ci_lo_if", "ci_hi_if", "IF Rate (actual)"),
        ("pred_accuracy", "ci_lo_pa", "ci_hi_pa", "Prediction Accuracy"),
        ("pred_instruction", "ci_lo_pi", "ci_hi_pi", "Predicted = Instruction"),
    ]
    _rows = []
    for _, _r in llama_pred.iterrows():
        for _col, _lo, _hi, _label in _metrics:
            _rows.append({
                "model_label": _r["model_label"],
                "n_turns_int": _r["n_turns_int"],
                "value": _r[_col],
                "ci_lo": _r[_lo],
                "ci_hi": _r[_hi],
                "metric": _label,
            })
    _llama_pred_long = pd.DataFrame(_rows)

    _domain = ["Llama 3.1 70B", "Llama 3.3 70B"]
    _range = ["#fb8072", "#fdb462"]
    _band = (
        alt.Chart(_llama_pred_long)
        .mark_area(opacity=0.2, interpolate="monotone")
        .encode(
            x=alt.X("n_turns_int:Q", title="Number of turns"),
            y=alt.Y("ci_lo:Q", scale=alt.Scale(domain=[0, 1]), title="Rate"),
            y2="ci_hi:Q",
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range)),
        )
    )
    _line = (
        alt.Chart(_llama_pred_long)
        .mark_line(interpolate="monotone", strokeWidth=1.6)
        .encode(
            x="n_turns_int:Q",
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range), legend=alt.Legend(title=None)),
        )
    )
    _dots = (
        alt.Chart(_llama_pred_long)
        .mark_point(size=28, filled=True)
        .encode(
            x="n_turns_int:Q",
            y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range)),
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("n_turns_int:Q", title="N"),
                alt.Tooltip("value:Q", format=".3f", title="Rate"),
                alt.Tooltip("metric:N", title="Metric"),
            ],
        )
    )
    _rule = (
        alt.Chart(alt.InlineData(values=[{"y": 0.5}]))
        .mark_rule(color="#aaa", strokeDash=[3, 3], strokeWidth=0.8)
        .encode(y=alt.Y("y:Q"))
    )
    _llama_pred_chart = (
        (_band + _line + _dots + _rule)
        .facet(facet=alt.Facet("metric:N", title=None), columns=3)
        .resolve_scale(y="shared")
        .properties(title="Llama 70B — Self-prediction by version")
    )
    _llama_pred_chart.save(str(plots_dir / "llama_prediction.png"), scale_factor=2)
    _llama_pred_chart
    return


if __name__ == "__main__":
    app.run()
