import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import pandas as pd
    from pathlib import Path
    from src.plotting_utils import DISPLAY_NAMES

    return Path, DISPLAY_NAMES, alt, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix: Per-model transition curves

    Instruction-following rate P(output=T) vs number of context turns N for each model,
    averaged across conditions within each group.

    - **Left panel per model**: fixed-output conditions (neutral, factual, value)
    - **Right panel per model**: task-based conditions (language, persona, style, preference)

    Error bands show ±1 SE. Data filtered to `instruction_no_hint` template.
    Layout: 4 panels per row (2 models × 2 condition types).
    """)
    return


@app.cell
def _(Path):
    plots_dir = (
        Path(__file__).resolve().parent.parent.parent
        / "outputs"
        / "plots"
        / "appendix-per-model"
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    return (plots_dir,)


@app.cell
def _(DISPLAY_NAMES, Path, pd):
    _root = Path(__file__).resolve().parent.parent.parent

    _STATIC_CONDS = {
        "neutral",
        "factual_aligned_earth",
        "factual_misaligned_earth",
        "value_aligned_helpful",
        "value_misaligned_helpful",
    }
    _DYNAMIC_CONDS = {
        "language_fr_ru",
        "language_ru_fr",
        "persona_casual_formal",
        "persona_formal_casual",
        "preference_aligned_helpful",
        "preference_misaligned_helpful",
        "style_javascript_python",
        "style_python_javascript",
    }

    # Load and filter static data
    _s = pd.read_parquet(_root / "outputs" / "viz" / "static" / "evals.parquet")
    _s = _s[_s["instruction"] == "instruction_no_hint"].copy()
    _s["n_turns_int"] = _s["n_turns"].astype(int)
    _s = _s[_s["condition"].isin(_STATIC_CONDS)]

    # Load and filter dynamic data
    _d = pd.read_parquet(_root / "outputs" / "viz" / "dynamic" / "evals.parquet")
    _d = _d[_d["instruction"] == "instruction_no_hint"].copy()
    _d["n_turns_int"] = _d["n_turns"].astype(int)
    _d = _d[_d["condition"].isin(_DYNAMIC_CONDS)]

    def _agg(df, ctype):
        agg = (
            df.groupby(["model", "n_turns_int"])
            .agg(score=("score", "mean"), stderr=("score_stderr", "mean"))
            .reset_index()
        )
        agg["ci_lo"] = (agg["score"] - agg["stderr"]).clip(lower=0)
        agg["ci_hi"] = (agg["score"] + agg["stderr"]).clip(upper=1)
        agg["condition_type"] = ctype
        return agg

    _static_agg = _agg(_s, "Fixed-output")
    _dynamic_agg = _agg(_d, "Task-based")

    combined = pd.concat([_static_agg, _dynamic_agg], ignore_index=True)
    combined["model_label"] = combined["model"].map(DISPLAY_NAMES).fillna(combined["model"])

    # Keep only models present in both condition types and in DISPLAY_NAMES
    _has_both = combined.groupby("model")["condition_type"].nunique() == 2
    _valid_models = [m for m in _has_both[_has_both].index if m in DISPLAY_NAMES]

    combined = combined[combined["model"].isin(_valid_models)].copy()

    # Build panel ordering: sort models alphabetically by display name,
    # interleave fixed-output / task-based so pairs of 2 models fit 4-per-row
    _model_order = sorted(_valid_models, key=lambda m: DISPLAY_NAMES[m])
    _type_order = ["Fixed-output", "Task-based"]

    _key_map = {
        (m, t): f"{DISPLAY_NAMES[m]}\n({t})"
        for m in _model_order
        for t in _type_order
    }
    combined["panel_key"] = combined.apply(
        lambda r: _key_map.get((r["model"], r["condition_type"]), ""), axis=1
    )
    panel_order = [_key_map[(m, t)] for m in _model_order for t in _type_order]

    return combined, panel_order


@app.cell
def _(DISPLAY_NAMES, alt, combined, mo, panel_order, plots_dir):
    # Build ordered list of (model_id, display_name) pairs
    _model_order = sorted(
        combined["model"].unique(),
        key=lambda m: DISPLAY_NAMES.get(m, m),
    )
    # Group into pairs (2 models × 2 types = 4 panels per row)
    _pairs = [_model_order[i:i+2] for i in range(0, len(_model_order), 2)]

    def _make_row_chart(models_in_row, row_panel_order):
        _data = combined[combined["model"].isin(models_in_row)].copy()
        _base = (
            alt.layer(
                alt.Chart(_data)
                .mark_area(opacity=0.15, interpolate="monotone", color="#4878cf")
                .encode(
                    x=alt.X("n_turns_int:Q", title="N"),
                    y=alt.Y(
                        "ci_lo:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        title="IF rate",
                    ),
                    y2="ci_hi:Q",
                ),
                alt.Chart(_data)
                .mark_line(interpolate="monotone", strokeWidth=1.4, color="#4878cf")
                .encode(
                    x="n_turns_int:Q",
                    y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
                ),
                alt.Chart(_data)
                .mark_rule(color="#bbb", strokeDash=[3, 3], strokeWidth=0.7)
                .encode(y=alt.datum(0.5)),
            )
            .properties(width=190, height=150)
        )
        return (
            _base.facet(
                facet=alt.Facet(
                    "panel_key:N",
                    sort=row_panel_order,
                    title=None,
                    header=alt.Header(
                        labelFontSize=10,
                        labelLimit=240,
                        labelLineHeight=13,
                    ),
                ),
                columns=4,
            )
            .configure_axis(labelFontSize=9, titleFontSize=10)
        )

    _charts = []
    for _i, _pair in enumerate(_pairs):
        _row_order = [p for p in panel_order if any(m in p for m in [DISPLAY_NAMES.get(m2, m2) for m2 in _pair])]
        _chart = _make_row_chart(_pair, _row_order)
        _fname = plots_dir / f"per_model_row_{_i+1:02d}.png"
        _chart.save(str(_fname), scale_factor=2)
        _charts.append(_chart)

    mo.vstack(_charts)


if __name__ == "__main__":
    app.run()
