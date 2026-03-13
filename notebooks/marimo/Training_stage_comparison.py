import marimo

__generated_with = "0.20.2"
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

    Data averaged across all conditions and the `instruction_no_hint` template.
    Error bands show ±1 SE.
    """)
    return


@app.cell
def _(Path, pd):
    _root = Path(__file__).resolve().parent.parent.parent
    _data_dir = _root / "outputs" / "viz" / "training-comparison"

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

    # Aggregate across conditions
    agg = (
        _df.groupby(["model", "model_label", "n_turns_int"])
        .agg(score=("score", "mean"), stderr=("score_stderr", "mean"))
        .reset_index()
    )
    agg["ci_lo"] = (agg["score"] - agg["stderr"]).clip(lower=0)
    agg["ci_hi"] = (agg["score"] + agg["stderr"]).clip(upper=1)

    olmo_df = agg[agg["model"].str.startswith("olmo")].copy()
    llama_df = agg[agg["model"].str.startswith("llama")].copy()
    return llama_df, olmo_df


@app.cell
def _(alt, olmo_df):
    # OLMo panel — SFT / SFT+DPO / SFT+DPO+RLVR
    _W, _H, _FONT = 500, 270, 14
    _domain = ["SFT", "SFT + DPO", "SFT + DPO + RLVR"]
    _range = ["#8dd3c7", "#bebada", "#80b1d3"]
    _rule = (
        alt.Chart(alt.InlineData(values=[{"y": 0.5}]))
        .mark_rule(color="#aaa", strokeDash=[3, 3], strokeWidth=0.8)
        .encode(y=alt.Y("y:Q"))
    )
    _color = alt.Color(
        "model_label:N",
        scale=alt.Scale(domain=_domain, range=_range),
        legend=alt.Legend(title=None),
    )
    _band = (
        alt.Chart(olmo_df)
        .mark_area(opacity=0.2, interpolate="monotone")
        .encode(
            x=alt.X("n_turns_int:Q", title="Number of turns", axis=alt.Axis(tickCount=6, labelFontSize=_FONT, titleFontSize=_FONT)),
            y=alt.Y("ci_lo:Q", scale=alt.Scale(domain=[0, 1]), title="Instruction following rate", axis=alt.Axis(labelFontSize=_FONT, titleFontSize=_FONT)),
            y2="ci_hi:Q",
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range)),
        )
    )
    _line = (
        alt.Chart(olmo_df)
        .mark_line(interpolate="monotone", strokeWidth=1.6)
        .encode(x="n_turns_int:Q", y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])), color=_color)
    )
    _dots = (
        alt.Chart(olmo_df)
        .mark_point(size=28, filled=True)
        .encode(
            x="n_turns_int:Q",
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range)),
            tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("n_turns_int:Q", title="N"), alt.Tooltip("score:Q", format=".3f", title="IF Rate")],
        )
    )
    (
        (_band + _line + _dots + _rule)
        .properties(width=_W, height=_H, title=alt.TitleParams("OLMo 3.1 32B — training stages", fontSize=_FONT + 1, fontWeight="normal"))
        .configure_view(stroke=None)
        .configure_axis(grid=False, domainColor="#888", tickColor="#888")
      #  .configure_legend(orient="right", labelFontSize=_FONT, symbolStrokeWidth=2, padding=4, titleFontSize=0)
    )
    return


@app.cell
def _(alt, llama_df):
    # Llama panel — 3.1 vs 3.3
    _W, _H, _FONT = 500, 270, 14
    _domain = ["Llama 3.1 70B", "Llama 3.3 70B"]
    _range = ["#fb8072", "#fdb462"]
    _rule = (
        alt.Chart(alt.InlineData(values=[{"y": 0.5}]))
        .mark_rule(color="#aaa", strokeDash=[3, 3], strokeWidth=0.8)
        .encode(y=alt.Y("y:Q"))
    )
    _color = alt.Color(
        "model_label:N",
        scale=alt.Scale(domain=_domain, range=_range),
        legend=alt.Legend(title=None),
    )
    _band = (
        alt.Chart(llama_df)
        .mark_area(opacity=0.2, interpolate="monotone")
        .encode(
            x=alt.X("n_turns_int:Q", title="Number of turns", axis=alt.Axis(tickCount=6, labelFontSize=_FONT, titleFontSize=_FONT)),
            y=alt.Y("ci_lo:Q", scale=alt.Scale(domain=[0, 1]), title="Instruction following rate", axis=alt.Axis(labelFontSize=_FONT, titleFontSize=_FONT)),
            y2="ci_hi:Q",
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range)),
        )
    )
    _line = (
        alt.Chart(llama_df)
        .mark_line(interpolate="monotone", strokeWidth=1.6)
        .encode(x="n_turns_int:Q", y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])), color=_color)
    )
    _dots = (
        alt.Chart(llama_df)
        .mark_point(size=28, filled=True)
        .encode(
            x="n_turns_int:Q",
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_label:N", scale=alt.Scale(domain=_domain, range=_range)),
            tooltip=[alt.Tooltip("model_label:N", title="Model"), alt.Tooltip("n_turns_int:Q", title="N"), alt.Tooltip("score:Q", format=".3f", title="IF Rate")],
        )
    )
    (
        (_band + _line + _dots + _rule)
        .properties(width=_W, height=_H, title=alt.TitleParams("Llama 70B — version comparison", fontSize=_FONT + 1, fontWeight="normal"))
        .configure_view(stroke=None)
        .configure_axis(grid=False, domainColor="#888", tickColor="#888")
        .configure_legend(orient="right", labelFontSize=_FONT, symbolStrokeWidth=2, padding=4, titleFontSize=0)
    )
    return


if __name__ == "__main__":
    app.run()
