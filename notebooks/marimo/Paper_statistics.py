# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "polars",
#     "scipy",
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
    import numpy as np
    import json
    import sys
    from pathlib import Path
    from scipy import stats

    alt.data_transformers.disable_max_rows()

    _nb_dir = Path(__file__).resolve().parent
    if str(_nb_dir) not in sys.path:
        sys.path.insert(0, str(_nb_dir))

    from paper_stats_config import (
        CORE_MODELS,
        TRAINING_MODELS,
        MODEL_ALIASES,
        DISPLAY_NAMES,
    )

    ROOT = Path(__file__).resolve().parent.parent.parent
    return (
        CORE_MODELS,
        DISPLAY_NAMES,
        MODEL_ALIASES,
        ROOT,
        TRAINING_MODELS,
        alt,
        json,
        mo,
        np,
        pl,
        stats,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Paper Statistics

    Loads **T=0, no-hint** data for main results (§3.1–§3.3, §3.5).
    T=1 and hint data are loaded separately for §3.4b–c only.

    **Uncertainty methodology:** Each cell (model × condition × N) aggregates n=35 (T0) or n=15 (T1)
    binary trials with different questions. The pre-computed `score_stderr` = SE of that cell's proportion.
    When averaging across cells (e.g., per-model means), the 95% CI uses the between-cell SE =
    SD(cell scores)/sqrt(K), which captures both condition/N variation and within-cell noise.
    For paired comparisons, CIs use the t-distribution on per-unit differences.

    | Section | Claim | Data source |
    |---------|-------|-------------|
    | §3.1a | Avg IF rate range 0.028 – 0.944  | `static/evals.parquet` |
    | §3.1b | First N ≤ 0.5 ranges N=2 to never | fixed-output + task-based |
    | §3.1c | Fixed-output all p>0.14; task-based IFBench r=0.62 p=0.015 | caps × IF rates |
    | §3.2 | Value gap 15pp, factual 12pp, overall t=1.40 p=0.18 | fixed-output aligned/misaligned |
    | §3.3 | OLMo SFT=0.057, +DPO=0.166, +RLVR=0.160 | training-comparison |
    | §3.3 | Llama 3.1=0.82, 3.3=0.94; at N=50: 3.1=0.38, 3.3=0.74 | training-comparison |
    | §3.4 | GPT-5.2 0.17 vs GPT-5.2-medium 0.65 (fixed-output) | fixed-output/task-based raw |
    | §3.4b | T0 vs T1 temperature effect | static + static-T1 |
    | §3.4c | Hint substantially increases robustness | static-T1 |
    | §3.5a | Grand mean accuracy=83.5%; predicted IF=14.3%, actual IF=26.8% | prediction parquet (5 fixed-output, 13 core models) |
    | §3.5b | Mean delta=-0.026, t=-2.91, p=0.004 | combined_filtered |
    """)
    return


@app.cell
def _(MODEL_ALIASES, ROOT, json, pl):
    def _normalize(df):
        """Replace model names using MODEL_ALIASES; unmatched values are kept as-is."""
        if not MODEL_ALIASES:
            return df
        _models = df["model"].unique().to_list()
        _full_map = {m: MODEL_ALIASES.get(m, m) for m in _models}
        return df.with_columns(pl.col("model").replace(_full_map))


    _static_path = ROOT / "outputs" / "viz" / "static"
    _t1_path = ROOT / "outputs" / "viz" / "static-T1"
    _dynamic_path = ROOT / "outputs" / "viz" / "dynamic"
    _training_path = ROOT / "outputs" / "viz" / "training-comparison"
    _followup_path = ROOT / "outputs" / "viz" / "additional-dynamic"

    static_raw = _normalize(
        pl.read_parquet(_static_path / "evals.parquet").filter(
            pl.col("instruction") == "instruction_no_hint"
        )
    )
    t1_raw = _normalize(pl.read_parquet(_t1_path / "evals.parquet"))
    dynamic_raw = _normalize(
        pl.read_parquet(_dynamic_path / "evals.parquet").filter(
            pl.col("instruction") == "instruction_no_hint"
        )
    )
    training_raw = _normalize(pl.read_parquet(_training_path / "evals.parquet"))
    pred_raw = _normalize(
        pl.read_parquet(_static_path / "evals_prediction.parquet").filter(
            pl.col("instruction") == "instruction_no_hint"
        )
    )
    combined_raw = _normalize(
        pl.read_parquet(_static_path / "_combined_filtered.parquet").filter(
            pl.col("instruction") == "instruction_no_hint"
        )
    )
    followup_raw = _normalize(
        pl.read_parquet(_followup_path / "evals.parquet").filter(
            pl.col("instruction") == "instruction_no_hint"
        )
    )

    with open(ROOT / "data" / "model_capability_scores.json") as _f:
        _caps_json = json.load(_f)

    if "hermes-4-70b-reasoninh" in _caps_json:
        _caps_json["hermes-4-70b-reasoning"] = _caps_json.pop(
            "hermes-4-70b-reasoninh"
        )

    _cap_rows = []
    for _model, _variants in _caps_json.items():
        for _reasoning_str, _data in _variants.items():
            _cap_rows.append(
                {
                    "model": MODEL_ALIASES.get(_model, _model),
                    "reasoning": _reasoning_str == "true",
                    "intelligence_index": _data.get("intelligence_index"),
                    "mmlu_pro": _data.get("mmlu_pro"),
                    "gpqa": _data.get("gpqa"),
                    "ifbench": _data.get("ifbench"),
                }
            )
    caps_df = pl.DataFrame(_cap_rows)
    return (
        caps_df,
        combined_raw,
        dynamic_raw,
        followup_raw,
        pred_raw,
        static_raw,
        t1_raw,
        training_raw,
    )


@app.cell
def _(CORE_MODELS, TRAINING_MODELS, mo):
    core_model_select = mo.ui.multiselect(
        options=CORE_MODELS,
        value=CORE_MODELS,
        label="Core models",
    )
    training_model_select = mo.ui.multiselect(
        options=TRAINING_MODELS,
        value=TRAINING_MODELS,
        label="Training models",
    )
    mo.hstack([core_model_select, training_model_select], widths="equal")
    return core_model_select, training_model_select


@app.cell
def _(
    combined_raw,
    core_model_select,
    dynamic_raw,
    followup_raw,
    pl,
    pred_raw,
    static_raw,
    t1_raw,
    training_model_select,
    training_raw,
):
    _core = core_model_select.value
    _training = training_model_select.value
    _token_conditions = ["token_states_countries", "token_countries_states"]

    static_df = static_raw.filter(
        pl.col("model").is_in(_core),
        ~pl.col("condition").is_in(_token_conditions),
    )
    dynamic_df = dynamic_raw.filter(pl.col("model").is_in(_core))
    followup_df = followup_raw  # only 3 models, no core filter needed
    training_df = training_raw.filter(
        pl.col("model").is_in(_training),
        ~pl.col("condition").is_in(_token_conditions),
    )
    pred_df = pred_raw.filter(
        pl.col("model").is_in(_core),
        ~pl.col("condition").is_in(_token_conditions),
    )
    combined_df = combined_raw.filter(
        pl.col("model").is_in(_core),
        ~pl.col("condition").is_in(_token_conditions),
    )

    t1_no_hint_df = t1_raw.filter(
        pl.col("instruction") == "instruction_no_hint",
        pl.col("model").is_in(_core),
    )
    t1_df = t1_raw.filter(pl.col("model").is_in(_core))
    return (
        combined_df,
        dynamic_df,
        followup_df,
        pred_df,
        static_df,
        t1_df,
        t1_no_hint_df,
        training_df,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## §3.1 — Instruction-Following Rates
    """)
    return


@app.cell
def _(DISPLAY_NAMES, mo, pl, static_df):
    mo.md("### §3.1a — Average IF rate per model (fixed-output T=0)")

    _avg = (
        static_df.group_by("model")
        .agg(
            pl.col("score").mean().alias("mean_if"),
            pl.col("score").std().alias("sd"),
            pl.col("score").len().alias("n_cells"),
        )
        .with_columns(
            (pl.col("sd") / pl.col("n_cells").sqrt()).alias("se"),
        )
        .with_columns(
            (pl.col("mean_if") - 1.96 * pl.col("se")).alias("ci_lo"),
            (pl.col("mean_if") + 1.96 * pl.col("se")).alias("ci_hi"),
        )
        .sort("mean_if", descending=True)
        .with_columns(
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .select(
            [
                "display_name",
                "mean_if",
                "ci_lo",
                "ci_hi",
                "n_cells",
            ]
        )
        .with_columns(
            pl.col("mean_if").round(3),
            pl.col("ci_lo").round(3),
            pl.col("ci_hi").round(3),
        )
    )

    _lo = _avg["mean_if"].min()
    _hi = _avg["mean_if"].max()
    mo.vstack(
        [
            mo.md(
                f"Range: **{_lo:.3f}** -- **{_hi:.3f}**. "
                f"95% CIs are between-cell (across conditions x N), reflecting systematic + sampling variation."
            ),
            _avg,
        ]
    )
    return


@app.cell
def _(DISPLAY_NAMES, dynamic_df, mo, pl, static_df):
    mo.md("""### §3.1b — First N where avg IF <= 0.5 (fixed-output and task-based)

    Score and within-cell SE at the threshold N show how sharp the crossing is.
    """)


    def _first_n_below(df, label):
        _by_model_n = (
            df.with_columns(pl.col("n_turns").cast(pl.Int64).alias("n_turns_int"))
            .group_by(["model", "n_turns_int"])
            .agg(
                pl.col("score").mean().alias("mean_score"),
                pl.col("score_stderr").mean().alias("mean_se"),
                pl.col("score").len().alias("n_cond"),
            )
            .sort(["model", "n_turns_int"])
        )
        _below = (
            _by_model_n.filter(pl.col("mean_score") <= 0.5)
            .sort(["model", "n_turns_int"])
            .group_by("model")
            .agg(
                pl.col("n_turns_int").first().alias("first_n"),
                pl.col("mean_score").first().alias("score_at_n"),
                pl.col("mean_se").first().alias("se_at_n"),
            )
        )
        _all_models = df["model"].unique().to_frame()
        return _all_models.join(_below, on="model", how="left").with_columns(
            pl.lit(label).alias("dataset")
        )


    _static_th = _first_n_below(static_df, "fixed-output")
    _dynamic_th = _first_n_below(dynamic_df, "task-based")
    _thresholds = pl.concat([_static_th, _dynamic_th])

    # Pivot for a compact display
    _first_n_pivot = (
        _thresholds.select(["model", "dataset", "first_n"])
        .with_columns(pl.col("first_n").cast(pl.Utf8).fill_null("never"))
        .pivot(on="dataset", index="model", values="first_n")
        .with_columns(pl.col("model").replace(DISPLAY_NAMES).alias("model"))
        .sort("model")
    )

    # Detail table with score at threshold
    _detail = (
        _thresholds.filter(pl.col("first_n").is_not_null())
        .with_columns(
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
            pl.col("score_at_n").round(3),
            pl.col("se_at_n").round(3),
        )
        .select(["display_name", "dataset", "first_n", "score_at_n", "se_at_n"])
        .sort(["dataset", "first_n"])
    )

    mo.vstack(
        [
            mo.md("**Threshold N (first N where mean IF <= 0.5)**"),
            _first_n_pivot,
            mo.md(
                "**Score at threshold** (mean +/- within-cell SE across conditions at that N)"
            ),
            _detail,
        ]
    )
    return


@app.cell
def _(CORE_MODELS, caps_df, dynamic_df, mo, np, pl, static_df, stats):
    mo.md("### §3.1c — Benchmark correlations with IF rates")

    _benchmarks = ["gpqa", "ifbench", "intelligence_index", "mmlu_pro"]

    _caps_base = caps_df.filter(~pl.col("reasoning"))

    _static_avg = (
        static_df.filter(pl.col("model").is_in(CORE_MODELS))
        .group_by("model")
        .agg(pl.col("score").mean().alias("if_static"))
    )
    _dynamic_avg = (
        dynamic_df.filter(pl.col("model").is_in(CORE_MODELS))
        .group_by("model")
        .agg(pl.col("score").mean().alias("if_dynamic"))
    )
    _joined = _caps_base.join(_static_avg, on="model", how="inner").join(
        _dynamic_avg, on="model", how="inner"
    )

    _rows = []
    for _bm in _benchmarks:
        _valid = _joined.filter(pl.col(_bm).is_not_null())
        _x = _valid[_bm].to_numpy()
        for _dataset, _col in [
            ("fixed-output", "if_static"),
            ("task-based", "if_dynamic"),
        ]:
            _y = _valid[_col].to_numpy()
            _n = len(_x)
            if _n < 4:
                continue
            _r_p, _p_p = stats.pearsonr(_x, _y)
            _r_s, _p_s = stats.spearmanr(_x, _y)
            # Fisher z-transform CI on Pearson r
            _z = np.arctanh(_r_p)
            _se_z = 1.0 / np.sqrt(_n - 3)
            _r_ci_lo = float(np.tanh(_z - 1.96 * _se_z))
            _r_ci_hi = float(np.tanh(_z + 1.96 * _se_z))
            _rows.append(
                {
                    "benchmark": _bm,
                    "dataset": _dataset,
                    "n": _n,
                    "pearson_r": round(float(_r_p), 3),
                    "r_ci_lo": round(_r_ci_lo, 3),
                    "r_ci_hi": round(_r_ci_hi, 3),
                    "pearson_p": round(float(_p_p), 3),
                    "spearman_r": round(float(_r_s), 3),
                    "spearman_p": round(float(_p_s), 3),
                    "sig": "***"
                    if _p_p < 0.001
                    else "**"
                    if _p_p < 0.01
                    else "*"
                    if _p_p < 0.05
                    else "",
                }
            )

    _corr_df = pl.DataFrame(_rows).sort(["dataset", "benchmark"])
    mo.vstack(
        [
            mo.md(" CIs via Fisher z-transform."),
            _corr_df,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §3.2 — Alignment Effects
    """)
    return


@app.cell
def _(mo, np, pl, static_df, stats):
    mo.md(
        "### §3.2a — Aligned vs misaligned IF rates (value + factual conditions)"
    )

    _alignment_df = static_df.filter(
        pl.col("condition_pair").is_in(["value", "factual"])
    )

    # Per-model: mean IF for aligned and misaligned
    _per_model = _alignment_df.group_by(["model", "instruction_aligned"]).agg(
        pl.col("score").mean().alias("mean_if")
    )
    _aligned_rates = (
        _per_model.filter(pl.col("instruction_aligned"))
        .sort("model")["mean_if"]
        .to_numpy()
    )
    _misaligned_rates = (
        _per_model.filter(~pl.col("instruction_aligned"))
        .sort("model")["mean_if"]
        .to_numpy()
    )

    # Paired t-test across models
    _diffs = _aligned_rates - _misaligned_rates
    _n_models = len(_diffs)
    _t, _p = stats.ttest_rel(_aligned_rates, _misaligned_rates)
    _df = _n_models - 1
    _t_crit = stats.t.ppf(0.975, _df)
    _mean_diff = float(np.mean(_diffs))
    _se_diff = float(np.std(_diffs, ddof=1) / np.sqrt(_n_models))
    _ci_lo_diff = _mean_diff - _t_crit * _se_diff
    _ci_hi_diff = _mean_diff + _t_crit * _se_diff

    # Per condition_pair: gap with CI from per-model gaps within each pair
    _pair_rows = []
    for _pair in ["value", "factual"]:
        _pair_df = _alignment_df.filter(pl.col("condition_pair") == _pair)
        _pm = _pair_df.group_by(["model", "instruction_aligned"]).agg(
            pl.col("score").mean().alias("mean_if")
        )
        _a = (
            _pm.filter(pl.col("instruction_aligned"))
            .sort("model")["mean_if"]
            .to_numpy()
        )
        _m = (
            _pm.filter(~pl.col("instruction_aligned"))
            .sort("model")["mean_if"]
            .to_numpy()
        )
        _d = _a - _m
        _gap = float(np.mean(_d))
        _se = float(np.std(_d, ddof=1) / np.sqrt(len(_d)))
        _tc = stats.t.ppf(0.975, len(_d) - 1)
        _pair_rows.append(
            {
                "condition_pair": _pair,
                "aligned": round(float(np.mean(_a)), 3),
                "misaligned": round(float(np.mean(_m)), 3),
                "gap_pp": round(_gap, 3),
                "gap_ci_lo": round(_gap - _tc * _se, 3),
                "gap_ci_hi": round(_gap + _tc * _se, 3),
            }
        )

    _pair_table = pl.DataFrame(_pair_rows)

    mo.vstack(
        [
            _pair_table,
        ]
    )
    return


@app.cell
def _(DISPLAY_NAMES, mo, pl, static_df):
    mo.md("### §3.2b — Per-model alignment gap")

    _align_df = static_df.filter(
        pl.col("condition_pair").is_in(["value", "factual"])
    )

    # Per-(model, instruction_aligned): mean and between-cell SE
    _per_model_stats = (
        _align_df.group_by(["model", "instruction_aligned"])
        .agg(
            pl.col("score").mean().alias("mean_if"),
            pl.col("score").std().alias("sd"),
            pl.col("score").len().alias("n"),
        )
        .with_columns(
            (pl.col("sd") / pl.col("n").sqrt()).alias("se"),
        )
    )

    _aligned = _per_model_stats.filter(pl.col("instruction_aligned")).select(
        ["model", pl.col("mean_if").alias("aligned"), pl.col("se").alias("se_a")]
    )
    _misaligned = _per_model_stats.filter(~pl.col("instruction_aligned")).select(
        [
            "model",
            pl.col("mean_if").alias("misaligned"),
            pl.col("se").alias("se_m"),
        ]
    )

    _gap_df = (
        _aligned.join(_misaligned, on="model")
        .with_columns(
            (pl.col("aligned") - pl.col("misaligned")).alias("gap_pp"),
            (pl.col("se_a").pow(2) + pl.col("se_m").pow(2)).sqrt().alias("se_gap"),
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .with_columns(
            (pl.col("gap_pp") - 1.96 * pl.col("se_gap")).alias("gap_ci_lo"),
            (pl.col("gap_pp") + 1.96 * pl.col("se_gap")).alias("gap_ci_hi"),
        )
        .select(
            [
                "display_name",
                "aligned",
                "misaligned",
                "gap_pp",
                "gap_ci_lo",
                "gap_ci_hi",
            ]
        )
        .with_columns(
            pl.col("aligned").round(3),
            pl.col("misaligned").round(3),
            pl.col("gap_pp").round(3),
            pl.col("gap_ci_lo").round(3),
            pl.col("gap_ci_hi").round(3),
        )
        .sort("gap_pp", descending=True)
    )
    mo.vstack(
        [
            mo.md(
                "CIs propagate between-cell SE for aligned and misaligned groups independently."
            ),
            _gap_df,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §3.3 — Training Stage Effects
    """)
    return


@app.cell
def _(DISPLAY_NAMES, alt, mo, pl, training_df):
    mo.md("### §3.3a — OLMo training stages")

    _olmo_models = [
        "olmo-3.1-32b-instruct-sft",
        "olmo-3.1-32b-instruct-dpo",
        "olmo-3.1-32b-instruct",
    ]
    _olmo_df = training_df.filter(pl.col("model").is_in(_olmo_models))

    _overall = (
        _olmo_df.group_by("model")
        .agg(
            pl.col("score").mean().alias("mean_if"),
            pl.col("score").std().alias("sd"),
            pl.col("score").len().alias("n_cells"),
        )
        .with_columns(
            (pl.col("sd") / pl.col("n_cells").sqrt()).alias("se"),
        )
        .with_columns(
            (pl.col("mean_if") - 1.96 * pl.col("se")).alias("ci_lo"),
            (pl.col("mean_if") + 1.96 * pl.col("se")).alias("ci_hi"),
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .select(["display_name", "mean_if", "ci_lo", "ci_hi", "n_cells"])
        .with_columns(
            pl.col("mean_if").round(3),
            pl.col("ci_lo").round(3),
            pl.col("ci_hi").round(3),
        )
        .sort("mean_if")
    )

    # Per (model, n_turns): mean across conditions with SE
    _by_n = (
        _olmo_df.with_columns(
            pl.col("n_turns").cast(pl.Int64).alias("n_turns_int")
        )
        .group_by(["model", "n_turns_int"])
        .agg(
            pl.col("score").mean().alias("mean_if"),
            pl.col("score").std().alias("sd"),
            pl.col("score").len().alias("n_cond"),
        )
        .with_columns(
            (pl.col("sd") / pl.col("n_cond").sqrt()).alias("se"),
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .with_columns(
            (pl.col("mean_if") - 1.96 * pl.col("se"))
            .clip(0.0, 1.0)
            .alias("ci_lo"),
            (pl.col("mean_if") + 1.96 * pl.col("se"))
            .clip(0.0, 1.0)
            .alias("ci_hi"),
        )
        .sort(["model", "n_turns_int"])
    )
    _by_n_pd = _by_n.to_pandas()

    _line = (
        alt.Chart(_by_n_pd)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_turns_int:Q", title="N turns"),
            y=alt.Y("mean_if:Q", title="IF rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "display_name:N", legend=alt.Legend(title="Training stage")
            ),
            tooltip=[
                "display_name",
                "n_turns_int",
                alt.Tooltip("mean_if:Q", format=".3f"),
            ],
        )
    )
    _band = (
        alt.Chart(_by_n_pd)
        .mark_area(opacity=0.2)
        .encode(
            x="n_turns_int:Q",
            y="ci_lo:Q",
            y2="ci_hi:Q",
            color=alt.Color("display_name:N"),
        )
    )
    _chart = (_line + _band).properties(
        title="OLMo: IF rate by N (band = 95% CI across conditions)",
        width=600,
        height=300,
    )

    mo.vstack(
        [
            mo.md("Claims: SFT=0.057, SFT+DPO=0.166, SFT+DPO+RLVR=0.160"),
            _overall,
            _chart,
        ]
    )
    return


@app.cell
def _(DISPLAY_NAMES, mo, pl, training_df):
    mo.md("### §3.3b — Llama 3.1 vs 3.3 comparison")

    _llama_models = ["llama-3.1-70b-instruct", "llama-3.3-70b-instruct"]
    _llama_df = training_df.filter(pl.col("model").is_in(_llama_models))

    _overall_llama = (
        _llama_df.group_by("model")
        .agg(
            pl.col("score").mean().alias("mean_if"),
            pl.col("score").std().alias("sd"),
            pl.col("score").len().alias("n"),
        )
        .with_columns((pl.col("sd") / pl.col("n").sqrt()).alias("se"))
        .with_columns(
            (pl.col("mean_if") - 1.96 * pl.col("se")).alias("ci_lo"),
            (pl.col("mean_if") + 1.96 * pl.col("se")).alias("ci_hi"),
        )
    )

    _at_n50 = (
        _llama_df.filter(pl.col("n_turns").cast(pl.Int64) == 50)
        .group_by("model")
        .agg(
            pl.col("score").mean().alias("mean_n50"),
            pl.col("score").std().alias("sd_n50"),
            pl.col("score").len().alias("n_n50"),
        )
        .with_columns((pl.col("sd_n50") / pl.col("n_n50").sqrt()).alias("se_n50"))
        .with_columns(
            (pl.col("mean_n50") - 1.96 * pl.col("se_n50")).alias("n50_ci_lo"),
            (pl.col("mean_n50") + 1.96 * pl.col("se_n50")).alias("n50_ci_hi"),
        )
    )

    _llama_table = (
        _overall_llama.join(_at_n50, on="model", how="left")
        .with_columns(pl.col("model").replace(DISPLAY_NAMES).alias("display_name"))
        .select(
            [
                "display_name",
                "mean_if",
                "ci_lo",
                "ci_hi",
                "mean_n50",
                "n50_ci_lo",
                "n50_ci_hi",
            ]
        )
        .with_columns(
            pl.col("mean_if").round(3),
            pl.col("ci_lo").round(3),
            pl.col("ci_hi").round(3),
            pl.col("mean_n50").round(3),
            pl.col("n50_ci_lo").round(3),
            pl.col("n50_ci_hi").round(3),
        )
        .sort("display_name")
    )

    mo.vstack(
        [
            mo.md("Claims: 3.1=0.82, 3.3=0.94; at N=50: 3.1=0.38, 3.3=0.74"),
            _llama_table,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §3.4 — Reasoning and Temperature Effects
    """)
    return


@app.cell
def _(DISPLAY_NAMES, dynamic_raw, mo, pl, static_raw):
    mo.md("### §3.4a — Reasoning vs non-reasoning (unfiltered raw data)")

    _reasoning_models = [
        "gpt-5.2",
        "gpt-5.2-medium",
        "hermes-4-70b",
        "hermes-4-70b-reasoning",
    ]

    _results = []
    for _ds_name, _ds in [
        ("fixed-output", static_raw),
        ("task-based", dynamic_raw),
    ]:
        _sub = (
            _ds.filter(pl.col("model").is_in(_reasoning_models))
            .group_by("model")
            .agg(
                pl.col("score").mean().alias("mean_if"),
                pl.col("score").std().alias("sd"),
                pl.col("score").len().alias("n"),
            )
        )
        for _row in _sub.iter_rows(named=True):
            _se = _row["sd"] / (_row["n"] ** 0.5) if _row["n"] > 1 else 0.0
            _results.append(
                {
                    "dataset": _ds_name,
                    "model": _row["model"],
                    "display_name": DISPLAY_NAMES.get(
                        _row["model"], _row["model"]
                    ),
                    "mean_if": round(_row["mean_if"], 3),
                    "ci_lo": round(_row["mean_if"] - 1.96 * _se, 3),
                    "ci_hi": round(_row["mean_if"] + 1.96 * _se, 3),
                    "n_cells": _row["n"],
                    "is_reasoning": _row["model"]
                    in ["gpt-5.2-medium", "hermes-4-70b-reasoning"],
                }
            )

    _reasoning_df = pl.DataFrame(_results)

    # Show as flat table (pivot loses the CIs)
    _table = _reasoning_df.select(
        [
            "display_name",
            "dataset",
            "mean_if",
            "ci_lo",
            "ci_hi",
            "n_cells",
        ]
    ).sort(["display_name", "dataset"])

    mo.vstack(
        [
            mo.md("Claim: GPT-5.2 0.17 vs GPT-5.2-medium 0.65 (fixed-output)"),
            _table,
        ]
    )
    return


@app.cell
def _(CORE_MODELS, DISPLAY_NAMES, mo, np, pl, static_df, stats, t1_no_hint_df):
    mo.md("""
    ### §3.4b — Temperature comparison: T=0 vs T=1 (no-hint only)

    > **Caveat:** T=0 = 35 samples/cell; T=1 = 15 samples/cell.
    > Only models present in both datasets (after alias normalization) are shown.
    """)

    _t1_models = set(t1_no_hint_df["model"].unique().to_list())
    _t0_models = set(static_df["model"].unique().to_list())
    _common_models = sorted(_t1_models & _t0_models & set(CORE_MODELS))

    _t0_avg = (
        static_df.filter(pl.col("model").is_in(_common_models))
        .group_by("model")
        .agg(
            pl.col("score").mean().alias("t0_if"),
            pl.col("score").std().alias("t0_sd"),
            pl.col("score").len().alias("t0_n"),
        )
        .with_columns((pl.col("t0_sd") / pl.col("t0_n").sqrt()).alias("t0_se"))
    )
    _t1_avg = (
        t1_no_hint_df.filter(pl.col("model").is_in(_common_models))
        .group_by("model")
        .agg(
            pl.col("score").mean().alias("t1_if"),
            pl.col("score").std().alias("t1_sd"),
            pl.col("score").len().alias("t1_n"),
        )
        .with_columns((pl.col("t1_sd") / pl.col("t1_n").sqrt()).alias("t1_se"))
    )

    _temp_df = (
        _t0_avg.join(_t1_avg, on="model")
        .with_columns(
            (pl.col("t1_if") - pl.col("t0_if")).alias("delta"),
            # Unpaired SE for delta (different sample sizes)
            (pl.col("t0_se").pow(2) + pl.col("t1_se").pow(2))
            .sqrt()
            .alias("se_delta"),
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .with_columns(
            (pl.col("delta") - 1.96 * pl.col("se_delta")).alias("delta_ci_lo"),
            (pl.col("delta") + 1.96 * pl.col("se_delta")).alias("delta_ci_hi"),
        )
        .select(
            [
                "display_name",
                "t0_if",
                "t1_if",
                "delta",
                "delta_ci_lo",
                "delta_ci_hi",
            ]
        )
        .with_columns(
            pl.col("t0_if").round(3),
            pl.col("t1_if").round(3),
            pl.col("delta").round(3),
            pl.col("delta_ci_lo").round(3),
            pl.col("delta_ci_hi").round(3),
        )
        .sort("display_name")
    )

    # Paired t-test across models
    _deltas = _temp_df["delta"].to_numpy()
    _n_m = len(_deltas)
    _t_stat, _p_val = stats.ttest_1samp(_deltas, 0)
    _df = _n_m - 1
    _t_crit = stats.t.ppf(0.975, _df)
    _mean_d = float(np.mean(_deltas))
    _se_d = float(np.std(_deltas, ddof=1) / np.sqrt(_n_m))

    mo.vstack(
        [
            mo.md(
                f"**{len(_common_models)} models** in common. "
                f"Grand mean delta (T1-T0): **{_mean_d:.3f}** [{_mean_d - _t_crit * _se_d:.3f}, {_mean_d + _t_crit * _se_d:.3f}], "
                f"t({_df}) = {_t_stat:.2f}, p = {_p_val:.3f}"
            ),
            _temp_df,
        ]
    )
    return


@app.cell
def _(CORE_MODELS, DISPLAY_NAMES, mo, np, pl, stats, t1_df):
    mo.md("### §3.4c — Hint effect within T=1 data")

    _t1_models_with_hint = set(
        t1_df.filter(pl.col("instruction") == "instruction_hint")["model"]
        .unique()
        .to_list()
    )
    _t1_models_no_hint = set(
        t1_df.filter(pl.col("instruction") == "instruction_no_hint")["model"]
        .unique()
        .to_list()
    )
    _hint_models = sorted(
        (_t1_models_with_hint & _t1_models_no_hint) & set(CORE_MODELS)
    )

    _no_hint_avg = (
        t1_df.filter(
            pl.col("instruction") == "instruction_no_hint",
            pl.col("model").is_in(_hint_models),
        )
        .group_by("model")
        .agg(
            pl.col("score").mean().alias("no_hint_if"),
            pl.col("score").std().alias("nh_sd"),
            pl.col("score").len().alias("nh_n"),
        )
        .with_columns((pl.col("nh_sd") / pl.col("nh_n").sqrt()).alias("nh_se"))
    )
    _hint_avg = (
        t1_df.filter(
            pl.col("instruction") == "instruction_hint",
            pl.col("model").is_in(_hint_models),
        )
        .group_by("model")
        .agg(
            pl.col("score").mean().alias("hint_if"),
            pl.col("score").std().alias("h_sd"),
            pl.col("score").len().alias("h_n"),
        )
        .with_columns((pl.col("h_sd") / pl.col("h_n").sqrt()).alias("h_se"))
    )

    _hint_effect_df = (
        _no_hint_avg.join(_hint_avg, on="model")
        .with_columns(
            (pl.col("hint_if") - pl.col("no_hint_if")).alias("hint_effect"),
            (pl.col("nh_se").pow(2) + pl.col("h_se").pow(2))
            .sqrt()
            .alias("se_effect"),
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .with_columns(
            (pl.col("hint_effect") - 1.96 * pl.col("se_effect")).alias(
                "effect_ci_lo"
            ),
            (pl.col("hint_effect") + 1.96 * pl.col("se_effect")).alias(
                "effect_ci_hi"
            ),
        )
        .select(
            [
                "display_name",
                "no_hint_if",
                "hint_if",
                "hint_effect",
                "effect_ci_lo",
                "effect_ci_hi",
            ]
        )
        .with_columns(
            pl.col("no_hint_if").round(3),
            pl.col("hint_if").round(3),
            pl.col("hint_effect").round(3),
            pl.col("effect_ci_lo").round(3),
            pl.col("effect_ci_hi").round(3),
        )
        .sort("hint_effect", descending=True)
    )

    # Paired t-test across models
    _effects = _hint_effect_df["hint_effect"].to_numpy()
    _n_m = len(_effects)
    _t_stat, _p_val = stats.ttest_1samp(_effects, 0)
    _df = _n_m - 1
    _t_crit = stats.t.ppf(0.975, _df)
    _mean_e = float(np.mean(_effects))
    _se_e = float(np.std(_effects, ddof=1) / np.sqrt(_n_m))

    mo.vstack(
        [
            mo.md(
                f"**{len(_hint_models)} models** with both hint/no-hint in T=1. "
                f"Grand mean hint effect: **{_mean_e:.3f}** [{_mean_e - _t_crit * _se_e:.3f}, {_mean_e + _t_crit * _se_e:.3f}], "
                f"t({_df}) = {_t_stat:.2f}, p = {_p_val:.3f}"
            ),
            _hint_effect_df,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §3.5 — Self-Prediction Protocol
    """)
    return


@app.cell
def _(DISPLAY_NAMES, mo, pl, pred_df):
    mo.md("### §3.5a — Self-prediction accuracy per model")

    _metrics = [
        ("score_prediction_accuracy", "pred_accuracy"),
        ("score_prediction_instruction", "predicted_if"),
        ("score_instruction_following", "actual_if"),
    ]

    _agg_exprs = []
    for _src, _alias in _metrics:
        _agg_exprs.extend(
            [
                pl.col(_src).mean().alias(_alias),
                pl.col(_src).std().alias(f"{_alias}_sd"),
                pl.col(_src).len().alias(f"{_alias}_n"),
            ]
        )

    _pred_summary = pred_df.group_by("model").agg(_agg_exprs)

    # Compute SE and CIs for each metric
    for _, _alias in _metrics:
        _pred_summary = _pred_summary.with_columns(
            (pl.col(f"{_alias}_sd") / pl.col(f"{_alias}_n").sqrt()).alias(
                f"{_alias}_se"
            ),
        )
        _pred_summary = _pred_summary.with_columns(
            (pl.col(_alias) - 1.96 * pl.col(f"{_alias}_se")).alias(
                f"{_alias}_ci_lo"
            ),
            (pl.col(_alias) + 1.96 * pl.col(f"{_alias}_se")).alias(
                f"{_alias}_ci_hi"
            ),
        )

    _pred_summary = (
        _pred_summary.with_columns(
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name")
        )
        .select(
            [
                "display_name",
                "pred_accuracy",
                "pred_accuracy_ci_lo",
                "pred_accuracy_ci_hi",
                "predicted_if",
                "predicted_if_ci_lo",
                "predicted_if_ci_hi",
                "actual_if",
                "actual_if_ci_lo",
                "actual_if_ci_hi",
            ]
        )
        .with_columns(
            pl.col("pred_accuracy").round(3),
            pl.col("pred_accuracy_ci_lo").round(3),
            pl.col("pred_accuracy_ci_hi").round(3),
            pl.col("predicted_if").round(3),
            pl.col("predicted_if_ci_lo").round(3),
            pl.col("predicted_if_ci_hi").round(3),
            pl.col("actual_if").round(3),
            pl.col("actual_if_ci_lo").round(3),
            pl.col("actual_if_ci_hi").round(3),
        )
        .sort("pred_accuracy", descending=True)
    )

    # Grand means with CIs (between-cell SE)
    _grand = {}
    for _src, _alias in _metrics:
        _vals = pred_df[_src]
        _mean = _vals.mean()
        _se = _vals.std() / (len(_vals) ** 0.5)
        _grand[_alias] = (_mean, _mean - 1.96 * _se, _mean + 1.96 * _se)

    mo.vstack(
        [
            mo.md(f"""
    Grand means (95% CI across all cells):
    - Prediction accuracy: **{_grand["pred_accuracy"][0]:.3f}** [{_grand["pred_accuracy"][1]:.3f}, {_grand["pred_accuracy"][2]:.3f}]
    - Predicted IF rate: **{_grand["predicted_if"][0]:.3f}** [{_grand["predicted_if"][1]:.3f}, {_grand["predicted_if"][2]:.3f}]
    - Actual IF rate: **{_grand["actual_if"][0]:.3f}** [{_grand["actual_if"][1]:.3f}, {_grand["actual_if"][2]:.3f}]
        """),
            _pred_summary,
        ]
    )
    return


@app.cell
def _(DISPLAY_NAMES, combined_df, mo, np, pl, stats):
    mo.md("### §3.5b — Effect of self-prediction on actual behavior")

    _per_row_delta = combined_df.with_columns(
        (pl.col("prediction_actual_score") - pl.col("behavioral_score")).alias(
            "delta"
        )
    )

    # Per-model: mean delta with between-cell CI
    _per_model_delta = (
        _per_row_delta.group_by("model")
        .agg(
            pl.col("behavioral_score").mean().alias("behavioral"),
            pl.col("prediction_actual_score").mean().alias("prediction_actual"),
            pl.col("delta").mean().alias("mean_delta"),
            pl.col("delta").std().alias("delta_sd"),
            pl.col("delta").len().alias("delta_n"),
        )
        .with_columns(
            (pl.col("delta_sd") / pl.col("delta_n").sqrt()).alias("delta_se"),
        )
        .with_columns(
            (pl.col("mean_delta") - 1.96 * pl.col("delta_se")).alias(
                "delta_ci_lo"
            ),
            (pl.col("mean_delta") + 1.96 * pl.col("delta_se")).alias(
                "delta_ci_hi"
            ),
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .select(
            [
                "display_name",
                "behavioral",
                "prediction_actual",
                "mean_delta",
                "delta_ci_lo",
                "delta_ci_hi",
            ]
        )
        .with_columns(
            pl.col("behavioral").round(3),
            pl.col("prediction_actual").round(3),
            pl.col("mean_delta").round(3),
            pl.col("delta_ci_lo").round(3),
            pl.col("delta_ci_hi").round(3),
        )
        .sort("mean_delta")
    )

    # Overall paired t-test on all (model, condition, n_turns) rows
    _b = _per_row_delta["behavioral_score"].to_numpy()
    _pa = _per_row_delta["prediction_actual_score"].to_numpy()
    _t, _p = stats.ttest_rel(_pa, _b)
    _deltas = _pa - _b
    _n = len(_deltas)
    _df = _n - 1
    _t_crit = stats.t.ppf(0.975, _df)
    _mean_d = float(np.mean(_deltas))
    _se_d = float(np.std(_deltas, ddof=1) / np.sqrt(_n))

    mo.vstack(
        [
            mo.md(f"""
    Paired t-test on {_n} cells: mean delta = **{_mean_d:.3f}** [{_mean_d - _t_crit * _se_d:.3f}, {_mean_d + _t_crit * _se_d:.3f}],
    t({_df}) = **{_t:.2f}**, p = **{_p:.4f}**
        """),
            _per_model_delta,
        ]
    )
    return


@app.cell
def _(DISPLAY_NAMES, combined_df, mo, pl, pred_df):
    mo.md("### Appendix — Expanded self-prediction tables")

    # Table 1: Per-model prediction metrics (matches tab_results.tex Self-prediction columns)
    _metrics = [
        ("score_prediction_accuracy", "accuracy"),
        ("score_prediction_instruction", "predicted_if"),
        ("score_instruction_following", "actual_if"),
    ]
    _agg_exprs = []
    for _src, _alias in _metrics:
        _agg_exprs.append(pl.col(_src).mean().alias(_alias))
    _pred_table = (
        pred_df.group_by("model")
        .agg(_agg_exprs)
        .with_columns(pl.col("model").replace(DISPLAY_NAMES).alias("display_name"))
        .sort("accuracy", descending=True)
        .select(["display_name", "accuracy", "predicted_if", "actual_if"])
        .with_columns(
            pl.col("accuracy").round(3),
            pl.col("predicted_if").round(3),
            pl.col("actual_if").round(3),
        )
    )

    # Table 2: Effect of prediction on behavior
    _effect = (
        combined_df.group_by("model")
        .agg(
            pl.col("behavioral_score").mean().alias("behavioral"),
            pl.col("prediction_actual_score").mean().alias("prediction_actual"),
        )
        .with_columns(
            (pl.col("prediction_actual") - pl.col("behavioral")).alias("delta"),
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .sort("delta")
        .select(["display_name", "behavioral", "prediction_actual", "delta"])
        .with_columns(
            pl.col("behavioral").round(3),
            pl.col("prediction_actual").round(3),
            pl.col("delta").round(3),
        )
    )

    mo.vstack(
        [
            mo.md("**Table: Per-model prediction metrics** (for appendix Table app-pred-metrics)"),
            _pred_table,
            mo.md("**Table: Effect of self-prediction on behavior** (for appendix Table app-pred-effect)"),
            _effect,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fixed-output vs Task-based IF rate per model

    Average IF rate per model for fixed-output conditions vs task-based conditions (T=0, no-hint).
    Only models present in both datasets are shown.
    """)
    return


@app.cell
def _(DISPLAY_NAMES, alt, dynamic_df, mo, pl, static_df):
    _static_models = set(static_df["model"].unique().to_list())
    _dynamic_models = set(dynamic_df["model"].unique().to_list())
    _common = sorted(_static_models & _dynamic_models)


    def _avg_if_by_model(df, label):
        return (
            df.filter(pl.col("model").is_in(_common))
            .group_by("model")
            .agg(
                pl.col("score").mean().alias("mean_if"),
                pl.col("score").std().alias("sd"),
                pl.col("score").len().alias("n"),
            )
            .with_columns(
                (pl.col("sd") / pl.col("n").sqrt()).alias("se"),
                pl.lit(label).alias("condition_type"),
                pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
            )
            .with_columns(
                (pl.col("mean_if") - 1.96 * pl.col("se")).alias("ci_lo"),
                (pl.col("mean_if") + 1.96 * pl.col("se")).alias("ci_hi"),
            )
            .select(
                [
                    "model",
                    "display_name",
                    "condition_type",
                    "mean_if",
                    "ci_lo",
                    "ci_hi",
                ]
            )
            .with_columns(
                pl.col("mean_if").round(3),
                pl.col("ci_lo").round(3),
                pl.col("ci_hi").round(3),
            )
        )


    _sd_combined = pl.concat(
        [
            _avg_if_by_model(static_df, "fixed-output"),
            _avg_if_by_model(dynamic_df, "task-based"),
        ]
    )
    _sd_pd = _sd_combined.to_pandas()

    _table_sd = (
        _sd_combined.pivot(
            on="condition_type", index=["model", "display_name"], values="mean_if"
        )
        .with_columns(
            (pl.col("task-based") - pl.col("fixed-output"))
            .round(3)
            .alias("task_based_advantage")
        )
        .sort("task_based_advantage", descending=True)
        .select(
            ["display_name", "fixed-output", "task-based", "task_based_advantage"]
        )
    )

    _bars_sd = (
        alt.Chart(_sd_pd)
        .mark_bar()
        .encode(
            x=alt.X(
                "mean_if:Q", title="Avg IF Rate", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y("display_name:N", sort="-x", title=None),
            color=alt.Color(
                "condition_type:N",
                scale=alt.Scale(
                    domain=["fixed-output", "task-based"],
                    range=["#e41a1c", "#4daf4a"],
                ),
                legend=alt.Legend(title="Condition type"),
            ),
            yOffset=alt.YOffset(
                "condition_type:N",
                scale=alt.Scale(domain=["fixed-output", "task-based"]),
            ),
            tooltip=[
                "display_name",
                "condition_type",
                alt.Tooltip("mean_if:Q", format=".3f"),
                alt.Tooltip("ci_lo:Q", format=".3f"),
                alt.Tooltip("ci_hi:Q", format=".3f"),
            ],
        )
        .properties(
            width=500,
            height=max(300, len(_common) * 35),
            title="Fixed-output vs Task-based avg IF rate per model",
        )
    )

    mo.vstack([_bars_sd, _table_sd])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Follow-up conditions (variety and classify)

    Three new conditions run on 3 models (gemma-3-12b-it, gpt-5.2, llama-3.3-70b-instruct):
    - **classify_sh_economics**: single-token output (science/humanities), requires reading the question
    - **variety_geography_animals**: 1–3 sentences about animals, ignoring the question
    - **variety_animals_geography**: 1–3 sentences about geography, ignoring the question

    Compared to fixed-output (neutral) and task-based (avg) baselines for the same 3 models.
    """)
    return


@app.cell
def _(DISPLAY_NAMES, alt, dynamic_raw, followup_df, mo, pl, static_raw):
    _fu_models = {"gemma-3-27b-it", "gpt-5.2", "llama-3.3-70b-instruct"}
    _fu_label_map = {
        "classify_sh_economics": "classify",
        "variety_geography_animals": "variety: geo→animals",
        "variety_animals_geography": "variety: animals→geo",
    }
    _cg_order = [
        "fixed-output (neutral)",
        "classify",
        "variety: geo→animals",
        "variety: animals→geo",
        "task-based (avg)",
    ]


    def _base_agg(df, label, model_filter=None):
        _d = (
            df
            if model_filter is None
            else df.filter(pl.col("model").is_in(model_filter))
        )
        return (
            _d.group_by("model")
            .agg(
                pl.col("score").mean().alias("mean_if"),
                pl.col("score").std().alias("sd"),
                pl.col("score").len().alias("n"),
            )
            .with_columns(
                (pl.col("sd") / pl.col("n").sqrt()).alias("se"),
                pl.lit(label).alias("condition_group"),
                pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
            )
            .with_columns(
                (pl.col("mean_if") - 1.96 * pl.col("se")).alias("ci_lo"),
                (pl.col("mean_if") + 1.96 * pl.col("se")).alias("ci_hi"),
            )
        )


    _fu_agg = (
        followup_df.with_columns(
            pl.col("condition").replace(_fu_label_map).alias("condition_group")
        )
        .group_by(["model", "condition_group"])
        .agg(
            pl.col("score").mean().alias("mean_if"),
            pl.col("score").std().alias("sd"),
            pl.col("score").len().alias("n"),
        )
        .with_columns(
            (pl.col("sd") / pl.col("n").sqrt()).alias("se"),
            pl.col("model").replace(DISPLAY_NAMES).alias("display_name"),
        )
        .with_columns(
            (pl.col("mean_if") - 1.96 * pl.col("se")).alias("ci_lo"),
            (pl.col("mean_if") + 1.96 * pl.col("se")).alias("ci_hi"),
        )
    )

    _static_base = _base_agg(
        static_raw.filter(
            pl.col("condition") == "neutral", pl.col("model").is_in(_fu_models)
        ),
        "fixed-output (neutral)",
    )
    _dynamic_base = _base_agg(
        dynamic_raw.filter(pl.col("model").is_in(_fu_models)),
        "task-based (avg)",
    )

    _cols = [
        "model",
        "display_name",
        "condition_group",
        "mean_if",
        "ci_lo",
        "ci_hi",
    ]
    _all_fu = pl.concat(
        [
            _static_base.select(_cols),
            _dynamic_base.select(_cols),
            _fu_agg.select(_cols),
        ]
    ).with_columns(
        pl.col("mean_if").round(3),
        pl.col("ci_lo").round(3),
        pl.col("ci_hi").round(3),
    )

    _all_fu_pd = _all_fu.to_pandas()

    _bars_fu = (
        alt.Chart(_all_fu_pd)
        .mark_bar()
        .encode(
            x=alt.X(
                "mean_if:Q", title="Avg IF Rate", scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y("condition_group:N", sort=_cg_order, title=None),
            color=alt.Color(
                "condition_group:N",
                scale=alt.Scale(
                    domain=_cg_order,
                    range=["#e41a1c", "#ff7f00", "#984ea3", "#a65628", "#4daf4a"],
                ),
                legend=None,
            ),
            tooltip=[
                "display_name",
                "condition_group",
                alt.Tooltip("mean_if:Q", format=".3f"),
            ],
        )
        .facet(facet=alt.Facet("display_name:N", title=None), columns=3)
        .properties(title="Follow-up conditions vs baselines, per model")
    )

    _table_fu = _all_fu.sort(["display_name", "condition_group"])

    mo.vstack([_bars_fu, _table_fu])
    return


@app.cell
def _(DISPLAY_NAMES, dynamic_raw, followup_df, mo, np, pl, static_raw, stats):
    mo.md("### Follow-up conditions: key statistics for §I")
    _fu_models = {"gemma-3-27b-it", "gpt-5.2", "llama-3.3-70b-instruct"}

    _neutral = static_raw.filter(
        pl.col("condition") == "neutral", pl.col("model").is_in(_fu_models)
    ).with_columns(pl.col("n_turns").cast(pl.Int64))
    _dynamic = dynamic_raw.filter(
        pl.col("model").is_in(_fu_models)
    ).with_columns(pl.col("n_turns").cast(pl.Int64))
    _fu = followup_df.with_columns(pl.col("n_turns").cast(pl.Int64))

    # Grand avg IF (across models) per condition
    _labels = [
        ("fixed-output (neutral)", _neutral),
        ("classify_sh_economics", _fu.filter(pl.col("condition") == "classify_sh_economics")),
        ("variety_animals_geography", _fu.filter(pl.col("condition") == "variety_animals_geography")),
        ("variety_geography_animals", _fu.filter(pl.col("condition") == "variety_geography_animals")),
        ("task-based (avg)", _dynamic),
    ]
    _grand_rows = []
    for _lbl, _d in _labels:
        _vals = _d["score"].to_numpy()
        _m = float(np.mean(_vals))
        _se = float(np.std(_vals, ddof=1) / np.sqrt(len(_vals)))
        _ci = 1.96 * _se
        _grand_rows.append({"condition": _lbl, "avg_if": round(_m, 3), "ci_lo": round(_m - _ci, 3), "ci_hi": round(_m + _ci, 3)})
    _grand_df = pl.DataFrame(_grand_rows)

    # N50: first N where avg IF rate < 0.5, per (model, condition)
    _n50_rows = []
    _cond_map = {
        "neutral": _neutral,
        "classify": _fu.filter(pl.col("condition") == "classify_sh_economics"),
        "variety (a→g)": _fu.filter(pl.col("condition") == "variety_animals_geography"),
        "variety (g→a)": _fu.filter(pl.col("condition") == "variety_geography_animals"),
        "task-based": _dynamic,
    }
    for _cond_lbl, _d in _cond_map.items():
        _cell = (
            _d.group_by(["model", "n_turns"])
            .agg(pl.col("score").mean().alias("if_rate"))
            .sort(["model", "n_turns"])
        )
        for _model in sorted(_fu_models):
            _sub = _cell.filter(pl.col("model") == _model)
            _drops = _sub.filter(pl.col("if_rate") < 0.5)["n_turns"].to_list()
            _n50 = int(min(_drops)) if _drops else None
            _avg = float(_d.filter(pl.col("model") == _model)["score"].mean())
            _n50_rows.append({
                "condition": _cond_lbl,
                "model": DISPLAY_NAMES.get(_model, _model),
                "n50": str(_n50) if _n50 else "never",
                "avg_if": round(_avg, 3),
            })
    _n50_df = pl.DataFrame(_n50_rows)

    # t-test: classify avg IF vs neutral avg IF (paired by model)
    _dyn_per_m = _dynamic.group_by("model").agg(pl.col("score").mean().alias("dyn"))
    _cls_per_m = _fu.filter(pl.col("condition") == "classify_sh_economics").group_by("model").agg(pl.col("score").mean().alias("cls"))
    _neu_per_m = _neutral.group_by("model").agg(pl.col("score").mean().alias("neu"))
    _var_per_m = (_fu.filter(pl.col("condition").str.starts_with("variety"))
                  .group_by("model").agg(pl.col("score").mean().alias("var")))
    _joined = _neu_per_m.join(_cls_per_m, on="model").join(_dyn_per_m, on="model").join(_var_per_m, on="model")

    _t_cls_neu, _p_cls_neu = stats.ttest_rel(_joined["cls"].to_numpy(), _joined["neu"].to_numpy())
    _t_var_dyn, _p_var_dyn = stats.ttest_rel(_joined["var"].to_numpy(), _joined["dyn"].to_numpy())

    mo.vstack([
        mo.md(f"""
**Grand avg IF rate:**

{_grand_df.to_pandas().to_string(index=False)}

**classify vs neutral** (paired t): t({len(_joined)-1})={_t_cls_neu:.2f}, p={_p_cls_neu:.3f}
**variety vs task-based** (paired t): t({len(_joined)-1})={_t_var_dyn:.2f}, p={_p_var_dyn:.3f}
        """),
        mo.md("**N50 per model and condition:**"),
        _n50_df.pivot(on="condition", values=["n50", "avg_if"], index="model"),
    ])


if __name__ == "__main__":
    app.run()
