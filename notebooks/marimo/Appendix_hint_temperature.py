import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import numpy as np
    import polars as pl
    from pathlib import Path
    from scipy import stats as scipy_stats
    import json

    return Path, json, mlines, mo, np, pl, plt, scipy_stats


@app.cell
def _(Path):
    ROOT = Path(__file__).resolve().parent.parent.parent
    STATIC_PATH = ROOT / "outputs" / "viz" / "static"
    T1_PATH = ROOT / "outputs" / "viz" / "static-T1"
    OUT_DIR = ROOT / "outputs" / "plots" / "appendix"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR, ROOT, STATIC_PATH, T1_PATH


@app.cell
def _():
    TOKEN_CONDITIONS = {"token_states_countries", "token_countries_states"}
    MODEL_ALIASES_T1 = {"claude-sonnet-4.6": "claude-4.6-sonnet"}
    DISPLAY_NAMES = {
        "claude-4.6-sonnet": "Claude 4.6 Sonnet",
        "gemini-2.5-flash": "Gemini 2.5 Flash",
        "gemma-3-12b-it": "Gemma-3 12B",
        "gemma-3-27b-it": "Gemma-3 27B",
        "gpt-5.2": "GPT-5.2",
        "llama-3.3-70b-instruct": "Llama 3.3 70B",
    }
    COMMON_MODELS = list(DISPLAY_NAMES.keys())
    N_ORDER = [1, 2, 3, 4, 6, 8, 11, 16, 22, 31, 43, 50]
    PALETTE_T0 = "#4c78a8"
    PALETTE_T1 = "#f58518"
    PALETTE_HINT = "#54a24b"
    PALETTE_NOHINT = "#4c78a8"
    return (
        COMMON_MODELS,
        DISPLAY_NAMES,
        MODEL_ALIASES_T1,
        N_ORDER,
        PALETTE_HINT,
        PALETTE_NOHINT,
        PALETTE_T0,
        PALETTE_T1,
        TOKEN_CONDITIONS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Appendix §D & §E: Hint effect and Temperature comparison

    - **§D**: Effect of the hardcoding hint on instruction-following (neutral condition, T=1, 15 trials/cell)
    - **§E**: T=0 vs T=1 comparison on 6 common models, no-hint instruction

    Plots are saved to `outputs/plots/appendix/`.
    """)
    return


@app.cell
def _(MODEL_ALIASES_T1, STATIC_PATH, T1_PATH, TOKEN_CONDITIONS, pl):
    def _load_t0():
        return pl.read_parquet(STATIC_PATH / "evals.parquet").filter(
            pl.col("instruction") == "instruction_no_hint",
            ~pl.col("condition").is_in(TOKEN_CONDITIONS),
        )

    def _load_t1():
        return (
            pl.read_parquet(T1_PATH / "evals.parquet")
            .with_columns(pl.col("model").replace(MODEL_ALIASES_T1))
            .filter(~pl.col("condition").is_in(TOKEN_CONDITIONS))
        )

    t0_df = _load_t0()
    t1_df = _load_t1()
    return t0_df, t1_df


@app.cell
def _(pl):
    def aggregate_by_n(df, group_cols=None):
        if group_cols is None:
            group_cols = ["n_turns"]
        return (
            df.with_columns(pl.col("n_turns").cast(pl.Int64))
            .group_by(group_cols)
            .agg(
                pl.col("score").mean().alias("if_rate"),
                pl.col("score").std().alias("sd"),
                pl.col("score").len().alias("n"),
            )
            .with_columns((pl.col("sd") / pl.col("n").sqrt()).alias("se"))
            .sort("n_turns")
        )

    return (aggregate_by_n,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## §D — Effect of the hardcoding hint (neutral condition)")
    return


@app.cell
def _(COMMON_MODELS, DISPLAY_NAMES, mo, np, pl, scipy_stats, t1_df):
    _hint_models = set(
        t1_df.filter(pl.col("instruction") == "instruction_hint")["model"].unique().to_list()
    )
    _nohint_models = set(
        t1_df.filter(pl.col("instruction") == "instruction_no_hint")["model"].unique().to_list()
    )
    _both_models = sorted(_hint_models & _nohint_models & set(COMMON_MODELS))
    _neutral = t1_df.filter(pl.col("condition") == "neutral", pl.col("model").is_in(_both_models))

    _nh = _neutral.filter(pl.col("instruction") == "instruction_no_hint")
    _h = _neutral.filter(pl.col("instruction") == "instruction_hint")

    _nh_avg = _nh.group_by("model").agg(pl.col("score").mean().alias("nh_if"))
    _h_avg = _h.group_by("model").agg(pl.col("score").mean().alias("h_if"))
    _joined = _nh_avg.join(_h_avg, on="model").with_columns(
        (pl.col("h_if") - pl.col("nh_if")).alias("delta")
    )

    _deltas = _joined["delta"].to_numpy()
    _t_stat, _p_val = scipy_stats.ttest_1samp(_deltas, 0)
    _n_m = len(_deltas)
    _t_crit = scipy_stats.t.ppf(0.975, _n_m - 1)
    _mean_d = float(np.mean(_deltas))
    _se_d = float(np.std(_deltas, ddof=1) / np.sqrt(_n_m))

    hint_stats = {
        "n_models": _n_m,
        "mean_delta": round(_mean_d, 3),
        "ci_lo": round(_mean_d - _t_crit * _se_d, 3),
        "ci_hi": round(_mean_d + _t_crit * _se_d, 3),
        "t_stat": round(float(_t_stat), 2),
        "p_val": round(float(_p_val), 4),
        "grand_no_hint_if": round(float(_nh["score"].mean()), 3),
        "grand_hint_if": round(float(_h["score"].mean()), 3),
        "per_model": {
            DISPLAY_NAMES.get(r["model"], r["model"]): {
                "no_hint": round(r["nh_if"], 3),
                "hint": round(r["h_if"], 3),
                "delta": round(r["delta"], 3),
            }
            for r in _joined.iter_rows(named=True)
        },
    }

    mo.md(f"""
    **Hint effect statistics (neutral condition, {_n_m} models):**

    | | No hint | Hint | Δ |
    |---|---|---|---|
    | Grand mean | {hint_stats['grand_no_hint_if']:.3f} | {hint_stats['grand_hint_if']:.3f} | {hint_stats['mean_delta']:+.3f} |

    Mean Δ = {hint_stats['mean_delta']:+.3f} (95% CI: [{hint_stats['ci_lo']:.3f}, {hint_stats['ci_hi']:.3f}]);
    t({_n_m-1}) = {hint_stats['t_stat']}, p = {hint_stats['p_val']}

    **Per-model:**

    | Model | No hint | Hint | Δ |
    |---|---|---|---|
    {"".join(f"| {name} | {v['no_hint']:.3f} | {v['hint']:.3f} | {v['delta']:+.3f} |" + chr(10) for name, v in sorted(hint_stats['per_model'].items(), key=lambda x: -x[1]['delta']))}
    """)
    return (hint_stats,)


@app.cell
def _(COMMON_MODELS, N_ORDER, PALETTE_HINT, PALETTE_NOHINT, OUT_DIR, aggregate_by_n, mo, pl, plt, t1_df):
    _hint_models = set(
        t1_df.filter(pl.col("instruction") == "instruction_hint")["model"].unique().to_list()
    )
    _nohint_models = set(
        t1_df.filter(pl.col("instruction") == "instruction_no_hint")["model"].unique().to_list()
    )
    _both_models = sorted(_hint_models & _nohint_models & set(COMMON_MODELS))
    _neutral = t1_df.filter(pl.col("condition") == "neutral", pl.col("model").is_in(_both_models))

    _nh_ns = set(_neutral.filter(pl.col("instruction") == "instruction_no_hint")["n_turns"].unique().to_list())
    _h_ns = set(_neutral.filter(pl.col("instruction") == "instruction_hint")["n_turns"].unique().to_list())
    _common_ns = _nh_ns & _h_ns

    _nh_agg = aggregate_by_n(
        _neutral.filter(pl.col("instruction") == "instruction_no_hint", pl.col("n_turns").is_in(_common_ns))
    )
    _h_agg = aggregate_by_n(
        _neutral.filter(pl.col("instruction") == "instruction_hint", pl.col("n_turns").is_in(_common_ns))
    )

    _fig_d1, _ax = plt.subplots(figsize=(4.5, 3.0))
    for _agg_df, _color, _label in [(_nh_agg, PALETTE_NOHINT, "No hint"), (_h_agg, PALETTE_HINT, "With hint")]:
        _x = _agg_df["n_turns"].to_numpy()
        _y = _agg_df["if_rate"].to_numpy()
        _se = _agg_df["se"].to_numpy()
        _ax.plot(_x, _y, "o-", color=_color, linewidth=1.5, markersize=4, label=_label)
        _ax.fill_between(_x, _y - 1.96 * _se, _y + 1.96 * _se, color=_color, alpha=0.15)

    _ax.set_xlabel("$N$ (hardcoded turns)")
    _ax.set_ylabel("IF rate")
    _ax.set_ylim(-0.05, 1.05)
    _ax.set_xscale("log")
    _ax.set_xticks(N_ORDER)
    _ax.set_xticklabels(N_ORDER, fontsize=7)
    _ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    _ax.legend(framealpha=0.9)
    _ax.set_title(f"Hint vs no-hint, neutral condition ({len(_both_models)} models, T=1)")
    _fig_d1.tight_layout()
    _fig_d1.savefig(OUT_DIR / "d1_hint_vs_nohint_curves.png", dpi=150)
    mo.md("**D1 saved** → `outputs/plots/appendix/d1_hint_vs_nohint_curves.png`")
    _fig_d1


@app.cell
def _(COMMON_MODELS, DISPLAY_NAMES, PALETTE_HINT, PALETTE_NOHINT, OUT_DIR, mo, pl, plt, t1_df):
    _hint_models = set(
        t1_df.filter(pl.col("instruction") == "instruction_hint")["model"].unique().to_list()
    )
    _nohint_models = set(
        t1_df.filter(pl.col("instruction") == "instruction_no_hint")["model"].unique().to_list()
    )
    _both_models = sorted(_hint_models & _nohint_models & set(COMMON_MODELS))
    _neutral = t1_df.filter(pl.col("condition") == "neutral", pl.col("model").is_in(_both_models))

    _nh_avg = (
        _neutral.filter(pl.col("instruction") == "instruction_no_hint")
        .group_by("model").agg(pl.col("score").mean().alias("nh_if"))
    )
    _h_avg = (
        _neutral.filter(pl.col("instruction") == "instruction_hint")
        .group_by("model").agg(pl.col("score").mean().alias("h_if"))
    )
    _joined = _nh_avg.join(_h_avg, on="model")

    _fig_d2, _ax = plt.subplots(figsize=(3.5, 3.5))
    _ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, alpha=0.6, label="no effect")
    for _row in _joined.iter_rows(named=True):
        _name = DISPLAY_NAMES.get(_row["model"], _row["model"])
        _x, _y = _row["nh_if"], _row["h_if"]
        _color = PALETTE_HINT if _y >= _x else PALETTE_NOHINT
        _ax.scatter(_x, _y, s=40, color=_color, zorder=3)
        _ax.annotate(_name, (_x, _y), textcoords="offset points", xytext=(4, 2), fontsize=6.5)

    _ax.set_xlabel("Avg IF rate — no hint")
    _ax.set_ylabel("Avg IF rate — with hint")
    _ax.set_xlim(-0.05, 1.05)
    _ax.set_ylim(-0.05, 1.05)
    _ax.set_title("Effect of hint on instruction-following (T=1)")
    _fig_d2.tight_layout()
    _fig_d2.savefig(OUT_DIR / "d2_hint_effect_by_model.png", dpi=150)
    mo.md("**D2 saved** → `outputs/plots/appendix/d2_hint_effect_by_model.png`")
    _fig_d2


@app.cell(hide_code=True)
def _(mo):
    mo.md("## §E — Temperature comparison (T=0 vs T=1)")
    return


@app.cell
def _(COMMON_MODELS, DISPLAY_NAMES, mo, np, pl, scipy_stats, t0_df, t1_df):
    _t1_nh = t1_df.filter(
        pl.col("instruction") == "instruction_no_hint",
        pl.col("model").is_in(COMMON_MODELS),
    )
    _t0_core = t0_df.filter(pl.col("model").is_in(COMMON_MODELS))

    _t0_avg = _t0_core.group_by("model").agg(pl.col("score").mean().alias("t0_if"))
    _t1_avg = _t1_nh.group_by("model").agg(pl.col("score").mean().alias("t1_if"))
    _joined = _t0_avg.join(_t1_avg, on="model").with_columns(
        (pl.col("t1_if") - pl.col("t0_if")).alias("delta")
    )

    _deltas = _joined["delta"].to_numpy()
    _t_stat, _p_val = scipy_stats.ttest_1samp(_deltas, 0)
    _n_m = len(_deltas)
    _t_crit = scipy_stats.t.ppf(0.975, _n_m - 1)
    _mean_d = float(np.mean(_deltas))
    _se_d = float(np.std(_deltas, ddof=1) / np.sqrt(_n_m))

    _t0_v = _joined.sort("model")["t0_if"].to_numpy()
    _t1_v = _joined.sort("model")["t1_if"].to_numpy()
    _r, _p_r = scipy_stats.pearsonr(_t0_v, _t1_v)

    _sd_d = float(np.std(_deltas, ddof=1))
    _excl_outlier = _deltas[_deltas != _deltas.max()]
    _median_d = float(np.median(_deltas))

    temp_stats = {
        "n_models": _n_m,
        "mean_delta": round(_mean_d, 3),
        "sd_delta": round(_sd_d, 3),
        "median_delta": round(_median_d, 3),
        "ci_lo": round(_mean_d - _t_crit * _se_d, 3),
        "ci_hi": round(_mean_d + _t_crit * _se_d, 3),
        "t_stat": round(float(_t_stat), 2),
        "p_val": round(float(_p_val), 4),
        "pearson_r": round(float(_r), 3),
        "pearson_p": round(float(_p_r), 4),
        "per_model": {
            DISPLAY_NAMES.get(r["model"], r["model"]): {
                "t0": round(r["t0_if"], 3),
                "t1": round(r["t1_if"], 3),
                "delta": round(r["delta"], 3),
            }
            for r in _joined.iter_rows(named=True)
        },
    }

    mo.md(f"""
    **Temperature comparison statistics ({_n_m} models):**

    | Statistic | Value |
    |---|---|
    | Mean Δ (T1−T0) | {temp_stats['mean_delta']:+.3f} (95% CI: [{temp_stats['ci_lo']:.3f}, {temp_stats['ci_hi']:.3f}]) |
    | **SD of Δ** | **{temp_stats['sd_delta']:.3f}** |
    | Median Δ | {temp_stats['median_delta']:+.3f} |
    | t({_n_m-1}) | {temp_stats['t_stat']}, p = {temp_stats['p_val']} |
    | Pearson r(T0, T1) | {temp_stats['pearson_r']:.3f} (p = {temp_stats['pearson_p']:.3f}) |

    SD = {temp_stats['sd_delta']:.2f} vs mean = {temp_stats['mean_delta']:+.3f}: high variance driven by outlier.
    Median excluding max-outlier models: {np.median(_excl_outlier):+.3f}

    **Per-model:**

    | Model | T=0 | T=1 | Δ |
    |---|---|---|---|
    {"".join(f"| {name} | {v['t0']:.3f} | {v['t1']:.3f} | {v['delta']:+.3f} |" + chr(10) for name, v in sorted(temp_stats['per_model'].items(), key=lambda x: -x[1]['t0']))}
    """)
    return (temp_stats,)


@app.cell
def _(COMMON_MODELS, DISPLAY_NAMES, PALETTE_T0, PALETTE_T1, OUT_DIR, mlines, mo, pl, plt, t0_df, t1_df):
    _t1_nh = t1_df.filter(
        pl.col("instruction") == "instruction_no_hint",
        pl.col("model").is_in(COMMON_MODELS),
    )
    _t0_core = t0_df.filter(pl.col("model").is_in(COMMON_MODELS))

    _n_models = len(COMMON_MODELS)
    _ncols = 3
    _nrows = (_n_models + _ncols - 1) // _ncols
    _fig_e1, _axes = plt.subplots(_nrows, _ncols, figsize=(6.5, _nrows * 2.0), sharey=True)
    _axes_flat = _axes.flatten()

    for _idx, _model in enumerate(sorted(COMMON_MODELS)):
        _ax = _axes_flat[_idx]
        _name = DISPLAY_NAMES.get(_model, _model)
        for _df_src, _color, _label, _lw in [
            (_t0_core, PALETTE_T0, "T=0 (35 trials)", 1.8),
            (_t1_nh, PALETTE_T1, "T=1 (15 trials)", 1.4),
        ]:
            _sub = (
                _df_src.filter(pl.col("model") == _model)
                .with_columns(pl.col("n_turns").cast(pl.Int64))
                .group_by("n_turns")
                .agg(pl.col("score").mean().alias("if_rate"), pl.col("score").std().alias("sd"), pl.col("score").len().alias("n"))
                .with_columns((pl.col("sd") / pl.col("n").sqrt()).alias("se"))
                .sort("n_turns")
            )
            if len(_sub) == 0:
                continue
            _x = _sub["n_turns"].to_numpy()
            _y = _sub["if_rate"].to_numpy()
            _se = _sub["se"].to_numpy()
            _ax.plot(_x, _y, "o-", color=_color, linewidth=_lw, markersize=3, label=_label)
            _ax.fill_between(_x, _y - 1.96 * _se, _y + 1.96 * _se, color=_color, alpha=0.12)
        _ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        _ax.set_ylim(-0.05, 1.05)
        _ax.set_xscale("log")
        _ax.set_xticks([1, 4, 16, 50])
        _ax.set_xticklabels(["1", "4", "16", "50"], fontsize=7)
        _ax.set_title(_name, fontsize=8)
        if _idx % _ncols == 0:
            _ax.set_ylabel("IF rate", fontsize=8)
        if _idx >= (_nrows - 1) * _ncols:
            _ax.set_xlabel("$N$", fontsize=8)

    for _idx in range(_n_models, len(_axes_flat)):
        _axes_flat[_idx].set_visible(False)

    _handles = [
        mlines.Line2D([], [], color=PALETTE_T0, marker="o", markersize=4, linewidth=1.5, label="T=0 (35 trials)"),
        mlines.Line2D([], [], color=PALETTE_T1, marker="o", markersize=4, linewidth=1.5, label="T=1 (15 trials)"),
    ]
    _fig_e1.legend(handles=_handles, loc="lower right", bbox_to_anchor=(0.98, 0.02), fontsize=8)
    _fig_e1.suptitle("T=0 vs T=1: IF rate curves for common models", fontsize=9, y=1.01)
    _fig_e1.tight_layout()
    _fig_e1.savefig(OUT_DIR / "e1_t0_vs_t1_curves.png", dpi=150, bbox_inches="tight")
    mo.md("**E1 saved** → `outputs/plots/appendix/e1_t0_vs_t1_curves.png`")
    _fig_e1


@app.cell
def _(COMMON_MODELS, DISPLAY_NAMES, PALETTE_T0, OUT_DIR, mo, pl, plt, scipy_stats, t0_df, t1_df):
    _t1_nh = t1_df.filter(
        pl.col("instruction") == "instruction_no_hint",
        pl.col("model").is_in(COMMON_MODELS),
    )
    _t0_core = t0_df.filter(pl.col("model").is_in(COMMON_MODELS))

    _t0_avg = _t0_core.group_by("model").agg(pl.col("score").mean().alias("t0_if"))
    _t1_avg = _t1_nh.group_by("model").agg(pl.col("score").mean().alias("t1_if"))
    _joined = _t0_avg.join(_t1_avg, on="model")

    _t0_v = _joined.sort("model")["t0_if"].to_numpy()
    _t1_v = _joined.sort("model")["t1_if"].to_numpy()
    _r, _ = scipy_stats.pearsonr(_t0_v, _t1_v)

    _fig_e2, _ax = plt.subplots(figsize=(3.5, 3.5))
    _ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8, alpha=0.6)
    for _row in _joined.iter_rows(named=True):
        _name = DISPLAY_NAMES.get(_row["model"], _row["model"])
        _x, _y = _row["t0_if"], _row["t1_if"]
        _ax.scatter(_x, _y, s=45, color=PALETTE_T0, zorder=3)
        _ax.annotate(_name, (_x, _y), textcoords="offset points", xytext=(4, 2), fontsize=6.5)

    _ax.set_xlabel("Avg IF rate — T=0")
    _ax.set_ylabel("Avg IF rate — T=1 (no hint)")
    _ax.set_xlim(-0.05, 1.05)
    _ax.set_ylim(-0.05, 1.05)
    _ax.set_title(f"T=0 vs T=1 correlation (r={_r:.2f})")
    _fig_e2.tight_layout()
    _fig_e2.savefig(OUT_DIR / "e2_t0_vs_t1_scatter.png", dpi=150)
    mo.md("**E2 saved** → `outputs/plots/appendix/e2_t0_vs_t1_scatter.png`")
    _fig_e2


@app.cell
def _(ROOT, hint_stats, json, temp_stats):
    _out_json = ROOT / "outputs" / "paper_stats" / "appendix_de_stats.json"
    _out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(_out_json, "w") as _f:
        json.dump({"hint": hint_stats, "temperature": temp_stats}, _f, indent=2)
    print(f"Stats written to {_out_json}")
    return


if __name__ == "__main__":
    app.run()
