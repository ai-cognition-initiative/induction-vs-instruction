import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy import stats

    return Path, alt, mo, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Question variation analysis

    This notebook investigates whether **question content** (seed) drives variation in model behavior, or whether the variance is purely stochastic sampling noise (T=1).

    All analysis uses the **neutral condition only** and operates **within each model independently** — cross-model comparison is not meaningful because models have completely different transition curves.

    **The core problem**: At T=1, we observe variation across seeds at any given N. But this variation could come from two very different sources:
    1. **Stable seed effect** — question content makes some seeds inherently easier/harder, shifting the whole transition curve
    2. **Stochastic noise** — T=1 sampling produces random variation that looks like a seed effect at one N but doesn't persist at other N values

    The decisive test is **cross-N stability**: if the same seeds are consistently above or below average across multiple N values, question content matters. If seed rankings shuffle randomly across N, it's just noise.

    Four checks:

    1. **Seed heterogeneity** (within-N) — do seeds have different T-rates? *Necessary but not sufficient.*
    2. **Unanimous seeds** (within-N) — are any seeds 100%T or 100%P? *Suggestive but not conclusive alone.*
    3. **Overdispersion trajectory** (within-N, all N) — where in the sweep do seed effects appear?
    4. **Cross-N seed stability** (decisive) — do seed effects *persist* across N values?
    """)
    return


@app.cell(hide_code=True)
def _(Path, pd):
    from inspect_ai.analysis import (
        EvalModel,
        SampleSummary,
        samples_df,
    )

    _root = Path(__file__).resolve().parent.parent.parent
    _log_dir = str(_root / "logs" / "protocol1-dynamic-sweep-2")

    _samples = samples_df(logs=_log_dir, columns=SampleSummary + EvalModel)

    # Coalesce score columns into a single binary score.
    # inspect-ai uses "C" (CORRECT = follows instruction) and "I" (INCORRECT = follows pattern).
    _SCORE_MAP = {"C": 1.0, "I": 0.0, "P": 0.5}
    _score_cols = [c for c in _samples.columns if c.startswith("score_")]
    _samples["score"] = pd.NA
    for _col in _score_cols:
        _mapped = _samples[_col].map(
            lambda v: _SCORE_MAP.get(str(v)) if pd.notna(v) else pd.NA
        )
        _samples["score"] = _samples["score"].fillna(_mapped)
    _samples["score"] = _samples["score"].astype(float)

    _samples["model"] = _samples["model"].astype(object).str.split("/").str[-1]

    _samples = _samples.rename(
        columns={
            "metadata_trial_index": "trial_index",
            "metadata_condition": "condition",
            "metadata_n_turns": "n_turns",
            "metadata_instruction_template": "instruction",
        }
    )
    _samples["n_turns"] = pd.to_numeric(_samples["n_turns"], errors="coerce").astype(
        int
    )

    # All analysis uses the neutral condition only.
    samples_neutral = _samples[
        #  (_samples["condition"] == "neutral")&
        (_samples["instruction"] == "instruction_no_hint")
    ].copy()
    samples_neutral["is_T"] = (samples_neutral["score"] >= 0.5).astype(int)

    n_seeds = samples_neutral["trial_index"].nunique()
    all_models = sorted(samples_neutral["model"].unique())
    all_n_values = sorted(samples_neutral["n_turns"].unique())
    return all_models, all_n_values, n_seeds, samples_neutral


@app.cell(hide_code=True)
def _(all_models, all_n_values, mo, n_seeds, samples_neutral):
    _epochs_per_cell = (
        samples_neutral.groupby(["model", "n_turns", "trial_index"]).size().median()
    )
    mo.md(f"""
    **Data summary (neutral condition, no hint):**
    - **{n_seeds}** unique seeds (trial indices)
    - **{len(all_models)}** models: {", ".join(all_models)}
    - **{len(all_n_values)}** N values: {all_n_values}
    - ~**{int(_epochs_per_cell)}** epochs per (model, N, seed) cell
    """)
    return


@app.cell
def _(all_models, all_n_values, mo):
    model_selector = mo.ui.dropdown(
        options=all_models,
        value=all_models[0],
        label="Model",
    )
    n_selector = mo.ui.dropdown(
        options=[str(n) for n in all_n_values],
        value="10",
        label="N value",
    )
    mo.hstack([model_selector, n_selector])
    return model_selector, n_selector


@app.cell
def _(np):
    def icc_oneway(pivot):
        """One-way random ICC(1,1) from a (subjects x raters) array."""
        k = pivot.shape[1]
        ms_b = k * np.var(pivot.mean(axis=1), ddof=1)
        ms_w = np.mean(np.var(pivot, axis=1, ddof=1))
        denom = ms_b + (k - 1) * ms_w
        return float((ms_b - ms_w) / denom) if denom > 0 else 0.0

    return (icc_oneway,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check 1: Seed heterogeneity within model

    At a given N, does each seed have the same T-rate, or do some seeds consistently produce T while others produce P? Under the null (no question effect), all seeds share one probability *p* and each epoch is an independent Bernoulli(*p*) trial. A chi-squared test of homogeneity asks whether the per-seed counts deviate from this expectation.
    """)
    return


@app.cell(hide_code=True)
def _(alt, mo, model_selector, n_selector, np, pd, samples_neutral, stats):
    _model = model_selector.value
    _n = int(n_selector.value)
    _data = samples_neutral[
        (samples_neutral["model"] == _model) & (samples_neutral["n_turns"] == _n)
    ]

    # Per-seed counts
    _seed_stats = (
        _data.groupby("trial_index")
        .agg(n_T=("is_T", "sum"), n_total=("is_T", "count"))
        .reset_index()
    )
    _seed_stats["T_rate"] = _seed_stats["n_T"] / _seed_stats["n_total"]
    _overall_rate = _data["is_T"].mean()
    _n_seeds = len(_seed_stats)

    # Chi-squared test of homogeneity
    # H0: all seeds share the same p = overall_rate
    # Observed: [n_T_i, n_P_i] for each seed
    # Expected: [n_total_i * p, n_total_i * (1-p)] for each seed
    _obs_table = np.column_stack(
        [
            _seed_stats["n_T"].values,
            _seed_stats["n_total"].values - _seed_stats["n_T"].values,
        ]
    )
    _exp_table = np.column_stack(
        [
            _seed_stats["n_total"].values * _overall_rate,
            _seed_stats["n_total"].values * (1 - _overall_rate),
        ]
    )
    # Avoid division by zero when overall rate is 0 or 1
    if 0 < _overall_rate < 1:
        _chi2 = float(np.sum((_obs_table - _exp_table) ** 2 / _exp_table))
        _df = _n_seeds - 1
        _p_val = 1 - stats.chi2.cdf(_chi2, _df)
    else:
        _chi2, _df, _p_val = 0.0, _n_seeds - 1, 1.0

    # Overdispersion ratio: observed variance of seed T-rates / expected binomial variance
    _expected_var = (
        _overall_rate * (1 - _overall_rate) / _seed_stats["n_total"].median()
        if 0 < _overall_rate < 1
        else 0
    )
    _observed_var = _seed_stats["T_rate"].var()
    _dispersion_ratio = (
        _observed_var / _expected_var if _expected_var > 0 else float("nan")
    )

    # Bar chart of per-seed T-rates
    _bar = (
        alt.Chart(_seed_stats)
        .mark_bar()
        .encode(
            x=alt.X("trial_index:O", title="Seed", sort="ascending"),
            y=alt.Y("T_rate:Q", title="T rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "T_rate:Q",
                scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]),
                legend=None,
            ),
            tooltip=["trial_index", "n_T", "n_total", "T_rate:Q"],
        )
        .properties(
            width=600,
            height=250,
            title=f"Per-seed T rate — {_model} at N={_n}",
        )
    )
    _rule = (
        alt.Chart(pd.DataFrame({"y": [_overall_rate]}))
        .mark_rule(strokeDash=[4, 2], color="black")
        .encode(y="y:Q")
    )

    mo.vstack(
        [
            mo.md(f"""
    **Results for {_model} at N={_n}:**
    - Overall T rate: **{_overall_rate:.3f}**
    - Chi-squared test of homogeneity: **χ²={_chi2:.2f}**, df={_df}, p=**{_p_val:.4f}**
    - Overdispersion ratio (observed/expected variance): **{_dispersion_ratio:.2f}** (1.0 = pure binomial noise, >1 = seed effect)

    {"**Significant seed heterogeneity** (p < 0.05) — seeds have different T-rates beyond sampling noise." if _p_val < 0.05 else "No significant heterogeneity — consistent with all seeds sharing one T-rate."}
    {"**Overdispersed** — seed-level variance exceeds binomial expectation." if _dispersion_ratio > 1.5 else ""}
        """),
            _bar + _rule,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check 2: Unanimous seeds (within-N)

    For a given model at a given N, are any seeds **100% T** or **100% P** across all epochs? At an N where the overall rate is moderate (say 30-70%), a seed that goes k/k T has probability p^k under the null — extremely unlikely for large k.

    **Caveat**: A seed unanimous at one N may not be at another — that would be T=1 noise, not a stable question content effect. Check 4 tests whether seed effects persist across the N sweep.
    """)
    return


@app.cell
def _(alt, mo, model_selector, n_selector, samples_neutral):
    _model = model_selector.value
    _n = int(n_selector.value)
    _data = samples_neutral[
        (samples_neutral["model"] == _model) & (samples_neutral["n_turns"] == _n)
    ]

    _seed_stats = (
        _data.groupby("trial_index")
        .agg(n_T=("is_T", "sum"), n_total=("is_T", "count"))
        .reset_index()
    )
    _seed_stats["T_rate"] = _seed_stats["n_T"] / _seed_stats["n_total"]
    _overall_rate = _data["is_T"].mean()

    _unanimous_T = _seed_stats[_seed_stats["n_T"] == _seed_stats["n_total"]]
    _unanimous_P = _seed_stats[_seed_stats["n_T"] == 0]
    _k = int(_seed_stats["n_total"].median())

    # Probability under null
    _p_all_T = _overall_rate**_k if 0 < _overall_rate < 1 else float("nan")
    _p_all_P = (1 - _overall_rate) ** _k if 0 < _overall_rate < 1 else float("nan")

    # Heatmap: seed × epoch (showing individual trial outcomes)
    _data_sorted = _data.sort_values(["trial_index", "is_T"])
    _data_sorted["epoch"] = _data_sorted.groupby("trial_index").cumcount()
    _heatmap = (
        alt.Chart(_data_sorted)
        .mark_rect()
        .encode(
            x=alt.X("trial_index:O", title="Seed", sort="ascending"),
            y=alt.Y("epoch:O", title="Epoch"),
            color=alt.Color(
                "is_T:Q",
                scale=alt.Scale(domain=[0, 1], range=["#e15759", "#4e79a7"]),
                title="T=1, P=0",
            ),
            tooltip=["trial_index", "epoch", "is_T"],
        )
        .properties(
            width=600,
            height=max(150, _k * 8),
            title=f"Seed × Epoch outcomes — {_model} at N={_n}",
        )
    )

    mo.vstack(
        [
            mo.md(f"""
    **Results for {_model} at N={_n}:**
    - **{_k}** epochs per seed
    - Overall T rate: **{_overall_rate:.3f}**
    - **Unanimous T seeds: {len(_unanimous_T)}** {f"(seeds: {_unanimous_T['trial_index'].tolist()})" if len(_unanimous_T) > 0 else ""}
    - **Unanimous P seeds: {len(_unanimous_P)}** {f"(seeds: {_unanimous_P['trial_index'].tolist()})" if len(_unanimous_P) > 0 else ""}
    - P(seed is all-T under null): {_p_all_T:.2e} per seed
    - P(seed is all-P under null): {_p_all_P:.2e} per seed

    {"**Strong evidence that question content matters** — unanimous seeds exist where the null probability is vanishingly small." if (len(_unanimous_T) > 0 or len(_unanimous_P) > 0) and 0.2 < _overall_rate < 0.8 else "No unanimous seeds at this N (or overall rate is too extreme to be informative). Try an N near the model's transition." if not (0.2 < _overall_rate < 0.8) else "No unanimous seeds found."}
        """),
            _heatmap,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check 3: Overdispersion trajectory (within-N, all N)

    For each model at each N, compute the **overdispersion ratio**: how much more variable are per-seed T-rates than expected from pure binomial sampling?

    - Ratio ≈ 1.0 → seed T-rates are consistent with independent coin flips (no seed effect)
    - Ratio > 1.0 → seeds differ beyond what sampling noise predicts

    This is plotted alongside the T-rate trajectory to see where within-N seed effects appear. **However, overdispersion at a single N is necessary but not sufficient** — it could be T=1 noise that happens to be coherent within one N but doesn't persist. Check 4 tests cross-N stability.
    """)
    return


@app.cell
def _(all_n_values, alt, mo, np, pd, samples_neutral, stats):
    _records = []
    for _model in samples_neutral["model"].unique():
        for _n in all_n_values:
            _data = samples_neutral[
                (samples_neutral["model"] == _model)
                & (samples_neutral["n_turns"] == _n)
            ]
            if len(_data) < 2:
                continue
            _t_rate = _data["is_T"].mean()
            _seed_stats = (
                _data.groupby("trial_index")
                .agg(n_T=("is_T", "sum"), n_total=("is_T", "count"))
                .reset_index()
            )
            _seed_stats["T_rate"] = _seed_stats["n_T"] / _seed_stats["n_total"]
            _n_seeds = len(_seed_stats)
            _k = int(_seed_stats["n_total"].median())

            # Overdispersion ratio: observed var of seed T-rates
            # vs expected under null (binomial: p(1-p)/k)
            _expected_var = (
                _t_rate * (1 - _t_rate) / _k if 0 < _t_rate < 1 and _k > 0 else 0
            )
            _observed_var = float(_seed_stats["T_rate"].var())
            _odr = _observed_var / _expected_var if _expected_var > 0 else float("nan")

            # Chi-squared test of homogeneity
            if 0 < _t_rate < 1:
                _exp_table = np.column_stack(
                    [
                        _seed_stats["n_total"].values * _t_rate,
                        _seed_stats["n_total"].values * (1 - _t_rate),
                    ]
                )
                _obs_table = np.column_stack(
                    [
                        _seed_stats["n_T"].values,
                        _seed_stats["n_total"].values - _seed_stats["n_T"].values,
                    ]
                )
                _chi2 = float(np.sum((_obs_table - _exp_table) ** 2 / _exp_table))
                _p_val = 1 - stats.chi2.cdf(_chi2, _n_seeds - 1)
            else:
                _chi2, _p_val = 0.0, 1.0

            # Unanimous seeds
            _unan_T = int((_seed_stats["n_T"] == _seed_stats["n_total"]).sum())
            _unan_P = int((_seed_stats["n_T"] == 0).sum())

            _records.append(
                {
                    "model": _model,
                    "n_turns": _n,
                    "overdispersion": round(_odr, 3),
                    "T_rate": round(_t_rate, 3),
                    "epochs_per_seed": _k,
                    "chi2_p": round(_p_val, 6),
                    "significant": _p_val < 0.05,
                    "unanimous_T": _unan_T,
                    "unanimous_P": _unan_P,
                }
            )

    within_n_df = pd.DataFrame(_records)

    _odr_chart = (
        alt.Chart(within_n_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_turns:Q", title="N (number of turns)"),
            y=alt.Y("overdispersion:Q", title="Overdispersion ratio"),
            color=alt.Color("model:N", title="Model"),
            tooltip=[
                "model",
                "n_turns",
                "overdispersion:Q",
                "T_rate:Q",
                "epochs_per_seed:Q",
            ],
        )
        .properties(
            width=700,
            height=300,
            title="Seed overdispersion across N — per model",
        )
    )
    _one_line = (
        alt.Chart(pd.DataFrame({"y": [1.0]}))
        .mark_rule(color="gray", strokeDash=[2, 2])
        .encode(y="y:Q")
    )
    _trate_chart = (
        alt.Chart(within_n_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("n_turns:Q", title="N (number of turns)"),
            y=alt.Y("T_rate:Q", title="T rate"),
            color=alt.Color("model:N", title="Model"),
            tooltip=["model", "n_turns", "T_rate:Q"],
        )
        .properties(
            width=700,
            height=200,
            title="T rate trajectory — per model",
        )
    )

    _k_val = within_n_df["epochs_per_seed"].median()
    mo.vstack(
        [
            alt.vconcat(alt.layer(_odr_chart, _one_line), _trate_chart).resolve_scale(
                x="shared"
            ),
            mo.md(f"""
    **Reading the chart:**
    - Dashed line at 1.0 = expected under pure binomial noise (no seed effect)
    - Values above 1 indicate seeds differ more than expected — but this is **within-N evidence only**
    - Overdispersion peaks near the transition (0.3-0.7 T-rate) where there's room for differentiation
    - With **{int(_k_val)} epochs per seed**: {"the test has reasonable power to detect seed effects" if _k_val >= 5 else "few epochs make it hard to distinguish seed effects from noise within a single N — the cross-N analysis (Check 4) is more informative"}
        """),
        ]
    )
    return (within_n_df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Check 4: Cross-N seed stability (decisive test)

    Checks 1-3 test whether seeds differ **at a single N**. But stochastic variation at T=1 can produce spurious seed effects at any given N that vanish at other N values. A seed might be unanimously T at N=10 but unanimously P at N=15 — that's noise, not a question content effect.

    The decisive test: **are seed effects stable across the N sweep?**

    For each model, we compute per-seed T-rate at each N (averaging over epochs), then subtract the model's mean at that N to get a **residual** — how much above or below average each seed is. We then ask: are these residuals correlated across different N values?

    - If a seed is consistently above average across multiple N values (even as the overall T-rate drops), its question content genuinely resists induction pressure — its transition curve is shifted right.
    - If seed residuals shuffle randomly across N, the within-N variation is just T=1 noise.

    Metrics:
    - **ICC(1,1) on residuals** — fraction of seed-level variance that's stable across N (N values as "raters" of each seed)
    - **Kendall's W** — rank concordance: do the same seeds rank high/low across N values?
    - **Consistently above/below median** — seeds above (or below) model mean at every transition-zone N; under the null this has probability 0.5^k per seed
    """)
    return


@app.cell
def _(alt, icc_oneway, mo, model_selector, np, samples_neutral, stats):
    _model = model_selector.value
    _data = samples_neutral[samples_neutral["model"] == _model].copy()

    # Per-seed T-rate at each N (averaging over epochs)
    _pivot = (
        _data.groupby(["trial_index", "n_turns"])["is_T"]
        .mean()
        .reset_index()
        .pivot(index="trial_index", columns="n_turns", values="is_T")
    )

    # Model mean T-rate at each N
    _n_means = _pivot.mean(axis=0)

    # Residuals: how much each seed deviates from model mean at each N
    _residuals = _pivot.subtract(_n_means, axis=1)

    # Restrict to transition zone (0.2 < T-rate < 0.8)
    _tz_ns = sorted([n for n in _pivot.columns if 0.2 < _n_means[n] < 0.8])

    def _icc_interp(v):
        if v < 0.05:
            return "negligible — seed rankings are essentially random across N"
        if v < 0.15:
            return "weak — slight tendency for seeds to maintain rank, mostly noise"
        if v < 0.3:
            return "moderate — seeds show meaningful stability across N"
        return "strong — seed effects are highly stable across N"

    def _w_interp(w, p):
        _sig = "significant" if p < 0.05 else "not significant"
        if w < 0.1:
            return f"negligible concordance ({_sig})"
        if w < 0.3:
            return f"weak concordance ({_sig})"
        if w < 0.5:
            return f"moderate concordance ({_sig})"
        return f"strong concordance ({_sig})"

    def _corr_interp(r):
        if r < 0.1:
            return "near-zero — seed ranks reshuffle between adjacent Ns"
        if r < 0.3:
            return "weak — some local persistence"
        if r < 0.5:
            return "moderate — seed ranks carry over between nearby Ns"
        return "strong — seed ranks are very stable between adjacent Ns"

    if len(_tz_ns) >= 2:
        _tz_res = _residuals[_tz_ns].dropna()
        _n_seeds = len(_tz_res)
        _k = len(_tz_ns)

        # ICC(1,1) on residual matrix (seeds x N)
        _icc = icc_oneway(_tz_res.values)

        # Kendall's W (coefficient of concordance on ranks)
        _ranks = _tz_res.rank(axis=0)
        _rank_sums = _ranks.sum(axis=1)
        _mean_rs = _rank_sums.mean()
        _ss = ((_rank_sums - _mean_rs) ** 2).sum()
        _w = (12 * _ss) / (_k**2 * (_n_seeds**3 - _n_seeds)) if _n_seeds > 1 else 0.0
        _friedman_chi2 = _k * (_n_seeds - 1) * _w
        _friedman_p = 1 - stats.chi2.cdf(_friedman_chi2, _n_seeds - 1)

        # Seeds consistently above/below model mean at ALL tz N values
        _above_all = int((_tz_res > 0).all(axis=1).sum())
        _below_all = int((_tz_res < 0).all(axis=1).sum())
        _p_consistent = 0.5**_k
        _expected = _n_seeds * _p_consistent

        # Mean adjacent-N Pearson correlation on residuals
        _adj_corrs = []
        for _i in range(len(_tz_ns) - 1):
            _r = _tz_res[_tz_ns[_i]].corr(_tz_res[_tz_ns[_i + 1]])
            if not np.isnan(_r):
                _adj_corrs.append(_r)
        _mean_adj_corr = float(np.mean(_adj_corrs)) if _adj_corrs else 0.0

        # Heatmap: seeds (sorted by mean residual) x N values
        _sorted_seeds = _tz_res.mean(axis=1).sort_values(ascending=False).index.tolist()
        _hm_long = (
            _tz_res.loc[_sorted_seeds]
            .reset_index()
            .melt(
                id_vars="trial_index",
                var_name="n_turns",
                value_name="residual",
            )
        )
        _hm_long["n_turns"] = _hm_long["n_turns"].astype(str)

        _heatmap = (
            alt.Chart(_hm_long)
            .mark_rect()
            .encode(
                x=alt.X(
                    "n_turns:O",
                    title="N",
                    sort=[str(n) for n in _tz_ns],
                ),
                y=alt.Y(
                    "trial_index:O",
                    title="Seed (sorted by mean residual)",
                    sort=_sorted_seeds,
                ),
                color=alt.Color(
                    "residual:Q",
                    scale=alt.Scale(scheme="redblue", domainMid=0),
                    title="T-rate residual",
                ),
                tooltip=[
                    "trial_index",
                    "n_turns:O",
                    alt.Tooltip("residual:Q", format=".3f"),
                ],
            )
            .properties(
                width=max(250, _k * 45),
                height=max(200, _n_seeds * 18),
                title=f"Seed residuals across N — {_model} (transition zone)",
            )
        )

        _stable = _icc > 0.1 and _friedman_p < 0.05
        _verdict = (
            "**Seed effects are stable across N** — the same seeds tend to "
            "over/under-perform at multiple N values. This is strong evidence "
            "that question content shifts the transition curve."
            if _stable
            else "**Seed effects are not stable across N** — rankings shuffle "
            "between N values. The within-N variation is likely T=1 noise."
        )

        _output = mo.vstack(
            [
                mo.md(f"""
    **Results for {_model} ({_k} N values in transition zone)**

    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | ICC(1,1) on residuals | **{_icc:.3f}** | {_icc_interp(_icc)} |
    | Kendall's W | **{_w:.3f}** (p={_friedman_p:.4f}) | {_w_interp(_w, _friedman_p)} |
    | Mean adjacent-N corr. | **{_mean_adj_corr:.3f}** | {_corr_interp(_mean_adj_corr)} |
    | Seeds above mean at all {_k} Ns | **{_above_all}** / {_n_seeds} (null expect: {_expected:.1f}) | {"more than null — some seeds consistently resist induction" if _above_all > _expected * 2 else "within null range"} |
    | Seeds below mean at all {_k} Ns | **{_below_all}** / {_n_seeds} (null expect: {_expected:.1f}) | {"more than null — some seeds consistently yield to induction" if _below_all > _expected * 2 else "within null range"} |

    {_verdict}

    **How to read these values:**
    - **ICC(1,1)**: Intraclass correlation treating N values as "raters" of each seed. <0.05 = negligible, 0.05–0.15 = weak, 0.15–0.3 = moderate, >0.3 = strong. This is the single most informative metric.
    - **Kendall's W**: Ranges 0–1. Concordance of seed rankings across N values. p-value from Friedman test; p<0.05 means rankings are non-random.
    - **Adjacent-N correlation**: Pearson r between seed residuals at consecutive N values. High values mean seed effects are locally persistent (not jumping around).
    - **Consistent seeds**: Seeds above (or below) model mean at *every* transition-zone N. Under null (independent flips), probability = 0.5^{_k} = {_p_consistent:.4f} per seed, so expect {_expected:.1f} out of {_n_seeds} by chance.
            """),
                _heatmap,
                mo.md(
                    "*Red = seed above model mean (more T), blue = below (more P)."
                    " Seeds sorted by mean residual. If question content matters,"
                    " you see persistent red rows at top and blue at bottom."
                    " If it's noise, colors scatter randomly.*"
                ),
            ]
        )
    else:
        _output = mo.callout(
            mo.md(
                f"**{_model}** has fewer than 2 N values in the transition"
                " zone — cannot assess cross-N stability."
            ),
            kind="warn",
        )
    _output  # type: ignore[reportUnusedExpression]
    return


@app.cell
def _(icc_oneway, np, pd, samples_neutral, stats):
    # Cross-N stability metrics for ALL models (used by summary)
    _records = []
    for _model in samples_neutral["model"].unique():
        _data = samples_neutral[samples_neutral["model"] == _model]
        _pivot = (
            _data.groupby(["trial_index", "n_turns"])["is_T"]
            .mean()
            .reset_index()
            .pivot(index="trial_index", columns="n_turns", values="is_T")
        )
        _n_means = _pivot.mean(axis=0)
        _residuals = _pivot.subtract(_n_means, axis=1)
        _tz_ns = sorted([n for n in _pivot.columns if 0.2 < _n_means[n] < 0.8])

        if len(_tz_ns) < 2:
            _records.append(
                {
                    "model": _model,
                    "n_tz": len(_tz_ns),
                    "icc_cross_n": float("nan"),
                    "kendall_w": float("nan"),
                    "kendall_p": float("nan"),
                    "mean_adj_corr": float("nan"),
                    "consistent_above": 0,
                    "consistent_below": 0,
                    "n_seeds": len(_pivot),
                }
            )
            continue

        _tz_res = _residuals[_tz_ns].dropna()
        _n_seeds = len(_tz_res)
        _k = len(_tz_ns)

        _icc = icc_oneway(_tz_res.values)

        _ranks = _tz_res.rank(axis=0)
        _rank_sums = _ranks.sum(axis=1)
        _mean_rs = _rank_sums.mean()
        _ss = ((_rank_sums - _mean_rs) ** 2).sum()
        _w = (12 * _ss) / (_k**2 * (_n_seeds**3 - _n_seeds)) if _n_seeds > 1 else 0.0
        _friedman_chi2 = _k * (_n_seeds - 1) * _w
        _friedman_p = 1 - stats.chi2.cdf(_friedman_chi2, _n_seeds - 1)

        _adj_corrs = []
        for _i in range(len(_tz_ns) - 1):
            _r = _tz_res[_tz_ns[_i]].corr(_tz_res[_tz_ns[_i + 1]])
            if not np.isnan(_r):
                _adj_corrs.append(_r)

        _records.append(
            {
                "model": _model,
                "n_tz": _k,
                "icc_cross_n": round(_icc, 4),
                "kendall_w": round(_w, 4),
                "kendall_p": round(_friedman_p, 6),
                "mean_adj_corr": round(float(np.mean(_adj_corrs)), 4)
                if _adj_corrs
                else float("nan"),
                "consistent_above": int((_tz_res > 0).all(axis=1).sum()),
                "consistent_below": int((_tz_res < 0).all(axis=1).sum()),
                "n_seeds": _n_seeds,
            }
        )

    cross_n_df = pd.DataFrame(_records)
    return (cross_n_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary: Is question content a meaningful factor?
    """)
    return


@app.cell
def _(alt, cross_n_df, mo, pd, samples_neutral, within_n_df):
    # --- Within-N metrics (from Check 3) ---
    _tz = within_n_df[
        (within_n_df["T_rate"] > 0.2) & (within_n_df["T_rate"] < 0.8)
    ].copy()
    _n_tz_cells = len(_tz)
    _n_tz_models = _tz["model"].nunique()
    _sig_rate_tz = _tz["significant"].mean() if _n_tz_cells > 0 else 0
    _odr_mean_tz = _tz["overdispersion"].mean() if _n_tz_cells > 0 else 1.0
    _odr_median_tz = _tz["overdispersion"].median() if _n_tz_cells > 0 else 1.0
    _has_unanimous_tz = (
        ((_tz["unanimous_T"] > 0) | (_tz["unanimous_P"] > 0))
        if _n_tz_cells > 0
        else pd.Series(dtype=bool)
    )
    _unan_rate_tz = _has_unanimous_tz.mean() if len(_has_unanimous_tz) > 0 else 0

    # --- Cross-N metrics (from Check 4, decisive) ---
    _valid_cross = cross_n_df[cross_n_df["n_tz"] >= 2]
    _n_cross_models = len(_valid_cross)
    _mean_icc = _valid_cross["icc_cross_n"].mean() if _n_cross_models > 0 else 0
    _n_sig_w = (
        int((_valid_cross["kendall_p"] < 0.05).sum()) if _n_cross_models > 0 else 0
    )
    _total_consistent = (
        int(
            _valid_cross["consistent_above"].sum()
            + _valid_cross["consistent_below"].sum()
        )
        if _n_cross_models > 0
        else 0
    )

    # --- Per-model summary table ---
    _model_summary = []
    for _model in sorted(within_n_df["model"].unique()):
        _m_wn = _tz[_tz["model"] == _model] if _n_tz_cells > 0 else pd.DataFrame()
        _m_cn = cross_n_df[cross_n_df["model"] == _model]
        _m_data = samples_neutral[samples_neutral["model"] == _model]
        _n_epochs = int(_m_data.groupby(["n_turns", "trial_index"]).size().median())
        _n_tz_n = len(_m_wn)
        _cn_row = _m_cn.iloc[0] if len(_m_cn) > 0 else {}
        _icc_val = _cn_row.get("icc_cross_n", float("nan"))
        _w_val = _cn_row.get("kendall_w", float("nan"))
        _model_summary.append(
            {
                "Model": _model,
                "k": _n_epochs,
                "N in tz": _n_tz_n,
                "ODR mean": round(float(_m_wn["overdispersion"].mean()), 2)
                if _n_tz_n > 0
                else "\u2014",
                "% sig \u03c7\u00b2": f"{_m_wn['significant'].mean():.0%}"
                if _n_tz_n > 0
                else "\u2014",
                "ICC cross-N": round(float(_icc_val), 3)
                if not pd.isna(_icc_val)
                else "\u2014",
                "W cross-N": round(float(_w_val), 3)
                if not pd.isna(_w_val)
                else "\u2014",
                "Stable seeds": int(
                    _cn_row.get("consistent_above", 0)
                    + _cn_row.get("consistent_below", 0)
                ),
            }
        )
    _summary_table = pd.DataFrame(_model_summary)

    # --- Verdict: cross-N stability is the primary driver ---
    _cross_n_strong = _mean_icc > 0.1 and _n_sig_w > _n_cross_models / 2
    _cross_n_moderate = _mean_icc > 0.05 or _n_sig_w > 0
    _within_n_signal = _sig_rate_tz > 0.5 and _odr_mean_tz > 1.5

    if _cross_n_strong and _within_n_signal:
        _verdict = (
            "**Question content is a meaningful and stable factor.** Seeds"
            " show significant heterogeneity within individual N values,"
            " AND these effects persist across the N sweep (stable cross-N"
            " ICC). Some seeds genuinely shift the transition curve."
            " Replications and investigation of question features are"
            " well-justified."
        )
        _verdict_short = "Positive \u2014 stable seed effects"
    elif _cross_n_moderate or _within_n_signal:
        _verdict = (
            "**Mixed evidence.** There is some signal"
            + (" of within-N seed heterogeneity" if _within_n_signal else "")
            + (" and/or modest cross-N stability" if _cross_n_moderate else "")
            + ", but the effects are not consistently strong across models."
            " The variation may be partly real and partly T=1 noise."
            " Additional replications or T=0 runs would help resolve"
            " the ambiguity."
        )
        _verdict_short = "Mixed \u2014 partial evidence"
    else:
        _verdict = (
            "**Question content is not a meaningful factor.** Seeds do not"
            " show stable effects across the N sweep \u2014 any within-N"
            " heterogeneity is likely T=1 sampling noise that doesn't"
            " replicate at other N values. Better investments: more diverse"
            " question sets, or T=0 runs."
        )
        _verdict_short = "Negative \u2014 noise dominates"

    # Cross-N ICC bar chart
    if _n_cross_models > 0:
        _icc_chart = (
            alt.Chart(_valid_cross)
            .mark_bar()
            .encode(
                x=alt.X(
                    "icc_cross_n:Q",
                    title="ICC(1,1) on seed residuals across N",
                ),
                y=alt.Y("model:N", title=None, sort="-x"),
                color=alt.condition(
                    alt.datum.kendall_p < 0.05,
                    alt.value("#4e79a7"),
                    alt.value("#ccc"),
                ),
                tooltip=[
                    "model",
                    "icc_cross_n:Q",
                    "kendall_w:Q",
                    "kendall_p:Q",
                    "n_tz:Q",
                ],
            )
            .properties(
                width=500,
                height=max(120, _n_cross_models * 30),
                title=(
                    "Cross-N seed stability per model (blue = significant Kendall's W)"
                ),
            )
        )
        _viz = _icc_chart
    else:
        _viz = mo.md("*No models with \u22652 transition-zone N values.*")

    mo.vstack(
        [
            mo.md(f"""
    ### Verdict: {_verdict_short}

    {_verdict}

    ### Cross-N evidence (decisive)

    | Metric | Value |
    |--------|-------|
    | Models with \u22652 transition-zone N values | **{_n_cross_models}** |
    | Mean ICC(1,1) on seed residuals across N | **{_mean_icc:.3f}** |
    | Models with significant Kendall's W (p < 0.05) | **{_n_sig_w}** / {_n_cross_models} |
    | Seeds consistently above/below median across all tz N values | **{_total_consistent}** |

    ### Within-N evidence (supporting)

    | Metric | Value |
    |--------|-------|
    | (model, N) cells in transition zone | **{_n_tz_cells}** across **{_n_tz_models}** models |
    | Cells with significant seed heterogeneity (\u03c7\u00b2, p < 0.05) | **{_sig_rate_tz:.0%}** |
    | Mean overdispersion ratio | **{_odr_mean_tz:.2f}** (1.0 = null) |
    | Median overdispersion ratio | **{_odr_median_tz:.2f}** |
    | Cells with \u22651 unanimous seed | **{_unan_rate_tz:.0%}** |
        """),
            _viz,
            mo.md("### Per-model summary"),
            _summary_table,
            mo.md("""
    ### How to read these results

    **Within-N metrics** (overdispersion, \u03c7\u00b2, unanimous seeds) tell you seeds differ at individual N values \u2014 but this is *necessary, not sufficient*. T=1 noise can create spurious within-N patterns that vanish at other N values.

    **Cross-N metrics** are the decisive test:
    - **ICC on residuals** = fraction of seed-level variance that's stable across N. Seeds \u00d7 N matrix of residuals (centered by model mean at each N), with N values as "raters". Values > 0.1 are meaningful.
    - **Kendall's W** = rank concordance. Significant W means seeds maintain their relative ordering across N.
    - **Consistent seeds** = above/below model mean at *every* transition-zone N. Under the null, probability is 0.5^k per seed \u2014 so even a few such seeds with many N values is strong evidence.

    **The key distinction**: A seed that's always-T at N=10 but average at N=15 is noise. A seed that's consistently above average at N=5, 10, 15, and 20 has question content that genuinely shifts the transition curve.
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
