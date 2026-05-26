"""Shared plotting utilities for analysis notebooks."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd


COLOR_SCHEME = "set2"

DISPLAY_NAMES: dict[str, str] = {
    "claude-4.6-sonnet": "Claude 4.6 Sonnet",
    "claude-opus-4.6": "Claude 4.6 Opus",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemma-3-12b-it": "Gemma-3 12B",
    "gemma-3-27b-it": "Gemma-3 27B",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2-medium": "GPT-5.2 (medium)",
    "hermes-4-70b": "Hermes-4 70B",
    "hermes-4-70b-reasoning": "Hermes-4 70B (reasoning)",
    "kimi-k2-instruct": "Kimi K2",
    "llama-3.1-70b-instruct": "Llama 3.1 70B",
    "llama-3.3-70b-instruct": "Llama 3.3 70B",
    "olmo-3.1-32b-instruct": "OLMo 3.1 32B",
    "olmo-3.1-32b-instruct-dpo": "OLMo 3.1 32B (SFT+DPO)",
    "olmo-3.1-32b-instruct-sft": "OLMo 3.1 32B (SFT)",
    "qwen3-235b-a22b-instruct-2507": "Qwen3 235B A22B",
    "qwen3-30b-a3b-instruct-2507": "Qwen3 30B A3B",
}

_FONT = "Helvetica Neue, Arial, sans-serif"


def _paper_theme() -> dict:
    """Altair theme suitable for a two-column academic paper."""
    return {
        "config": {
            "background": "white",
            "font": _FONT,
            "title": {
                "font": _FONT,
                "fontSize": 20,
                "fontWeight": "normal",
                "anchor": "start",
                "color": "#222",
            },
            "axis": {
                "labelFont": _FONT,
                "labelFontSize": 17,
                "titleFont": _FONT,
                "titleFontSize": 18,
                "titleFontWeight": "normal",
                "gridColor": "#e0e0e0",
                "domainColor": "#999",
                "tickColor": "#999",
                "labelColor": "#444",
                "titleColor": "#222",
            },
            "legend": {
                "labelFont": _FONT,
                "labelFontSize": 17,
                "titleFont": _FONT,
                "titleFontSize": 18,
                "titleFontWeight": "normal",
            },
            "header": {
                "labelFont": _FONT,
                "labelFontSize": 17,
                "titleFont": _FONT,
                "titleFontSize": 18,
            },
            "mark": {"tooltip": True},
            "view": {
                "width": 300,
                "height": 280,
                "stroke": "transparent",
            },
            "range": {
                "category": {"scheme": COLOR_SCHEME},
                "ordinal": {"scheme": COLOR_SCHEME},
            },
        }
    }


alt.themes.register("paper", _paper_theme)
alt.themes.enable("paper")


BENCHMARKS = {
    "gpqa": "GPQA",
    "ifbench": "IFBench",
}


def nudge_labels(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_range: list[float],
    y_range: list[float],
    pad_x: float = 0.02,
    pad_y: float = 0.03,
    iters: int = 50,
) -> pd.DataFrame:
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


def prep_benchmark_data(
    df: pd.DataFrame,
    y_col: str,
    caps_df: pd.DataFrame,
    nudge_fn=nudge_labels,
) -> pd.DataFrame:
    """Prepare data for benchmark scatter plots."""
    _wide = df.merge(caps_df, on="model", how="inner")
    parts = []
    for _col, _label in BENCHMARKS.items():
        _part = _wide[["model", y_col, _col]].dropna(subset=[_col]).copy()
        if len(_part) == 0:
            continue
        _part = _part.rename(columns={_col: "benchmark_value"})
        _part["benchmark"] = _label
        _xr = [
            _part["benchmark_value"].min() * 0.95,
            _part["benchmark_value"].max() * 1.05,
        ]
        _yr = [0, 1] if y_col in ["avg_if_rate", "prediction_rate", "prediction_accuracy"] else None
        if _yr:
            _part = nudge_fn(_part, "benchmark_value", y_col, _xr, _yr)
        else:
            _yr = [_part[y_col].min() - 5, _part[y_col].max() + 5]
            _part = nudge_fn(_part, "benchmark_value", y_col, _xr, _yr)
        if "label_x" not in _part.columns:
            _part = _part.copy()
            _part["label_x"] = _part["benchmark_value"]
            _part["label_y"] = _part[y_col]
        parts.append(_part)
    result = pd.concat(parts, ignore_index=True)
    return result.where(pd.notnull(result), None)


def make_scatter_chart(
    df: pd.DataFrame,
    y_col: str,
    y_title: str,
    title: str,
    log_x: bool = False,
    log_y: bool = False,
    y_domain: list[float] | None = None,
) -> alt.FacetChart:
    """Create a faceted scatter chart with labels and trendline."""
    x_scale = alt.Scale(type="log") if log_x else alt.Scale(zero=False)
    if log_y:
        y_scale = alt.Scale(type="log")
    elif y_domain is not None:
        y_scale = alt.Scale(domain=y_domain)
    else:
        y_scale = alt.Scale(zero=False)
    points = (
        alt.Chart(df)
        .mark_point(size=100, filled=True)
        .encode(
            x=alt.X(
                "benchmark_value:Q",
                title="Benchmark Score",
                scale=x_scale,
            ),
            y=alt.Y(f"{y_col}:Q", title=y_title, scale=y_scale),
            color=alt.Color("model:N", scale=alt.Scale(scheme=COLOR_SCHEME), legend=None),
            tooltip=[
                "model",
                f"{y_col}:Q",
                "benchmark_value:Q",
                "benchmark:N",
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
        .mark_text(align="left", dx=3, fontSize=10)
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
        .configure_title(fontSize=14)
        .configure_axis(labelFontSize=11, titleFontSize=12)
        .configure_header(labelFontSize=11, titleFontSize=12)
        .configure_legend(labelFontSize=11, titleFontSize=12)
    )


def make_radar_chart(Plot, js, points, categories, title, width=600, height=500):
    """Create a radar chart using pyobsplot (Observable Plot).

    Args:
        Plot: pyobsplot Plot module
        js: pyobsplot js function
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

    return Plot.plot(
        {
            "projection": {
                "type": "azimuthal-equidistant",
                "rotate": [0, -90],
                "domain": js("d3.geoCircle().center([0, 90]).radius(0.625)()"),
            },
            "marks": [
                Plot.geo(
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    {
                        "geometry": js(
                            "(r) => d3.geoCircle().center([0, 90]).radius(r)()"
                        ),
                        "stroke": "currentColor",
                        "strokeOpacity": 0.2,
                    },
                ),
                Plot.link(
                    js(f"({_lon_js}).domain()"),
                    {
                        "x1": js(f"(d) => ({_lon_js})(d)"),
                        "y1": 90,
                        "x2": 0,
                        "y2": 90,
                        "stroke": "currentColor",
                        "strokeOpacity": 0.2,
                    },
                ),
                Plot.text(
                    js(f"({_lon_js}).domain()"),
                    {
                        "x": js(f"(d) => ({_lon_js})(d)"),
                        "y": 89.4,
                        "text": js("Plot.identity"),
                        "lineWidth": 5,
                    },
                ),
                Plot.text(
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    {
                        "x": 180,
                        "y": js("(r) => 90 - r"),
                        "dx": 2,
                        "textAnchor": "start",
                        "text": js("(r) => `${r * 200}%`"),
                        "fill": "currentColor",
                        "stroke": "white",
                        "fontSize": 8,
                    },
                ),
                Plot.area(
                    points,
                    {
                        "x1": js(f"(d) => ({_lon_js})(d.key)"),
                        "y1": js("(d) => 90 - d.value * 0.5"),
                        "fill": "name",
                        "stroke": "name",
                        "curve": "cardinal-closed",
                        "fillOpacity": 0.2,
                        "strokeWidth": 1.5,
                    },
                ),
                Plot.dot(
                    points,
                    {
                        "x": js(f"(d) => ({_lon_js})(d.key)"),
                        "y": js("(d) => 90 - d.value * 0.5"),
                        "fill": "name",
                        "stroke": "white",
                        "r": 3,
                    },
                ),
                Plot.text(
                    points,
                    Plot.pointer(
                        {
                            "x": js(f"(d) => ({_lon_js})(d.key)"),
                            "y": js("(d) => 90 - d.value * 0.5"),
                            "text": js(
                                "(d) => `${d.name}: ${(d.value * 100).toFixed(1)}%`"
                            ),
                            "fill": "currentColor",
                            "stroke": "white",
                            "fontSize": 10,
                            "dy": -10,
                        }
                    ),
                ),
            ],
            "width": width,
            "marginLeft": 100,
            "marginRight": 100,
            "height": height,
            "title": title,
            "color": {"legend": True, "scheme": COLOR_SCHEME.capitalize()},
        }
    )


# Likert offset JS for diverging stacked bars
LIKERT_OFFSET_JS = """(faceI, X1, X2, Z) => {
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
}"""

CATEGORY_ORDER = ["PF-dominant", "Mixed", "IF-dominant"]
CATEGORY_COLORS = ["#d62728", "#999999", "#2ca02c"]
