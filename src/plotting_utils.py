"""Shared plotting utilities for analysis notebooks."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd


BENCHMARKS = {
    "intelligence_index": "Intelligence Index",
    "mmlu_pro": "MMLU Pro",
    "gpqa": "GPQA",
    "ifbench": "IFBench",
}

SHAPE_SCALE = alt.Scale(
    domain=["non-reasoning", "reasoning"], range=["circle", "diamond"]
)


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
    reasoning_models: set[str],
    nudge_fn=nudge_labels,
) -> pd.DataFrame:
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
            _wide[["model", y_col, "reasoning", _col]].dropna(subset=[_col]).copy()
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
            _part = nudge_fn(_part, "benchmark_value", y_col, _xr, _yr)
        else:
            _yr = [_part[y_col].min() - 5, _part[y_col].max() + 5]
            _part = nudge_fn(_part, "benchmark_value", y_col, _xr, _yr)
        parts.append(_part)
    result = pd.concat(parts, ignore_index=True)
    return result.where(pd.notnull(result), None)


def make_scatter_chart(
    df: pd.DataFrame, y_col: str, y_title: str, title: str
) -> alt.FacetChart:
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
            shape=alt.Shape("reasoning:N", scale=SHAPE_SCALE, title="Model type"),
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

    return Plot.plot({
        "projection": {
            "type": "azimuthal-equidistant",
            "rotate": [0, -90],
            "domain": js("d3.geoCircle().center([0, 90]).radius(0.625)()"),
        },
        "marks": [
            Plot.geo([0.5, 0.4, 0.3, 0.2, 0.1], {
                "geometry": js("(r) => d3.geoCircle().center([0, 90]).radius(r)()"),
                "stroke": "currentColor",
                "strokeOpacity": 0.2,
            }),
            Plot.link(js(f"({_lon_js}).domain()"), {
                "x1": js(f"(d) => ({_lon_js})(d)"),
                "y1": 90,
                "x2": 0,
                "y2": 90,
                "stroke": "currentColor",
                "strokeOpacity": 0.2,
            }),
            Plot.text(js(f"({_lon_js}).domain()"), {
                "x": js(f"(d) => ({_lon_js})(d)"),
                "y": 89.4,
                "text": js("Plot.identity"),
                "lineWidth": 5,
            }),
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
            Plot.area(points, {
                "x1": js(f"(d) => ({_lon_js})(d.key)"),
                "y1": js("(d) => 90 - d.value * 0.5"),
                "fill": "name",
                "stroke": "name",
                "curve": "cardinal-closed",
                "fillOpacity": 0.2,
                "strokeWidth": 1.5,
            }),
            Plot.dot(points, {
                "x": js(f"(d) => ({_lon_js})(d.key)"),
                "y": js("(d) => 90 - d.value * 0.5"),
                "fill": "name",
                "stroke": "white",
                "r": 3,
            }),
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
        "marginLeft": 100,
        "marginRight": 100,
        "height": height,
        "title": title,
        "color": {"legend": True},
    })


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
