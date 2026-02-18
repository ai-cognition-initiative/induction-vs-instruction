"""Shared plotting helpers for per-protocol interactive reports.

Uses inspect-viz primitives to build reusable plot components.
"""

from __future__ import annotations

from inspect_viz import Data, Selection
from inspect_viz.mark import bar_x, cell, line, text
from inspect_viz.plot import legend as make_legend, plot


def labeled_line_chart(
    data: Data,
    label_data: Data,
    *,
    selection: Selection | None = None,
    score_col: str = "score",
    title: str = "Score vs N",
    width: int = 900,
    height: int = 400,
    x_domain: list[str] | None = None,
):
    """Labeled multi-line chart: one line per model, text label at rightmost point.

    Args:
        data: Main data source (all rows). Used for line marks.
        label_data: Data filtered to max n_turns only. Used for text labels.
        selection: Shared Selection that filters by condition/instruction.
        score_col: Column name for the y-axis score.
        title: Plot title.
        width: Plot width in pixels.
        height: Plot height in pixels.
        x_domain: Explicit x-axis domain (sorted n_turns values).
    """
    return plot(
        line(
            data,
            x="n_turns",
            y=score_col,
            stroke="model",
            filter_by=selection,
            marker=True,
            tip=True,
            channels={
                "N": "n_turns",
                "Score": score_col,
                "Model": "model",
            },
        ),
        text(
            label_data,
            x="n_turns",
            y=score_col,
            text="model",
            filter_by=selection,
            fill="model",
            stroke="white",
            stroke_width=4,
            dx=5,
            line_anchor="middle",
            styles={"text_anchor": "start", "font_size": 11, "font_weight": 600},
        ),
        x_label="N (hardcoded turns)",
        y_label="Instruction Following Rate",
        title=title,
        width=width,
        height=height,
        y_domain=[0, 1.05],
        x_domain=x_domain,
        margin_right=200,
    )


def bullet_graph(
    data: Data,
    baseline_data: Data,
    *,
    selection: Selection | None = None,
    score_col: str = "score",
    title: str = "Bullet Graph",
    width: int = 900,
    height: int | None = None,
):
    """Bullet graph: foreground bars (condition) over background bars (baseline).

    Background bars are wider and grey. Foreground bars are narrower and colored.
    Faceted by instruction (fx) and condition_pair (fy). Y-axis is model.

    Args:
        data: Main data (non-neutral conditions).
        baseline_data: Baseline data (neutral condition scores).
        selection: Shared Selection for filtering (e.g., by n_turns).
        score_col: Column for bar length.
    """
    return plot(
        # Background: baseline (neutral) — wide, grey
        bar_x(
            baseline_data,
            x=score_col,
            y="model",
            fx="instruction",
            fy="condition",
            fill="#e0e0e0",
            filter_by=selection,
            inset=2,
        ),
        # Foreground: condition score — narrower, colored
        bar_x(
            data,
            x=score_col,
            y="model",
            fx="instruction",
            fy="condition",
            fill="condition",
            filter_by=selection,
            inset=6,
            tip=True,
            channels={
                "Condition": "condition",
                "Score": score_col,
                "Model": "model",
            },
        ),
        legend=make_legend(
            "color",
            frame_anchor="top",
            columns=3,
            margin_top=4,
            margin_bottom=4,
            margin_left=4,
            margin_right=4,
        ),
        x_label="Instruction Following Rate",
        x_domain=[0, 1.05],
        y_label=None,
        fx_label=None,
        fy_label=None,
        fy_axis=False,
        width=width,
        height=height,
        margin_top=70,
        margin_left=110,
    )


def paired_bullet_graph(
    aligned_data: Data,
    misaligned_data: Data,
    *,
    selection: Selection | None = None,
    score_col: str = "score",
    title: str = "Paired Bullet Graph",
    width: int = 1100,
    height: int | None = None,
):
    """Paired bullet graph: aligned vs misaligned conditions within each pair.

    Background bars = misaligned condition. Foreground bars = aligned condition.
    Faceted by instruction (fx) and condition_pair (fy). Y-axis is model.
    """
    return plot(
        # Background: misaligned condition — wide, light grey
        bar_x(
            misaligned_data,
            x=score_col,
            y="model",
            fx="instruction",
            fy="condition_pair",
            fill="#d0d0d0",
            filter_by=selection,
            inset=2,
        ),
        # Foreground: aligned condition — narrower, colored
        bar_x(
            aligned_data,
            x=score_col,
            y="model",
            fx="instruction",
            fy="condition_pair",
            fill="condition",
            filter_by=selection,
            inset=6,
            tip=True,
            channels={
                "Condition": "condition",
                "Score": score_col,
                "Model": "model",
            },
        ),
        legend=make_legend(
            "color",
            frame_anchor="top",
            columns=3,
            margin_top=4,
            margin_bottom=4,
            margin_left=4,
            margin_right=4,
        ),
        x_label="Instruction Following Rate",
        x_domain=[0, 1.05],
        y_label=None,
        fx_label=None,
        fy_label=None,
        fy_axis=False,
        width=width,
        height=height,
        margin_top=70,
        margin_left=110,
    )


def calibration_line_chart(
    data: Data,
    label_data: Data,
    *,
    selection: Selection | None = None,
    actual_col: str = "score_instruction_following",
    predicted_col: str = "score_prediction_instruction",
    title: str = "Instruction Following: Actual (solid) vs Predicted (dashed)",
    width: int = 900,
    height: int = 400,
    x_domain: list[str] | None = None,
):
    """Calibration chart: actual IF rate (solid) vs predicted IF rate (dashed).

    Both lines share the same color per model so the gap is immediately visible.
    Text labels at the rightmost N mark the actual (solid) lines.

    Args:
        data: Main data source (wide format with both score columns).
        label_data: Data filtered to max n_turns for text labels.
        selection: Shared Selection for condition/instruction filtering.
        actual_col: Column for actual instruction-following rate (solid line).
        predicted_col: Column for predicted instruction-following rate (dashed line).
    """
    return plot(
        line(
            data,
            x="n_turns",
            y=actual_col,
            stroke="model",
            filter_by=selection,
            marker=True,
            tip=True,
            channels={"N": "n_turns", "Actual IF": actual_col, "Model": "model"},
        ),
        line(
            data,
            x="n_turns",
            y=predicted_col,
            stroke="model",
            stroke_dasharray="4 2",
            filter_by=selection,
            marker=True,
            tip=True,
            channels={"N": "n_turns", "Predicted IF": predicted_col, "Model": "model"},
        ),
        text(
            label_data,
            x="n_turns",
            y=actual_col,
            text="model",
            filter_by=selection,
            fill="model",
            stroke="white",
            stroke_width=4,
            dx=5,
            line_anchor="middle",
            styles={"text_anchor": "start", "font_size": 11, "font_weight": 600},
        ),
        x_label="N (hardcoded turns)",
        y_label="Rate",
        title=title,
        width=width,
        height=height,
        y_domain=[0, 1.05],
        x_domain=x_domain,
        margin_right=200,
    )


def overview_heatmap(
    data: Data,
    *,
    x: str = "n_turns",
    y: str = "condition",
    fill_col: str = "score",
    title: str = "Overview",
    width: int = 900,
    height: int = 400,
):
    """Cell + text heatmap for condition x N overview."""
    return plot(
        cell(
            data,
            x=x,
            y=y,
            fill=fill_col,
            tip=True,
            inset=1,
        ),
        text(
            data,
            x=x,
            y=y,
            text=fill_col,
            fill="white",
            font_weight=600,
        ),
        padding=0,
        color_scheme="rdylgn",
        title=title,
        width=width,
        height=height,
        x_label="N (hardcoded turns)",
        y_label=None,
        margin_left=160,
    )
