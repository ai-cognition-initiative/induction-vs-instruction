import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    # add imports here
    return (mo,)


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
def _():
    # todo
    # for now, filter all the data to only include instruction_template='instruction_no_hint'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Behavioral analysis
    - Plot A1 - stacked bar plot (Altair). For each model on the x axis, show three stacks: number of N values with IF >= THRESHOLD, number of N values with PF >= THRESHOLD, remaining N values in the dataset. Average across conditions.THRESHOLD should be set interactively in the notebook, with a default of 0.8.
    - Plot A2 - avg IF rate vs capability. Compute average IF rate and make a scatterplot of (IF rate, intelligence_index). Color is model.
    - Plot A3 - horizontal bar chart with avg IF rate + error bars to show uncertainty. Error bars come from the stderr metric in the dataframe. Faceted by condition (col size 3 or 4), y shows models. x axis should be fixed from 0 to 1. Add dropdown to select which instruction_template to show
    - Plot A4: vertical colored stripes of IF rate for increasing N: similar to https://observablehq.com/@observablehq/plot-stacked-unit-chart. But here the data is sorted by N, not by the value of the data itself. The quantity to plot is again the mean IF rate across condition per model (models on the y axis). Each stacked unit is the IF rate for one N value per increasing N, with color showing the average IF rate.
    """)
    return


@app.cell
def _():
    # plot a1
    return


@app.cell
def _():
    # plot A2
    return


@app.cell
def _():
    # plot A3
    return


@app.cell
def _():
    # Plot A4
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
def _():
    # plot B1
    return


@app.cell
def _():
    # plot B2
    return


@app.cell
def _():
    # plot B3
    return


@app.cell
def _():
    # plot B4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction analysis
    - Plot C1: avg "predicts instruction following" rate by model and capability, like plots A2 and B2
    """)
    return


@app.cell
def _():
    # Plot C1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of hint
    Spec coming soon
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of prediction
    Spec coming soon
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of aligned/misaligned
    Spec coming soon
    """)
    return


if __name__ == "__main__":
    app.run()
