import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from inspect_ai.log import list_eval_logs
    from inspect_ai.analysis import evals_df, samples_df, EvalInfo, EvalModel, EvalTask, EvalResults, EvalScores

    return EvalModel, EvalScores, EvalTask, Path, evals_df, mo, samples_df


@app.cell
def _(Path, mo):
    logs_base = Path("logs")
    log_subfolders = sorted([d.name for d in logs_base.iterdir() if d.is_dir()])

    subfolder_selector = mo.ui.dropdown(
        options=log_subfolders,
        label="Select log subfolder",
    )
    subfolder_selector
    return logs_base, subfolder_selector


@app.cell
def _(logs_base, mo, subfolder_selector):
    mo.stop(
        not subfolder_selector.value,
        mo.md("Select a log subfolder to view evals."),
    )

    log_dir = str(logs_base / subfolder_selector.value)
    log_dir
    return (log_dir,)


@app.cell
def _(EvalModel, EvalScores, EvalTask, evals_df, log_dir, mo):
    with mo.status.spinner("Loading evals..."):
        evals = evals_df(log_dir,  columns= EvalTask + EvalModel + EvalScores)
    evals
    return (evals,)


@app.cell
def _(evals):
    evals.columns
    return


@app.cell
def _(evals, mo, samples_df):
    mo.stop(evals.empty, mo.md("No evals found in this directory."))

    with mo.status.spinner("Loading samples..."):
        samples = samples_df(evals["log"].tolist())
    samples
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
