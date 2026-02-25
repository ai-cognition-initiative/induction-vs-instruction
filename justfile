# Reporting commands
# Generate visualization notebooks for eval logs

# Convert Windows backslashes to forward slashes for shell compatibility
root := replace(justfile_directory(), "\\", "/")


logs:
    @echo "Available log folders:"
    @ls -1 logs/ 2>/dev/null || echo "  (none)"

[group("reporting")]
report folder:
    #!/usr/bin/env bash
    set -e
    echo "Generating reports for logs/{{folder}}"
    rm -rf outputs/viz/{{folder}} outputs/notebooks/{{folder}}
    mkdir -p outputs/viz/{{folder}} outputs/notebooks/{{folder}}

    behavioral_ok=false
    prediction_ok=false

    if uv run python scripts/prepare_viz_data.py \
        --log-dir logs/{{folder}} \
        --output-dir outputs/viz/{{folder}} \
        --protocol behavioral; then
        behavioral_ok=true
    else
        echo "Skipping behavioral report (no behavioral logs in logs/{{folder}})"
    fi

    if uv run python scripts/prepare_viz_data.py \
        --log-dir logs/{{folder}} \
        --output-dir outputs/viz/{{folder}} \
        --protocol prediction; then
        prediction_ok=true
    else
        echo "Skipping prediction report (no prediction logs in logs/{{folder}})"
    fi

    if [ "$behavioral_ok" = "true" ]; then
        QUARTO_PYTHON="{{root}}/.venv/Scripts/python.exe" quarto render notebooks/behavioral_analysis.qmd \
            --output-dir {{root}}/outputs/notebooks/{{folder}} \
            --execute \
            -P evals_path:outputs/viz/{{folder}}/evals.parquet
    fi

    if [ "$prediction_ok" = "true" ]; then
        QUARTO_PYTHON="{{root}}/.venv/Scripts/python.exe" quarto render notebooks/prediction_analysis.qmd \
            --output-dir {{root}}/outputs/notebooks/{{folder}} \
            --execute \
            -P evals_path:outputs/viz/{{folder}}/evals_prediction.parquet
    fi

    echo "Done. Reports at outputs/notebooks/{{folder}}/"

[group("reporting")]
report-prediction behavioral_folder prediction_folder:
    #!/usr/bin/env bash
    set -e
    echo "Generating combined report: logs/{{behavioral_folder}} (behavioral) + logs/{{prediction_folder}} (prediction)"
    rm -rf outputs/viz/{{behavioral_folder}}_vs_{{prediction_folder}} outputs/notebooks/{{behavioral_folder}}_vs_{{prediction_folder}}
    mkdir -p outputs/viz/{{behavioral_folder}}_vs_{{prediction_folder}} outputs/notebooks/{{behavioral_folder}}_vs_{{prediction_folder}}

    uv run python scripts/prepare_viz_data.py \
        --log-dir logs/{{behavioral_folder}} \
        --log-dir-2 logs/{{prediction_folder}} \
        --output-dir outputs/viz/{{behavioral_folder}}_vs_{{prediction_folder}} \
        --protocol combined

    QUARTO_PYTHON="{{root}}/.venv/Scripts/python.exe" quarto render notebooks/behavioral_vs_prediction_analysis.qmd \
        --output-dir {{root}}/outputs/notebooks/{{behavioral_folder}}_vs_{{prediction_folder}} \
        --execute \
        -P evals_path:outputs/viz/{{behavioral_folder}}_vs_{{prediction_folder}}/evals_combined.parquet

    echo "Done. Report at outputs/notebooks/{{behavioral_folder}}_vs_{{prediction_folder}}/"

[group("reporting")]
report-config config_file:
    #!/usr/bin/env bash
    set -e
    uv run python scripts/generate_report_from_config.py --config {{config_file}}

[group("reporting")]
rescore-prediction folder:
    #!/usr/bin/env bash
    set -e
    echo "Rescoring prediction logs in logs/{{folder}}"
    for f in logs/{{folder}}/*.eval; do
        echo "  Scoring $f"
        uv run inspect score "$f" \
            --scorer src/scorers/prediction.py@prediction_scorer \
            --action overwrite \
            --overwrite
    done
    echo "Done."

[group("reporting")]
preview folder protocol="behavioral_analysis" evals_file="evals.parquet":
    @QUARTO_PYTHON="{{root}}/.venv/Scripts/python.exe" quarto preview notebooks/{{protocol}}.qmd \
        --execute \
        -P evals_path:outputs/viz/{{folder}}/{{evals_file}}

[group("reporting")]
clean-reports:
    rm -rf outputs/notebooks/*
    rm -rf outputs/viz/*

[group("reporting")]
token-usage:
    @uv run python src/utils/token_usage.py
    @cat logs/token_usage.md
