# Reporting commands
# Generate visualization notebooks for eval logs

# Convert Windows backslashes to forward slashes for shell compatibility
root := replace(justfile_directory(), "\\", "/")


logs:
    @echo "Available log folders:"
    @ls -1 logs/ 2>/dev/null || echo "  (none)"

[group("reporting")]
report folder:
    @echo "Generating reports for logs/{{folder}}"
    @rm -rf outputs/viz/{{folder}} outputs/notebooks/{{folder}}
    @mkdir -p outputs/viz/{{folder}} outputs/notebooks/{{folder}}
    @uv run python scripts/prepare_viz_data.py \
        --log-dir logs/{{folder}} \
        --output-dir outputs/viz/{{folder}}
    @QUARTO_PYTHON="{{root}}/.venv/Scripts/python.exe" quarto render notebooks/behavioral_analysis.qmd \
        --output-dir {{root}}/outputs/notebooks/{{folder}} \
        --execute \
        -P evals_path:outputs/viz/{{folder}}/evals.parquet
    # @QUARTO_PYTHON="{{root}}/.venv/Scripts/python.exe" quarto render notebooks/prediction_analysis.qmd \
    #     --output-dir {{root}}/outputs/notebooks/{{folder}} \
    #     --execute \
    #     -P evals_path:outputs/viz/{{folder}}/evals.parquet
    @echo "Reports generated at outputs/notebooks/{{folder}}/"

[group("reporting")]
preview folder protocol="behavioral_analysis":
    @QUARTO_PYTHON="{{root}}/.venv/Scripts/python.exe" quarto preview notebooks/{{protocol}}.qmd \
        --execute \
        -P evals_path:outputs/viz/{{folder}}/evals.parquet

[group("reporting")]
clean-reports:
    rm -rf outputs/notebooks/*
    rm -rf outputs/viz/*
