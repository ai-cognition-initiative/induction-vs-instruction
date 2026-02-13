# Reporting commands
# Generate visualization notebooks for eval logs

# Convert Windows backslashes to forward slashes for shell compatibility
root := replace(justfile_directory(), "\\", "/")

[group("reporting")]
logs:
    @echo "Available log folders:"
    @ls -1 logs/ 2>/dev/null || echo "  (none)"

[group("reporting")]
report folder protocol="behavioral":
    @echo "Generating report for logs/{{folder}} (protocol: {{protocol}})"
    @mkdir -p outputs/viz outputs/notebooks/{{folder}}
    @uv run python scripts/prepare_viz_data.py \
        --log-dir logs/{{folder}} \
        --output outputs/viz/{{folder}}.parquet
    @quarto render notebooks/{{protocol}}_analysis.qmd \
        --output-dir {{root}}/outputs/notebooks/{{folder}} \
        --execute \
        -P data_path:{{root}}/outputs/viz/{{folder}}.parquet
    @echo "Report generated at outputs/notebooks/{{folder}}/"

[group("reporting")]
preview folder protocol="behavioral":
    @quarto preview notebooks/{{protocol}}_analysis.qmd \
        --execute \
        -P data_path:{{root}}/outputs/viz/{{folder}}.parquet

[group("reporting")]
publish folder protocol="behavioral":
    quarto publish gh-pages outputs/notebooks/{{folder}}/{{protocol}}_analysis.html

[group("reporting")]
clean-reports:
    rm -rf outputs/notebooks/*
    rm -rf outputs/viz/*
