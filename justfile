# Reporting
# Generate visualization notebooks for eval logs

# List available log folders
logs:
    @echo "Available log folders:"
    @ls -1 logs/ 2>/dev/null || echo "  (none)"

# Generate report for a log folder
# Usage: just report <folder> [protocol]
# Example: just report olmo-32b-static behavioral
report folder protocol="behavioral":
    @echo "Generating report for logs/{{folder}} (protocol: {{protocol}})"
    @mkdir -p outputs/viz outputs/notebooks/{{folder}}
    @uv run python scripts/prepare_viz_data.py \
        --log-dir logs/{{folder}} \
        --output outputs/viz/{{folder}}.parquet
    @quarto render notebooks/{{protocol}}_analysis.qmd \
        --output-dir outputs/notebooks/{{folder}} \
        --execute \
        -P data_path:outputs/viz/{{folder}}.parquet
    @echo "Report generated at outputs/notebooks/{{folder}}/"

# Preview a report
preview folder protocol="behavioral":
    @quarto preview notebooks/{{protocol}}_analysis.qmd \
        --execute \
        -P data_path:outputs/viz/{{folder}}.parquet

# Publish report to GitHub Pages
publish folder protocol="behavioral":
    quarto publish gh-pages outputs/notebooks/{{folder}}/{{protocol}}_analysis.html

# Clean generated reports
clean-reports:
    rm -rf outputs/notebooks/*
    rm -rf outputs/viz/*
