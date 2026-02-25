"""Generate reports from a YAML config file specifying log folders per protocol.

Usage:
    uv run python scripts/generate_report_from_config.py --config configs/reports/my_report.yaml

Config file format:
    report_name: my_combined_report  # Output folder name in outputs/notebooks/

    behavioral:
      log_folders:
        - logs/run1
        - logs/run2

    prediction:
      log_folders:
        - logs/run3

Logic:
    - Only behavioral → render behavioral_analysis.qmd
    - Only prediction → render prediction_analysis.qmd
    - Both behavioral and prediction → render behavioral_vs_prediction_analysis.qmd (combined)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from scripts.prepare_viz_data import (
    prepare_behavioral_multi,
    prepare_combined_multi,
    prepare_prediction_multi,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    if "report_name" not in config:
        raise ValueError("Config must contain 'report_name'")

    has_behavioral = "behavioral" in config and config["behavioral"].get("log_folders")
    has_prediction = "prediction" in config and config["prediction"].get("log_folders")

    if not has_behavioral and not has_prediction:
        raise ValueError(
            "Config must contain at least one of 'behavioral' or 'prediction' with log_folders"
        )

    all_folders = []
    if has_behavioral:
        all_folders.extend(config["behavioral"]["log_folders"])
    if has_prediction:
        all_folders.extend(config["prediction"]["log_folders"])

    missing = [f for f in all_folders if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"Log folders not found: {missing}")


def get_quarto_python() -> str:
    venv_python = Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def render_notebook(
    notebook: str, evals_path: str, output_dir: str, quarto_python: str
) -> None:
    root = Path(__file__).parent.parent
    notebook_path = root / "notebooks" / f"{notebook}.qmd"
    output_path = root / output_dir

    cmd = [
        "quarto",
        "render",
        str(notebook_path),
        "--output-dir",
        str(output_path),
        "--execute",
        f"-P",
        f"evals_path:{evals_path}",
    ]

    env = os.environ.copy()
    env["QUARTO_PYTHON"] = quarto_python

    print(f"Rendering {notebook}.qmd...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error rendering {notebook}.qmd:")
        print(result.stderr)
        sys.exit(1)

    print(f"  Saved to {output_path}/")


def main(config_path: str) -> None:
    root = Path(__file__).parent.parent

    print(f"Loading config from {config_path}")
    config = load_config(config_path)

    validate_config(config)

    report_name = config["report_name"]
    viz_dir = f"outputs/viz/{report_name}"
    notebook_dir = f"outputs/notebooks/{report_name}"

    has_behavioral = "behavioral" in config and config["behavioral"].get("log_folders")
    has_prediction = "prediction" in config and config["prediction"].get("log_folders")

    print(f"Generating report: {report_name}")

    Path(root / viz_dir).mkdir(parents=True, exist_ok=True)
    Path(root / notebook_dir).mkdir(parents=True, exist_ok=True)

    quarto_python = get_quarto_python()

    if has_behavioral and has_prediction:
        print("\nPreparing combined behavioral + prediction data...")
        behavioral_folders = config["behavioral"]["log_folders"]
        prediction_folders = config["prediction"]["log_folders"]

        print(f"  Behavioral logs: {behavioral_folders}")
        print(f"  Prediction logs: {prediction_folders}")

        prepare_combined_multi(behavioral_folders, prediction_folders, viz_dir)

        render_notebook(
            "behavioral_vs_prediction_analysis",
            f"{viz_dir}/evals_combined.parquet",
            notebook_dir,
            quarto_python,
        )

    elif has_behavioral:
        print("\nPreparing behavioral data...")
        folders = config["behavioral"]["log_folders"]
        print(f"  Log folders: {folders}")

        prepare_behavioral_multi(folders, viz_dir)

        render_notebook(
            "behavioral_analysis",
            f"{viz_dir}/evals.parquet",
            notebook_dir,
            quarto_python,
        )

    elif has_prediction:
        print("\nPreparing prediction data...")
        folders = config["prediction"]["log_folders"]
        print(f"  Log folders: {folders}")

        prepare_prediction_multi(folders, viz_dir)

        render_notebook(
            "prediction_analysis",
            f"{viz_dir}/evals_prediction.parquet",
            notebook_dir,
            quarto_python,
        )

    print(f"\nDone. Report at {notebook_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate reports from a YAML config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    main(args.config)
