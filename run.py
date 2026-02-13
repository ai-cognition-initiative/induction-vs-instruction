"""Run parameterized evaluation grids.

Uses inspect-ai's eval_set() to expand conditions x n_turns into tasks.
This follows the inspect-ai pattern for parameterized grids:
https://inspect.aisi.org.uk/eval-sets.html

Usage:
    uv run python run.py <config.yaml> --model <model> [--log-dir <dir>]

Examples:
    uv run python run.py configs/example.yaml \
        --model openrouter/google/gemini-2.0-flash-001

    uv run python run.py configs/example.yaml \
        --model openrouter/google/gemini-2.0-flash-001,openrouter/anthropic/claude-3.5-haiku \
        --log-dir logs/my-run

For single condition/N runs, use inspect eval directly:
    uv run inspect eval src/tasks/behavioral.py \
        -T condition=neutral -T n_turns=5 \
        --model openrouter/google/gemini-2.0-flash-001
"""
from __future__ import annotations

import argparse
import sys
from itertools import product

import yaml

from src.config import CONDITIONS, DEFAULT_EPOCHS
from src.tasks.behavioral import behavioral_baseline
from src.tasks.prediction import self_prediction

PROTOCOL_MAP = {
    "behavioral": behavioral_baseline,
    "prediction": self_prediction,
}


def main():
    parser = argparse.ArgumentParser(description="Run parameterized evaluation grids")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--model", required=True, help="Model(s), comma-separated")
    parser.add_argument("--log-dir", default=None, help="Log directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    protocol = config["protocol"]
    task_fn = PROTOCOL_MAP.get(protocol)
    if task_fn is None:
        print(f"Unknown protocol: {protocol}. Options: {list(PROTOCOL_MAP.keys())}")
        sys.exit(1)

    conditions = config.get("conditions", ["neutral"])
    n_turns = config.get("n_turns", [5])
    if isinstance(n_turns, int):
        n_turns = [n_turns]
    hint = config.get("hint", True)
    epochs = config.get("epochs", DEFAULT_EPOCHS)
    question_seed = config.get("question_seed", None)

    # Validate
    for c in conditions:
        if c not in CONDITIONS:
            print(f"Unknown condition: {c}. Options: {list(CONDITIONS.keys())}")
            sys.exit(1)

    # Build task grid: conditions x n_turns (inspect-ai eval_set pattern)
    tasks = [
        task_fn(
            condition=condition,
            n_turns=n,
            hint=hint,
            question_seed=question_seed,
            epochs=epochs,
        )
        for condition, n in product(conditions, n_turns)
    ]

    models = [m.strip() for m in args.model.split(",")]
    log_dir = args.log_dir or f"logs/{protocol}"

    print(f"Protocol: {protocol}")
    print(f"Conditions: {conditions}")
    print(f"N values: {n_turns}")
    print(f"Epochs: {epochs}")
    print(f"Tasks: {len(tasks)}")
    print(f"Models: {models}")
    print(f"Log dir: {log_dir}")

    from inspect_ai import eval_set

    success, logs = eval_set(
        tasks=tasks,
        model=models if len(models) > 1 else models[0],
        log_dir=log_dir,
    )

    if not success:
        print("Some tasks failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
