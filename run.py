"""Run parameterized evaluation grids.

Uses inspect-ai's eval_set() to expand conditions x n_turns into tasks.
This follows the inspect-ai pattern for parameterized grids:
https://inspect.aisi.org.uk/eval-sets.html

Usage:
    uv run python run.py <config.yaml> --models-yaml <models.yaml> [--eval-set-name <name>] [--log-dir <dir>]

Examples:
    uv run python run.py configs/example.yaml --models-yaml models.yaml

    uv run python run.py configs/example.yaml \
        --models-yaml models.yaml \
        --eval-set-name my-experiment

    uv run python run.py configs/example.yaml \
        --models-yaml models.yaml \
        --log-dir custom-logs/my-run

For OpenRouter, disable reasoning in models.yaml:
    reasoning_enabled: false

For single condition/N runs, use inspect eval directly:
    uv run inspect eval src/tasks/behavioral.py \
        -T condition=neutral -T n_turns=5 \
        --model openrouter/google/gemini-2.0-flash-001
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from itertools import product

import yaml

from src.config import CONDITIONS, DEFAULT_N_TRIALS
from src.tasks.behavioral import behavioral_baseline
from src.tasks.prediction import self_prediction

PROTOCOL_MAP = {
    "behavioral": behavioral_baseline,
    "prediction": self_prediction,
}


def load_models_from_yaml(path: str) -> tuple[list[str], dict]:
    with open(path) as f:
        models_config = yaml.safe_load(f)

    provider = models_config.get("provider", "openrouter")
    models = []
    for model_name in models_config.get("models", []):
        full_model = f"{provider}/{model_name}"
        models.append(full_model)

    model_args = {}
    provider_args = models_config.get("provider_args", {})
    if provider_args:
        model_args["provider"] = provider_args

    models_arg = models_config.get("models")
    if models_arg:
        model_args["models"] = models_arg

    transforms = models_config.get("transforms")
    if transforms:
        model_args["transforms"] = transforms

    reasoning_enabled = models_config.get("reasoning_enabled")
    if reasoning_enabled is not None:
        model_args["reasoning_enabled"] = reasoning_enabled

    return models, model_args


def main():
    parser = argparse.ArgumentParser(description="Run parameterized evaluation grids")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("--models-yaml", required=True, help="Path to models.yaml file")
    parser.add_argument(
        "--eval-set-name",
        default=None,
        help="Name for eval set (logs saved to logs/<name>)",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Explicit log directory (overrides --eval-set-name)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Model temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens (default: 2048)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="none",
        help="Reasoning effort ('none', minimal', 'low', 'medium', 'high', or 'xhigh')",
    )
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
    instruction_templates = config.get("instruction_templates", ["instruction_hint"])
    if isinstance(instruction_templates, str):
        instruction_templates = [instruction_templates]
    n_trials = config.get("n_trials", DEFAULT_N_TRIALS)

    for c in conditions:
        if c not in CONDITIONS:
            print(f"Unknown condition: {c}. Options: {list(CONDITIONS.keys())}")
            sys.exit(1)

    tasks = [
        task_fn(
            condition=condition,
            n_turns=n,
            instruction_template=tmpl,
            n_trials=n_trials,
        )
        for condition, n, tmpl in product(conditions, n_turns, instruction_templates)
    ]

    models, model_args = load_models_from_yaml(args.models_yaml)

    if args.log_dir:
        log_dir = args.log_dir
    elif args.eval_set_name:
        log_dir = f"logs/{args.eval_set_name}"
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"logs/{timestamp}"

    print(f"Protocol: {protocol}")
    print(f"Conditions: {conditions}")
    print(f"N values: {n_turns}")
    print(f"Instruction templates: {instruction_templates}")
    print(f"Trials: {n_trials}")
    print(f"Tasks: {len(tasks)}")
    print(f"Models: {models}")
    if model_args:
        print(f"Model args: {model_args}")
    print(f"Log dir: {log_dir}")

    from inspect_ai import eval_set

    eval_kwargs = {
        "tasks": tasks,
        "model": models,
        "log_dir": log_dir,
        "max_tokens": args.max_tokens,
    }
    if model_args:
        eval_kwargs["model_args"] = model_args
    if args.temperature is not None:
        eval_kwargs["temperature"] = args.temperature
    if args.reasoning_effort is not None:
        eval_kwargs["reasoning_effort"] = args.reasoning_effort

    success, logs = eval_set(**eval_kwargs)

    if not success:
        print("Some tasks failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
