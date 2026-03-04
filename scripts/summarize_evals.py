import argparse
import json
import os
from pathlib import Path
from collections import defaultdict


def find_eval_set_files(base_path: str, max_depth: int = 2) -> list[str]:
    eval_files = []
    base = Path(base_path)
    for root, dirs, files in os.walk(base):
        rel_path = Path(root).relative_to(base)
        depth = len(rel_path.parts)
        if depth <= max_depth and "eval-set.json" in files:
            eval_files.append(os.path.join(root, "eval-set.json"))
    return eval_files


def parse_eval_set(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        data = json.load(f)
    return data.get("tasks", [])


def shorten_model(model: str) -> str:
    if "/" in model:
        return model.split("/")[-1]
    return model


def main():
    parser = argparse.ArgumentParser(description="Summarize eval results")
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Optional subfolder under logs/ to process (e.g., 'experiment1' -> logs/experiment1)",
    )
    args = parser.parse_args()

    logs_dir = f"logs/{args.folder}" if args.folder else "logs"
    eval_files = find_eval_set_files(logs_dir, max_depth=2)

    folder_data = defaultdict(
        lambda: {
            "models": set(),
            "n_turns": set(),
            "instructions": set(),
            "conditions": set(),
            "tasks": set(),
        }
    )

    for filepath in eval_files:
        folder = os.path.dirname(filepath)
        tasks = parse_eval_set(filepath)

        for task in tasks:
            folder_data[folder]["tasks"].add(task.get("name", "unknown"))
            folder_data[folder]["models"].add(task.get("model", "unknown"))
            folder_data[folder]["n_turns"].add(
                task.get("task_args", {}).get("n_turns", "unknown")
            )
            folder_data[folder]["instructions"].add(
                task.get("task_args", {}).get("instruction_template", "unknown")
            )
            folder_data[folder]["conditions"].add(
                task.get("task_args", {}).get("condition", "unknown")
            )

    all_models = set()
    all_n_turns = set()
    all_instructions = set()
    all_conditions = set()
    all_tasks = set()

    for data in folder_data.values():
        all_models.update(data["models"])
        all_n_turns.update(data["n_turns"])
        all_instructions.update(data["instructions"])
        all_conditions.update(data["conditions"])
        all_tasks.update(data["tasks"])

    def format_set(s):
        return ", ".join(str(x) for x in sorted(s, key=lambda x: (x is None, x)))

    lines = ["# Eval Summary\n"]
    lines.append(f"**Folders:** {len(folder_data)} | **Files:** {len(eval_files)}\n")

    lines.append("## Dimensions Covered\n")
    lines.append(f"- **Tasks:** {format_set(all_tasks)}")
    lines.append(f"- **Models:** {format_set(shorten_model(m) for m in all_models)}")
    lines.append(f"- **N_turns:** {format_set(all_n_turns)}")
    lines.append(f"- **Instructions:** {format_set(all_instructions)}")
    lines.append(f"- **Conditions:** {format_set(all_conditions)}")

    lines.append("\n## Per-Folder Coverage\n")
    lines.append("| Folder | Model | N_turns | Instructions | Conditions |")
    lines.append("|--------|-------|---------|--------------|------------|")

    for folder in sorted(folder_data.keys()):
        data = folder_data[folder]
        folder_short = folder.replace(f"{logs_dir}\\", "").replace(f"{logs_dir}/", "")
        models = format_set(shorten_model(m) for m in data["models"])
        n_turns = format_set(data["n_turns"])
        instructions = format_set(data["instructions"])
        conditions = format_set(data["conditions"])
        lines.append(
            f"| {folder_short} | {models} | {n_turns} | {instructions} | {conditions} |"
        )

    output_path = f"{logs_dir}/eval_summary.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
