"""Rescore existing eval logs with the (now multi-judge) original scorers.

Walks `log_dir` recursively for `.eval` files, reads each log's condition from
task metadata, constructs the appropriate scorer via the existing
`get_behavioral_scorer(condition)` dispatch (which now returns the multi-judge
version), and calls inspect-ai's `score()` with action="overwrite" so the
existing column is updated in place. Column names are preserved.

By default only conditions whose scorer involves an LLM judge are rescored
(LLM-judge, language-detect, and the python/javascript format conditions).
Deterministic scorers produce identical scores, so re-running them costs
nothing useful — pass --all to override.

Usage:
    uv run python scripts/rescore_multijudge.py logs/protocol1
    uv run python scripts/rescore_multijudge.py logs/protocol1 --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env (inspect-ai's own CLI does this, but score() from a script doesn't).
load_dotenv()

# Importing the package registers every @scorer-decorated factory so that
# inspect-ai's registry can look up scorers when we reconstruct them below.
import src.scorers  # noqa: F401, E402
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log

from src.config import CONDITIONS
from src.scorers import get_behavioral_scorer

LLM_JUDGE_SCORER_TYPES: set[str] = {"llm_judge", "language_detect"}
LLM_JUDGE_FORMAT_PREFIXES: tuple[str, ...] = (
    "style_python_javascript",
    "style_javascript_python",
)


def _needs_multi_judge(condition) -> bool:
    if condition.scorer_type in LLM_JUDGE_SCORER_TYPES:
        return True
    if condition.scorer_type == "format_check" and condition.name.startswith(
        LLM_JUDGE_FORMAT_PREFIXES
    ):
        return True
    return False


def main(log_dir: str, dry_run: bool = False, rescore_all: bool = False) -> None:
    root = Path(log_dir)
    log_paths = sorted(root.rglob("*.eval"))
    print(f"Found {len(log_paths)} .eval files under {log_dir}", file=sys.stderr)

    n_rescored = 0
    n_skipped = 0
    n_failed = 0
    for log_path in log_paths:
        try:
            header = read_eval_log(str(log_path), header_only=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! read failed {log_path}: {exc}", file=sys.stderr)
            n_failed += 1
            continue

        task_args = header.eval.task_args or {}
        condition_name = task_args.get("condition")
        if not condition_name or condition_name not in CONDITIONS:
            print(
                f"  ? unknown condition in {log_path}: {condition_name!r}",
                file=sys.stderr,
            )
            n_skipped += 1
            continue
        condition = CONDITIONS[condition_name]

        if not rescore_all and not _needs_multi_judge(condition):
            n_skipped += 1
            continue

        rel = log_path.relative_to(root)
        print(
            f"  rescore {rel}  ({condition_name}, {condition.scorer_type})",
            file=sys.stderr,
        )

        if dry_run:
            n_rescored += 1
            continue

        try:
            full_log = read_eval_log(str(log_path))
            scored = score(
                full_log,
                get_behavioral_scorer(condition),
                action="overwrite",
            )
            write_eval_log(scored, str(log_path))
            n_rescored += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ! rescore failed {log_path}: {exc}", file=sys.stderr)
            n_failed += 1

    print(
        f"Done. rescored={n_rescored} skipped={n_skipped} failed={n_failed}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir", help="Root directory to walk recursively for .eval files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which files would be rescored without actually rescoring",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="rescore_all",
        help="Rescore every condition, not just LLM-judge ones",
    )
    args = parser.parse_args()
    main(args.log_dir, dry_run=args.dry_run, rescore_all=args.rescore_all)
