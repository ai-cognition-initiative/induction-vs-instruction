from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from src.config import CONDITIONS, DEFAULT_N_TRIALS
from src.datasets.sample_builder import build_behavioral_sample
from src.scorers import get_behavioral_scorer
from src.solvers.protocols import behavioral_solver


@task
def behavioral_baseline(
    condition: str = "neutral",
    n_turns: int = 5,
    instruction_template: str = "instruction_hint",
    n_trials: int = DEFAULT_N_TRIALS,
) -> Task:
    """Protocol 1: Behavioral Baseline task.

    Tests whether the model follows instruction (T) or pattern (P)
    under induction pressure.

    Args:
        condition: Name of condition (neutral, value_aligned_cats, value_misaligned_cats, etc.)
        n_turns: Number of hardcoded assistant turns before free generation.
        instruction_template: Name of instruction template file (without .txt).
        n_trials: Number of trials to run (each with different questions).

    Returns:
        A Task ready for evaluation.
    """
    cond = CONDITIONS.get(condition)
    if cond is None:
        raise ValueError(
            f"Unknown condition: {condition}. Valid options: {list(CONDITIONS.keys())}"
        )

    samples = [
        build_behavioral_sample(
            condition=cond,
            n_turns=n_turns,
            instruction_template=instruction_template,
            trial_index=trial_index,
            sample_id=f"{condition}_n{n_turns}_{instruction_template}_trial{trial_index}",
        )
        for trial_index in range(n_trials)
    ]

    return Task(
        dataset=MemoryDataset(samples=samples),
        solver=behavioral_solver(),
        scorer=get_behavioral_scorer(cond),
    )
