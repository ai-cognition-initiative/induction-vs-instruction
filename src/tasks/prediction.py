from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from src.config import CONDITIONS, DEFAULT_N_TRIALS
from src.datasets.sample_builder import build_prediction_sample
from src.scorers.prediction import prediction_scorer
from src.solvers.protocols import prediction_solver


@task
def self_prediction(
    condition: str = "neutral",
    n_turns: int = 5,
    instruction_template: str = "instruction_hint",
    n_trials: int = DEFAULT_N_TRIALS,
) -> Task:
    """Protocol 2: Self-Prediction task.

    Tests whether the model can predict its own behavior before generating.

    Args:
        condition: Name of condition (neutral, value_pattern, value_target, etc.)
        n_turns: Number of hardcoded assistant turns.
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
        build_prediction_sample(
            condition=cond,
            n_turns=n_turns,
            instruction_template=instruction_template,
            trial_index=trial_index,
            sample_id=f"{condition}_n{n_turns}_{instruction_template}_trial{trial_index}_pred",
        )
        for trial_index in range(n_trials)
    ]

    return Task(
        dataset=MemoryDataset(samples=samples),
        solver=prediction_solver(),
        scorer=prediction_scorer(),
    )
