from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset

from src.config import CONDITIONS, DEFAULT_EPOCHS
from src.datasets.sample_builder import build_prediction_sample
from src.scorers.prediction import prediction_scorer
from src.solvers.protocols import prediction_solver


@task
def self_prediction(
    condition: str = "neutral",
    n_turns: int = 5,
    hint: bool = True,
    question_seed: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
) -> Task:
    """Protocol 2: Self-Prediction task.

    Tests whether the model can predict its own behavior before generating.

    Args:
        condition: Name of condition (neutral, value_pattern, value_target, etc.)
        n_turns: Number of hardcoded assistant turns.
        hint: Whether to include the hardcoding hint in instruction.
        question_seed: Random seed for question selection.
        epochs: Number of times to repeat each sample.

    Returns:
        A Task ready for evaluation.
    """
    cond = CONDITIONS.get(condition)
    if cond is None:
        raise ValueError(
            f"Unknown condition: {condition}. Valid options: {list(CONDITIONS.keys())}"
        )

    sample = build_prediction_sample(
        condition=cond,
        n_turns=n_turns,
        hint=hint,
        question_seed=question_seed,
        sample_id=f"{condition}_n{n_turns}_hint{hint}_pred",
    )

    return Task(
        dataset=MemoryDataset(samples=[sample]),
        solver=prediction_solver(),
        scorer=prediction_scorer(),
        epochs=epochs,
        metadata={
            "condition": condition,
            "n_turns": n_turns,
            "hint": hint,
            "question_seed": question_seed,
        },
    )
