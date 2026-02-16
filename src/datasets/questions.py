from __future__ import annotations

import json
import random
from pathlib import Path

DEFAULT_QUESTIONS_PATH = (
    Path(__file__).parent.parent.parent / "data" / "questions_factual.json"
)


def load_questions(path: Path | str | None = None) -> list[str]:
    """Load questions from JSON file.

    Args:
        path: Path to questions JSON file. Defaults to data/questions_factual.json

    Returns:
        List of question strings.
    """
    if path is None:
        path = DEFAULT_QUESTIONS_PATH
    else:
        path = Path(path)

    with open(path) as f:
        data = json.load(f)

    return data["questions"]


def get_question_sequence(
    trial_index: int,
    n_questions: int,
    question_bank: list[str] | None = None,
) -> list[str]:
    """Return a fixed question sequence for a given trial.

    Same trial_index always produces the same sequence,
    regardless of model, condition, or protocol.

    Args:
        trial_index: The trial number (used as seed).
        n_questions: Number of questions to sample.
        question_bank: List of questions to sample from. If None, loads default.

    Returns:
        List of n_questions sampled questions, deterministic for trial_index.
    """
    if question_bank is None:
        question_bank = load_questions()

    rng = random.Random(trial_index)

    if n_questions >= len(question_bank):
        return question_bank.copy()

    return rng.sample(question_bank, n_questions)


def sample_questions(
    n: int,
    questions: list[str] | None = None,
    seed: int | None = None,
) -> list[str]:
    """Sample n questions from the question bank.

    Args:
        n: Number of questions to sample.
        questions: List of questions to sample from. If None, loads default.
        seed: Random seed for reproducibility.

    Returns:
        List of n sampled questions.
    """
    if questions is None:
        questions = load_questions()

    if seed is not None:
        random.seed(seed)

    if n >= len(questions):
        return questions.copy()

    return random.sample(questions, n)
