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
