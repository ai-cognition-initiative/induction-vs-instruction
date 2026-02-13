from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy(), stderr()])
def pattern_match() -> Scorer:
    """Score based on whether output matches target (T) or pattern (P).

    Returns CORRECT if output contains target (instruction-following).
    Returns INCORRECT if output contains pattern (induction).
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.lower().strip()
        pattern = state.metadata["pattern"].lower()
        target_text = state.metadata["target"].lower()

        pattern_match_score = _fuzzy_match(output, pattern)
        target_match_score = _fuzzy_match(output, target_text)

        if target_match_score > pattern_match_score:
            return Score(
                value=CORRECT,
                answer=state.output.completion,
                explanation=f"Output matched target '{target_text}' (instruction-following)",
            )
        else:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Output matched pattern '{pattern}' (induction)",
            )

    return score


def _fuzzy_match(text: str, pattern: str) -> float:
    """Check if pattern appears in text with fuzzy matching.

    Returns a score from 0 to 1 indicating match quality.
    """
    text = text.lower().strip()
    pattern = pattern.lower().strip()

    if pattern in text:
        return 1.0

    words = pattern.split()
    if len(words) > 1:
        matches = sum(1 for word in words if word in text)
        return matches / len(words)

    return 0.0
