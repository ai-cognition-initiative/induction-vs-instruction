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

from src.scorers.classify import _extract_answer, classify_static


@scorer(metrics=[accuracy(), stderr()])
def pattern_match() -> Scorer:
    """Score based on whether output matches target (T) or pattern (P).

    Returns CORRECT if output contains target (instruction-following).
    Returns INCORRECT if output contains pattern (induction).
    """

    async def score(state: TaskState, target: Target) -> Score:
        output_to_score = _extract_answer(state.output.completion)
        pattern = state.metadata["pattern"]
        target_text = state.metadata["target"]

        choice = classify_static(output_to_score, pattern, target_text)

        if choice == "target":
            return Score(
                value=CORRECT,
                answer=state.output.completion,
                explanation=f"Output matched target '{target_text}' (instruction-following). Scored on: '{output_to_score}'",
                metadata={"classification": "target"},
            )
        elif choice == "pattern":
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Output matched pattern '{pattern}' (induction). Scored on: '{output_to_score}'",
                metadata={"classification": "pattern"},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Output matched neither target '{target_text}' nor pattern '{pattern}'. Scored on: '{output_to_score}'",
                metadata={"classification": "unknown"},
            )

    return score
