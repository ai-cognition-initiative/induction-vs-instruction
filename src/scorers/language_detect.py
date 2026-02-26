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

from src.scorers.classify import classify_language


@scorer(metrics=[accuracy(), stderr()])
def language_scorer(
    target_language: str,
    pattern_language: str,
) -> Scorer:
    """Score based on whether output is in the target language or pattern language.

    Delegates to classify_language() for the actual detection logic (single source
    of truth shared with the prediction scorer).
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()
        classification = classify_language(output, pattern_language, target_language)

        if classification == "target":
            return Score(
                value=CORRECT,
                answer=output,
                explanation=f"Detected target language ({target_language})",
                metadata={"classification": "target"},
            )
        elif classification == "pattern":
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Detected pattern language ({pattern_language})",
                metadata={"classification": "pattern"},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation="Detected neither target nor pattern language",
                metadata={"classification": "unknown"},
            )

    return score
