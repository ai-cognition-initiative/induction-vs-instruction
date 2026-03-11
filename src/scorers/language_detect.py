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

from src.scorers.classify import _extract_answer, classify_language


@scorer(metrics=[accuracy(), stderr()])
def language_scorer(
    target_language: str,
    pattern_language: str,
) -> Scorer:
    """Score based on whether output is in the target language or pattern language.

    Delegates to classify_language() which uses an LLM judge (single source
    of truth shared with the prediction scorer).
    """

    async def score(state: TaskState, target: Target) -> Score:
        output = _extract_answer(state.output.completion)
        classification = await classify_language(output, pattern_language, target_language)

        if classification == "target":
            return Score(
                value=CORRECT,
                answer=state.output.completion,
                explanation=f"Detected target language ({target_language}). Scored on: '{output}'",
                metadata={"classification": "target"},
            )
        elif classification == "pattern":
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Detected pattern language ({pattern_language}). Scored on: '{output}'",
                metadata={"classification": "pattern"},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Detected neither target nor pattern language. Scored on: '{output}'",
                metadata={"classification": "unknown"},
            )

    return score
