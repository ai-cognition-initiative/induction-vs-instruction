from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    multi_scorer,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from src.scorers.classify import (
    JUDGE_MODELS,
    _extract_answer,
    classify_language,
    majority_vote_with_agreement,
)


def _judge_scorer(target_language: str, pattern_language: str, judge_model: str) -> Scorer:
    """One judge's language classification, tagged with judge_id for the reducer."""
    judge_id = judge_model.rsplit("/", 1)[-1]

    async def score(state: TaskState, target: Target) -> Score:
        output = _extract_answer(state.output.completion)
        classification = await classify_language(
            output, pattern_language, target_language, model=judge_model
        )

        if classification == "target":
            value = CORRECT
            explanation = f"Judge {judge_id}: detected target language ({target_language})"
        elif classification == "pattern":
            value = INCORRECT
            explanation = f"Judge {judge_id}: detected pattern language ({pattern_language})"
        else:
            value = INCORRECT
            explanation = f"Judge {judge_id}: detected neither target nor pattern language"

        return Score(
            value=value,
            answer=state.output.completion,
            explanation=f"{explanation}. Scored on: '{output}'",
            metadata={
                "judge_id": judge_id,
                "judge_model": judge_model,
                "classification": classification,
            },
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def language_scorer(
    target_language: str,
    pattern_language: str,
) -> Scorer:
    """LLM-judge scorer for language conditions, with multi-judge aggregation.

    Runs every model in JUDGE_MODELS in parallel and reduces to a majority
    vote with agreement statistics in Score.metadata.
    """
    return multi_scorer(
        scorers=[
            _judge_scorer(target_language, pattern_language, m) for m in JUDGE_MODELS
        ],
        reducer=majority_vote_with_agreement,
    )
