from __future__ import annotations

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
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
    PERSONA_RUBRIC,
    PREFERENCE_RUBRIC,
    TOPIC_RUBRIC,
    _extract_answer,
    majority_vote_with_agreement,
)


def _judge_scorer(
    target_description: str,
    pattern_description: str,
    condition_type: str,
    judge_model: str,
) -> Scorer:
    """One judge's view on a style/persona/preference/variety condition.

    Tags its Score with `judge_id` in metadata so the reducer can preserve
    per-judge votes.
    """
    if condition_type == "preference":
        rubric = PREFERENCE_RUBRIC
    elif condition_type == "variety":
        rubric = TOPIC_RUBRIC
    else:
        rubric = PERSONA_RUBRIC

    judge_id = judge_model.rsplit("/", 1)[-1]

    async def score(state: TaskState, target: Target) -> Score:
        output = _extract_answer(state.output.completion)

        prompt = rubric.format(
            target_description=target_description,
            pattern_description=pattern_description,
            output=output,
        )

        grader = get_model(judge_model)
        result = await grader.generate(
            [
                ChatMessageSystem(
                    content="You are a precise evaluator. Respond with exactly one word."
                ),
                ChatMessageUser(content=prompt),
            ]
        )

        judgment = result.completion.strip().upper()
        if "TARGET" in judgment:
            value = CORRECT
            classification = "target"
        elif "PATTERN" in judgment:
            value = INCORRECT
            classification = "pattern"
        else:
            value = INCORRECT
            classification = "unknown"

        return Score(
            value=value,
            answer=state.output.completion,
            explanation=(
                f"Judge {judge_id} classified as {classification.upper()}. "
                f"Scored on: '{output}'"
            ),
            metadata={
                "judge_id": judge_id,
                "judge_model": judge_model,
                "classification": classification,
                "judge_response": judgment,
            },
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def style_scorer(
    target_description: str,
    pattern_description: str,
    condition_type: str = "persona",
) -> Scorer:
    """LLM-judge scorer for persona/preference/variety conditions.

    Runs every model in JUDGE_MODELS in parallel and reduces to a majority
    vote with agreement statistics in Score.metadata. With a single judge
    configured the behavior is identical to the original single-grader scorer
    (just with agreement_rate=1.0 in metadata).
    """
    return multi_scorer(
        scorers=[
            _judge_scorer(target_description, pattern_description, condition_type, m)
            for m in JUDGE_MODELS
        ],
        reducer=majority_vote_with_agreement,
    )
