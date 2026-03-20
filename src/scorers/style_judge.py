from __future__ import annotations

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
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

from src.scorers.classify import GRADER_MODEL, PERSONA_RUBRIC, PREFERENCE_RUBRIC, TOPIC_RUBRIC, _extract_answer


@scorer(metrics=[accuracy(), stderr()])
def style_scorer(
    target_description: str,
    pattern_description: str,
    condition_type: str = "persona",
) -> Scorer:
    """LLM judge scorer for persona drift and preference weaving conditions."""

    if condition_type == "preference":
        rubric = PREFERENCE_RUBRIC
    elif condition_type == "variety":
        rubric = TOPIC_RUBRIC
    else:
        rubric = PERSONA_RUBRIC

    async def score(state: TaskState, target: Target) -> Score:
        output = _extract_answer(state.output.completion)

        prompt = rubric.format(
            target_description=target_description,
            pattern_description=pattern_description,
            output=output,
        )

        grader = get_model(GRADER_MODEL)
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
            return Score(
                value=CORRECT,
                answer=state.output.completion,
                explanation=f"Judge classified as TARGET ({target_description}). Scored on: '{output}'",
                metadata={"judge_response": judgment, "classification": "target"},
            )
        elif "PATTERN" in judgment:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Judge classified as PATTERN ({pattern_description}). Scored on: '{output}'",
                metadata={"judge_response": judgment, "classification": "pattern"},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=state.output.completion,
                explanation=f"Judge classified as NEITHER (judge said: {judgment}). Scored on: '{output}'",
                metadata={"judge_response": judgment, "classification": "unknown"},
            )

    return score
