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

from src.scorers.classify import GRADER_MODEL

PERSONA_RUBRIC = """You are evaluating whether a response matches a particular communication style.

The response should match the TARGET style: {target_description}
The response should NOT match the PATTERN style: {pattern_description}

Here is the response to evaluate:
---
{output}
---

Does this response match the TARGET style or the PATTERN style?
Answer with exactly one of:
- TARGET: if the response matches the target style
- PATTERN: if the response matches the pattern style
- NEITHER: if the response matches neither style

Respond with just one word: TARGET, PATTERN, or NEITHER."""

PREFERENCE_RUBRIC = """You are evaluating whether a response incorporates a particular preference or theme.

The response SHOULD incorporate: {target_description}
The response should NOT incorporate: {pattern_description}

Here is the response to evaluate:
---
{output}
---

Does this response incorporate the TARGET preference or the PATTERN preference?
Answer with exactly one of:
- TARGET: if the response incorporates the target preference
- PATTERN: if the response incorporates the pattern preference
- NEITHER: if the response incorporates neither preference

Respond with just one word: TARGET, PATTERN, or NEITHER."""


@scorer(metrics=[accuracy(), stderr()])
def style_scorer(
    target_description: str,
    pattern_description: str,
    condition_type: str = "persona",
) -> Scorer:
    """LLM judge scorer for persona drift and preference weaving conditions."""

    rubric = PREFERENCE_RUBRIC if condition_type == "preference" else PERSONA_RUBRIC

    async def score(state: TaskState, target: Target) -> Score:
        output = state.output.completion.strip()

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
                answer=output,
                explanation=f"Judge classified as TARGET ({target_description})",
                metadata={"judge_response": judgment, "classification": "target"},
            )
        elif "PATTERN" in judgment:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Judge classified as PATTERN ({pattern_description})",
                metadata={"judge_response": judgment, "classification": "pattern"},
            )
        else:
            return Score(
                value=INCORRECT,
                answer=output,
                explanation=f"Judge classified as NEITHER (judge said: {judgment})",
                metadata={"judge_response": judgment, "classification": "unknown"},
            )

    return score
